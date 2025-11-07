"""Embedding generation CLI for DETree."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Literal, Optional

import pandas as pd
import torch
import torch.nn.functional as F
from lightning import Fabric
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from detree.model.text_embedding import TextEmbeddingModel
from detree.utils.dataset import SCLDataset, load_datapath


def infer(passages_dataloader, fabric, tokenizer, model, args):
    if fabric.global_rank == 0:
        passages_dataloader = tqdm(passages_dataloader)
        all_ids, all_embeddings, all_labels = [], {}, []
        for layer in args.need_layer:
            all_embeddings[layer] = []
    with torch.no_grad():
        for batch in passages_dataloader:
            text, label, write_model, ids = batch
            encoded_batch = tokenizer.batch_encode_plus(
                text,
                return_tensors="pt",
                max_length=args.max_length,
                padding="max_length",
                truncation=True,
            )
            encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
            embeddings = model(encoded_batch, hidden_states=True)
            embeddings = fabric.all_gather(embeddings).view(-1, embeddings.size(-2), embeddings.size(-1))
            label = fabric.all_gather(write_model).view(-1)
            ids = fabric.all_gather(ids).view(-1)
            if fabric.global_rank == 0:
                embeddings = F.normalize(embeddings, dim=-1).cpu().to(torch.bfloat16)
                for layer in args.need_layer:
                    all_embeddings[layer].append(embeddings[:, layer, :].clone())
                all_ids.extend(ids.cpu().tolist())
                all_labels.extend(label.cpu().tolist())
            del embeddings, label, ids
    if fabric.global_rank == 0:
        for layer in args.need_layer:
            all_embeddings[layer] = torch.cat(all_embeddings[layer], dim=0)
        return torch.tensor(all_ids), all_embeddings, torch.tensor(all_labels)
    return [], [], []


def stable_long_hash(input_string: str) -> int:
    import hashlib

    hash_object = hashlib.sha256(input_string.encode())
    hex_digest = hash_object.hexdigest()
    int_hash = int(hex_digest, 16)
    return int_hash & ((1 << 63) - 1)


def load_data(split: Literal["train", "test", "extra"], include_adversarial: bool, fp: Path) -> pd.DataFrame:
    if split not in ("train", "test", "extra"):
        raise ValueError("`split` must be one of (\"train\", \"test\", \"extra\")")

    fname = f"{split}.csv" if include_adversarial else f"{split}_none.csv"
    fp = fp / fname
    return pd.read_csv(fp)


class PassagesDataset(Dataset):
    def __init__(self, data):
        self.passages = []
        for item in data:
            if item["attack"] not in ("none", "paraphrase") and stable_long_hash(item["generation"]) % 10 < 5:
                continue
            self.passages.append(item)
        classes = sorted({item["model"] for item in data})
        self.classes = list(classes)
        self.human_id = self.classes.index("human")

    def __len__(self):
        return len(self.passages)

    def __getitem__(self, idx):
        data_now = self.passages[idx]
        text = data_now["generation"]
        model = self.classes.index(data_now["model"])
        label = int(model == self.human_id)
        ids = stable_long_hash(text)
        return text, int(label), int(model), int(ids)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate embedding databases for DETree evaluators",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device-num", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)

    parser.add_argument("--path", type=Path, required=True, help="Dataset root directory or JSONL file path.")
    parser.add_argument("--database-name", type=str, default="M4_monolingual")
    parser.add_argument(
        "--model-name",
        type=str,
        default="FacebookAI/roberta-large",
        help=(
            "Model identifier for embeddings generation. Accepts either a Hugging Face "
            "model hub name or a local path to a directory in Hugging Face format."
        ),
    )

    parser.add_argument("--pooling", type=str, default="max", choices=("max", "average", "cls"))
    parser.add_argument("--need-layer", type=int, nargs="+", default=[16, 17, 18, 19, 22, 23])

    parser.add_argument("--adversarial", dest="adversarial", action="store_true")
    parser.add_argument("--no-adversarial", dest="adversarial", action="store_false")
    parser.set_defaults(adversarial=True)

    parser.add_argument("--has-mix", dest="has_mix", action="store_true")
    parser.add_argument("--no-has-mix", dest="has_mix", action="store_false")
    parser.set_defaults(has_mix=False)

    parser.add_argument("--savedir", type=Path, required=True, help="Output directory for the embedding database.")
    parser.add_argument("--name", type=str, required=True, help="Filename (without extension) for the saved embeddings.")
    parser.add_argument("--split", type=str, default="train", choices=("train", "test", "extra"))

    return parser


def generate_embeddings(args: argparse.Namespace) -> None:
    if args.device_num > 1:
        fabric = Fabric(accelerator="cuda", precision="bf16-mixed", devices=args.device_num, strategy="ddp")
    else:
        fabric = Fabric(accelerator="cuda", precision="bf16-mixed", devices=args.device_num)
    fabric.launch()

    model = TextEmbeddingModel(
        args.model_name,
        output_hidden_states=True,
        infer=True,
        use_pooling=args.pooling,
    ).cuda()
    tokenizer = model.tokenizer
    model.eval()

    path_str = str(args.path)
    if "LLM_detect_data" in path_str:
        now_data = load_data(args.split, include_adversarial=args.adversarial, fp=args.path)
        now_data = now_data.to_dict(orient="records")
        dataset = PassagesDataset(now_data)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
        dataloader = fabric.setup_dataloaders(dataloader)
    elif path_str.endswith(".jsonl"):
        dataset = SCLDataset([path_str], fabric, tokenizer, need_ids=True, adv_p=0)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
        dataloader = fabric.setup_dataloaders(dataloader, use_distributed_sampler=False)
    else:
        data_path = load_datapath(
            path_str,
            include_adversarial=args.adversarial,
            dataset_name=args.database_name,
        )[args.split]
        dataset = SCLDataset(data_path, fabric, tokenizer, need_ids=True, adv_p=0, has_mix=args.has_mix)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
        dataloader = fabric.setup_dataloaders(dataloader, use_distributed_sampler=False)

    model = fabric.setup(model)
    classes = dataset.classes
    train_ids, train_embeddings, train_labels = infer(dataloader, fabric, tokenizer, model, args)

    torch.cuda.empty_cache()
    if fabric.global_rank == 0:
        args.savedir.mkdir(parents=True, exist_ok=True)
        emb_dict = {
            "embeddings": train_embeddings,
            "labels": train_labels,
            "ids": train_ids,
            "classes": classes,
        }
        output_path = args.savedir / f"{args.name}.pt"
        torch.save(emb_dict, output_path)
        print(f"Saved embedding database to {output_path}")


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    generate_embeddings(args)


if __name__ == "__main__":
    main()

__all__ = ["build_argument_parser", "generate_embeddings", "main"]
