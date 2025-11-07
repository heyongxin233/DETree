"""kNN evaluation against raw text datasets."""

from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from lightning import Fabric
from torch.nn.functional import softmax as F_softmax
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from detree.model.text_embedding import TextEmbeddingModel
from detree.utils.index import Indexer
from detree.utils.utils import evaluate_metrics

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")


def load_jsonl(file_path: Path) -> List[dict]:
    out = []
    with file_path.open(mode="r", encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            item = json.loads(line)
            out.append(item)
    print(f"Loaded {len(out)} examples from {file_path}")
    return out


class PassagesDataset(Dataset):
    def __init__(self, data: Sequence[dict]):
        self.passages = list(data)

    def __len__(self) -> int:
        return len(self.passages)

    def __getitem__(self, idx: int):
        data_now = self.passages[idx]
        text = data_now["text"]
        label = data_now["label"]
        ids = data_now["id"]
        return text, int(label), int(ids)


def infer(passages_dataloader, fabric, tokenizer, model, max_length: int = 512):
    if fabric.global_rank == 0:
        passages_dataloader = tqdm(passages_dataloader)
        all_ids: List[int] = []
        all_embeddings: List[torch.Tensor] = []
        all_labels: List[int] = []
    with torch.no_grad():
        for batch in passages_dataloader:
            text, label, ids = batch
            encoded_batch = tokenizer.batch_encode_plus(
                text,
                return_tensors="pt",
                max_length=max_length,
                padding="max_length",
                truncation=True,
            )
            encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
            embeddings = model(encoded_batch, hidden_states=True)
            embeddings = fabric.all_gather(embeddings).view(-1, embeddings.size(-2), embeddings.size(-1))
            label = fabric.all_gather(label).view(-1)
            ids = fabric.all_gather(ids).view(-1)
            if fabric.global_rank == 0:
                all_embeddings.append(embeddings.cpu())
                all_ids.extend(ids.cpu().tolist())
                all_labels.extend(label.cpu().tolist())
    if fabric.global_rank == 0:
        embeddings_tensor = torch.cat(all_embeddings, dim=0)
        embeddings_tensor = F.normalize(embeddings_tensor, dim=-1).permute(1, 0, 2)
        return all_ids, embeddings_tensor.numpy(), all_labels
    return [], [], []


def save_pt(train_embeddings, all_labels, train_ids, args, best_layer):
    save_layer = [best_layer, train_embeddings.shape[0] - 1]
    all_embeddings = {i: torch.tensor(train_embeddings[i]) for i in save_layer}
    emb_dict = {
        "embeddings": all_embeddings,
        "labels": torch.tensor(all_labels),
        "ids": torch.tensor(train_ids),
        "classes": ["llm", "human"],
    }
    args.savedir.mkdir(parents=True, exist_ok=True)
    output_path = args.savedir / f"{args.name}.pt"
    torch.save(emb_dict, output_path)
    print(f"Saved embedding snapshot to {output_path}")


def dict2str(metrics: dict) -> str:
    out_str = ""
    if "layer" in metrics:
        out_str += f"layer:{metrics['layer']} "
    if "k" in metrics:
        out_str += f"k:{metrics['k']} "
    for key, value in metrics.items():
        if key not in {"layer", "k"}:
            out_str += f"{key}:{value} "
    return out_str.strip()


def process_element(args: Tuple[Sequence[int], Sequence[float], Sequence[int], float]):
    ids, scores, labels, temperature = args
    now_score = torch.zeros(2)
    sorted_indices = np.argsort(scores)[::-1]
    element_preds = {}

    for k, idx in enumerate(sorted_indices):
        label = labels[idx]
        now_score[label] += scores[idx] * temperature
        prob = F_softmax(now_score, dim=-1)[1].item()
        element_preds[k + 1] = prob

    return element_preds


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate DETree checkpoints using a kNN classifier over hidden states.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device-num", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)

    parser.add_argument("--database-path", type=Path, required=True, help="Training set JSONL file.")
    parser.add_argument("--test-dataset-path", type=Path, required=True, help="Evaluation set JSONL file.")
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        required=True,
        help="Model identifier from Hugging Face or local path to a merged checkpoint.",
    )
    parser.add_argument("--temperature", type=float, default=0.05)

    parser.add_argument("--max-k", type=int, default=50, dest="max_K", help="Maximum k to evaluate for kNN.")
    parser.add_argument("--min-layer", type=int, default=15, help="Minimum hidden layer index to evaluate.")
    parser.add_argument("--pooling", type=str, default="max", choices=("max", "average", "cls"))

    parser.add_argument("--embedding-dim", type=int, default=1024)
    parser.add_argument("--n-subquantizers", type=int, default=1)
    parser.add_argument("--n-bits", type=int, default=8)

    parser.add_argument("--savedir", type=Path, default=Path("runs"))
    parser.add_argument("--name", type=str, default="database_knn_eval")
    parser.add_argument("--pool-workers", type=int, default=min(32, cpu_count()))
    parser.add_argument("--save-embeddings", action="store_true", help="Persist embeddings for the best-performing layer.")
    parser.add_argument("--log-file", type=Path, default=Path("runs/val.txt"))

    return parser


def evaluate(args: argparse.Namespace) -> None:
    if args.device_num > 1:
        fabric = Fabric(accelerator="cuda", devices=args.device_num, strategy="ddp", precision="bf16-mixed")
    else:
        fabric = Fabric(accelerator="cuda", devices=args.device_num, precision="bf16-mixed")
    fabric.launch()

    model = TextEmbeddingModel(
        args.model_name_or_path,
        output_hidden_states=True,
        infer=True,
        use_pooling=args.pooling,
    ).cuda()
    tokenizer = model.tokenizer
    model.eval()

    database = load_jsonl(args.database_path)
    test_database = load_jsonl(args.test_dataset_path)

    passages_dataset = PassagesDataset(database)
    test_dataset = PassagesDataset(test_database)

    passages_dataloader = DataLoader(
        passages_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
    )

    passages_dataloader, test_dataloader = fabric.setup_dataloaders(passages_dataloader, test_dataloader)
    model = fabric.setup(model)

    train_ids, train_embeddings, train_labels = infer(passages_dataloader, fabric, tokenizer, model, args.max_length)
    test_ids, test_embeddings, test_labels = infer(test_dataloader, fabric, tokenizer, model, args.max_length)

    torch.cuda.empty_cache()
    if fabric.global_rank != 0:
        return

    layer_num = train_embeddings.shape[0]
    test_labels = [int(label) for label in test_labels]

    label_dict = {train_ids[i]: int(train_labels[i]) for i in range(len(train_ids))}

    all_details = []
    index = Indexer(args.embedding_dim, args.n_subquantizers, args.n_bits)
    index.label_dict = label_dict

    with Pool(processes=args.pool_workers) as pool:
        for i in range(args.min_layer, layer_num):
            now_best_metrics = None
            index.reset()
            index.index_data(train_ids, train_embeddings[i])
            preds = {k: [] for k in range(1, args.max_K + 1)}
            top_ids_and_scores = index.search_knn(test_embeddings[i], args.max_K, index_batch_size=128)

            args_list = [
                (ids, scores, labels, args.temperature)
                for ids, scores, labels in top_ids_and_scores
            ]
            for result in tqdm(pool.imap(process_element, args_list), total=len(args_list)):
                for k, value in result.items():
                    preds[k].append(value)

            for k in range(2, args.max_K + 1):
                metric = evaluate_metrics(test_labels, preds[k], threshold_param=-1)
                if now_best_metrics is None or now_best_metrics["auroc"] < metric["auroc"]:
                    now_best_metrics = metric
                    now_best_metrics["k"] = k
                    now_best_metrics["layer"] = i

            if now_best_metrics:
                print(dict2str(now_best_metrics))
                all_details.append(now_best_metrics)

    if not all_details:
        return

    max_ids = max(range(len(all_details)), key=lambda idx: all_details[idx]["auroc"])
    best_metrics = all_details[max_ids]

    if args.save_embeddings:
        save_pt(train_embeddings, train_labels, train_ids, args, best_metrics["layer"])

    print("Best " + dict2str(best_metrics))
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    with args.log_file.open("a+", encoding="utf-8") as fp:
        fp.write(
            f"test model:{args.model_name_or_path} database_path:{args.database_path} mode:{args.test_dataset_path}\n"
        )
        fp.write(f"Last {dict2str(all_details[-1])}\n")
        fp.write(f"Best {dict2str(best_metrics)}\n")
        fp.write("------------------------------------------\n")


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    evaluate(args)


if __name__ == "__main__":
    main()

__all__ = ["build_argument_parser", "evaluate", "main"]
