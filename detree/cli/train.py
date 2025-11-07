"""Training CLI for DETree."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.nn.functional as F  # noqa: F401  # retained for backward compat with downstream imports
import torch.optim as optim
import yaml
from lightning import Fabric
from lightning.fabric.strategies import DeepSpeedStrategy, DDPStrategy
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer

from detree.model.simclr import SimCLR_Tree
from detree.utils.dataset import SCLDataset, load_datapath


@dataclass
class ExperimentPaths:
    """Utility container describing where to store experiment artefacts."""

    root: Path
    runs: Path


def _build_collate_fn(tokenizer, max_length: int):
    def collate_fn(batch: Iterable):
        text, label, write_model = default_collate(batch)
        encoded_batch = tokenizer.batch_encode_plus(
            text,
            return_tensors="pt",
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return encoded_batch, label, write_model

    return collate_fn


def _prepare_output_dir(
    output_dir: Path, experiment_name: str, resume: bool, *, create_dirs: bool = True
) -> ExperimentPaths:
    output_dir = output_dir.expanduser().resolve()

    candidate = output_dir / experiment_name
    if candidate.exists() and not resume:
        suffix = 0
        while (output_dir / f"{experiment_name}_v{suffix}").exists():
            suffix += 1
        candidate = output_dir / f"{experiment_name}_v{suffix}"

    runs_dir = candidate / "runs"
    if create_dirs:
        candidate.mkdir(parents=True, exist_ok=True)
        runs_dir.mkdir(parents=True, exist_ok=True)

    return ExperimentPaths(root=candidate, runs=runs_dir)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train DETree using the hierarchical contrastive objective",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-name", type=str, default="FacebookAI/roberta-large", help="Backbone encoder identifier.")
    parser.add_argument("--device-num", type=int, default=1, help="Number of CUDA devices to use.")
    parser.add_argument("--path", type=Path, required=True, help="Root directory of the dataset.")
    parser.add_argument("--dataset-name", type=str, default="all", help="Dataset configuration name.")
    parser.add_argument(
        "--dataset", type=str, default="train", choices=("train", "test", "extra"), help="Dataset split to consume."
    )
    parser.add_argument("--tree-txt", type=Path, required=True, help="Tree definition file as produced by the HAT pipeline.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs"), help="Directory where experiment folders are saved.")
    parser.add_argument("--experiment-name", type=str, default="detree_experiment", help="Base name for the run directory.")
    parser.add_argument("--resume", action="store_true", help="Reuse the given experiment directory if it already exists.")

    parser.add_argument("--projection-size", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--per-gpu-batch-size", type=int, default=64)
    parser.add_argument("--per-gpu-eval-batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length for the tokenizer.")
    parser.add_argument("--total-epoch", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--min-lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--adv-p", type=float, default=0.5, help="Probability of sampling adversarial data.")
    parser.add_argument("--num-workers-eval", type=int, default=8, help="Reserved for compatibility.")

    parser.add_argument("--lora-r", type=int, default=128)
    parser.add_argument("--lora-alpha", type=int, default=256)
    parser.add_argument("--lora-dropout", type=float, default=0.0)

    parser.add_argument("--freeze-layer", type=int, default=0, help="Number of initial encoder layers to freeze.")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--adapter-path", type=Path, default=None, help="Optional path to resume LoRA training from.")
    parser.add_argument("--pooling", type=str, default="max", choices=("max", "average", "cls"))

    parser.add_argument("--lora", dest="lora", action="store_true", help="Enable LoRA adapters.")
    parser.add_argument("--no-lora", dest="lora", action="store_false", help="Disable LoRA adapters.")
    parser.set_defaults(lora=True)

    parser.add_argument("--freeze-embedding-layer", dest="freeze_embedding_layer", action="store_true")
    parser.add_argument("--no-freeze-embedding-layer", dest="freeze_embedding_layer", action="store_false")
    parser.set_defaults(freeze_embedding_layer=True)

    parser.add_argument("--adversarial", dest="adversarial", action="store_true")
    parser.add_argument("--no-adversarial", dest="adversarial", action="store_false")
    parser.set_defaults(adversarial=True)

    parser.add_argument("--include-attack", dest="include_attack", action="store_true")
    parser.add_argument("--no-include-attack", dest="include_attack", action="store_false")
    parser.set_defaults(include_attack=True)

    parser.add_argument("--has-mix", dest="has_mix", action="store_true")
    parser.add_argument("--no-has-mix", dest="has_mix", action="store_false")
    parser.set_defaults(has_mix=True)

    parser.add_argument("--deepspeed", action="store_true", help="Use DeepSpeed strategy when multiple GPUs are available.")

    return parser


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_float32_matmul_precision("medium")

    if args.device_num > 1:
        if args.deepspeed:
            strategy = DeepSpeedStrategy()
        else:
            strategy = DDPStrategy(find_unused_parameters=True)
        fabric = Fabric(accelerator="cuda", precision="bf16-mixed", devices=args.device_num, strategy=strategy)
    else:
        fabric = Fabric(accelerator="cuda", precision="bf16-mixed", devices=args.device_num)

    fabric.launch()

    experiment_paths = ExperimentPaths(root=Path(args.output_dir), runs=Path(args.runs_dir))
    if fabric.global_rank == 0:
        experiment_paths.root.mkdir(parents=True, exist_ok=True)
        experiment_paths.runs.mkdir(parents=True, exist_ok=True)
    fabric.barrier()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    collate_fn = _build_collate_fn(tokenizer, args.max_length)

    model = SimCLR_Tree(args, fabric).train()

    data_path = load_datapath(
        str(args.path),
        include_adversarial=args.adversarial,
        dataset_name=args.dataset_name,
        include_attack=args.include_attack,
    )[args.dataset]

    train_dataset = SCLDataset(
        data_path,
        fabric,
        tokenizer,
        name2id=model.names2id,
        has_mix=args.has_mix,
        adv_p=args.adv_p,
    )

    passages_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_gpu_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    model.train()
    if args.freeze_embedding_layer:
        for name, param in model.model.named_parameters():
            if "emb" in name or "model.pooler" in name:
                param.requires_grad = False
            if args.freeze_layer > 0:
                for i in range(args.freeze_layer):
                    if f"encoder.layer.{i}." in name:
                        param.requires_grad = False

    model = torch.compile(model)
    if fabric.global_rank == 0:
        print("Model has been initialized!")
        for name, param in model.model.named_parameters():
            print(name, param.requires_grad)

    passages_dataloader = fabric.setup_dataloaders(passages_dataloader, use_distributed_sampler=False)
    if fabric.global_rank == 0:
        print("DataLoader has been initialized!")

    if fabric.global_rank == 0:
        writer = SummaryWriter(str(experiment_paths.runs))
        print(f"Save dir is {args.output_dir}")
        opt_dict = vars(args)
        opt_dict["output_dir"] = str(args.output_dir)
        with open(Path(args.output_dir) / "config.yaml", "w", encoding="utf-8") as file:
            yaml.dump(opt_dict, file, sort_keys=False)
    else:
        writer = None

    experiment_dir = experiment_paths.root

    num_batches_per_epoch = len(passages_dataloader)
    warmup_steps = args.warmup_steps
    lr = args.lr
    total_steps = args.total_epoch * num_batches_per_epoch - warmup_steps

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps, eta_min=args.min_lr)
    model, optimizer = fabric.setup(model, optimizer)

    if fabric.global_rank == 0:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.requires_grad)

    for epoch in range(args.total_epoch):
        model.train()
        avg_loss = 0.0
        iterator = enumerate(passages_dataloader)
        if fabric.global_rank == 0:
            iterator = tqdm(iterator, total=len(passages_dataloader))
            print(("\n" + "%11s" * 5) % ("Epoch", "GPU_mem", "loss1", "Avgloss", "lr"))
        for i, batch in iterator:
            current_step = epoch * num_batches_per_epoch + i
            if current_step < warmup_steps:
                current_lr = lr * current_step / max(warmup_steps, 1)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr
            current_lr = optimizer.param_groups[0]["lr"]

            encoded_batch, label, write_model = batch
            loss, loss_classify = model(encoded_batch, write_model)

            avg_loss = (avg_loss * i + loss.item()) / (i + 1)
            fabric.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            if current_step >= warmup_steps:
                schedule.step()

            mem = f"{torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0:.3g}G"
            if fabric.global_rank == 0:
                iterator.set_description(
                    ("%11s" * 2 + "%11.4g" * 3)
                    % (f"{epoch + 1}/{args.total_epoch}", mem, loss_classify.item(), avg_loss, current_lr)
                )
                if writer and current_step % 10 == 0:
                    writer.add_scalar("lr", current_lr, current_step)
                    writer.add_scalar("loss", loss.item(), current_step)
                    writer.add_scalar("avg_loss", avg_loss, current_step)
                    writer.add_scalar("loss_classify", loss_classify.item(), current_step)

        if fabric.global_rank == 0:
            checkpoint_dir = experiment_dir / f"epoch_{epoch:02d}"
            model.save_pretrained(str(checkpoint_dir), save_tokenizer=(epoch == 0))
            print(f"Saved adapter checkpoint to {checkpoint_dir}", flush=True)

            last_dir = experiment_dir / "last"
            model.save_pretrained(str(last_dir), save_tokenizer=False)
            print(f"Updated latest checkpoint at {last_dir}", flush=True)

        fabric.barrier()

    if writer:
        writer.flush()
        writer.close()


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    experiment_paths = _prepare_output_dir(
        args.output_dir, args.experiment_name, resume=args.resume, create_dirs=False
    )
    args.output_dir = str(experiment_paths.root)
    args.runs_dir = str(experiment_paths.runs)
    train(args)


__all__ = ["build_argument_parser", "main", "train"]


if __name__ == "__main__":
    main()
