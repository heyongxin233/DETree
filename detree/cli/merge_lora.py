"""Merge LoRA adapters into base models."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

from peft import PeftModel
from transformers import AutoModel, AutoTokenizer


def merge_lora_adapter(base_model: str, adapter_path: Path, output_dir: Path, safe_serialization: bool = True) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModel.from_pretrained(base_model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    peft_model = PeftModel.from_pretrained(model, str(adapter_path))
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(output_dir, safe_serialization=safe_serialization)
    tokenizer.save_pretrained(output_dir)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into its base Hugging Face model.")
    parser.add_argument("--base-model", type=str, required=True, help="Base model name or path.")
    parser.add_argument("--adapter-path", type=Path, required=True, help="Directory containing the LoRA adapter weights.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to store the merged model.")
    parser.add_argument(
        "--no-safe-serialization",
        action="store_true",
        help="Disable safetensors when saving the merged model.",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    merge_lora_adapter(
        args.base_model,
        args.adapter_path,
        args.output_dir,
        safe_serialization=not args.no_safe_serialization,
    )


if __name__ == "__main__":
    main()

__all__ = ["build_argument_parser", "merge_lora_adapter", "main"]
