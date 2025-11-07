"""Tree generation CLI utilities for DETree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence, Set

from detree.utils.dataset import load_datapath, model_alias_mapping


def _str2bool(value: str) -> bool:
    """Parse common textual boolean representations used by legacy scripts."""

    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {value}")


def get_data_model(data_path: Iterable[Path], has_mix: bool = True) -> Set[str]:
    """Collect all model identifiers present in the provided dataset paths."""

    llm_name: Set[str] = set()
    cnt = 0
    for path in data_path:
        print(f"reading {path}")
        with path.open(mode="r", encoding="utf-8") as jsonl_file:
            for line in jsonl_file:
                now = json.loads(line)
                if now["src"] not in model_alias_mapping:
                    model_alias_mapping[now["src"]] = now["src"]
                now["src"] = model_alias_mapping[now["src"]]
                if not has_mix and "human" in now["src"] and now["src"] != "human":
                    continue
                if now["src"] not in llm_name:
                    llm_name.add(now["src"])
                cnt += 1
    print(cnt)
    return llm_name


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the tree generation CLI."""

    parser = argparse.ArgumentParser(
        description="Generate DETree-compatible tree definitions from dataset files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--path", type=Path, default=Path("/opt/AI-text-Dataset"), help="Root directory of the dataset.")
    parser.add_argument("--dataset_name", type=str, default="all", help="Dataset configuration name.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=("train", "test", "extra"),
        default="train",
        help="Dataset split to consume.",
    )
    parser.add_argument("--tree_txt", type=Path, default=Path("output/Tree_RAID_pcl.txt"), help="Output tree definition path.")
    parser.add_argument("--adversarial", type=_str2bool, default=True, help="Whether to include adversarial data splits.")
    parser.add_argument("--has_mix", type=_str2bool, default=True, help="Whether to keep mixed human/model generations.")
    return parser


def main(args: argparse.Namespace) -> None:
    """Entry point for building DETree-compatible tree structures."""

    dataset_paths: Sequence[str] = load_datapath(args.path, args.adversarial, args.dataset_name)[args.mode]
    print(f"data_path: {dataset_paths}")
    llm_name = sorted(get_data_model((Path(p) for p in dataset_paths), args.has_mix))
    root = len(llm_name)
    args.tree_txt.parent.mkdir(parents=True, exist_ok=True)
    with args.tree_txt.open("w", encoding="utf-8") as f:
        for i, item in enumerate(llm_name):
            f.write(f"{i} {root} {item}\n")
        f.write(f"{root} -1 none\n")


if __name__ == "__main__":
    parser = build_argument_parser()
    main(parser.parse_args())
