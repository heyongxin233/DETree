"""Utility script to run DETree kNN inference against a saved database."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Optional

# Ensure the ``detree`` package is importable when running from the repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from detree.inference import Detector  # noqa: E402


def _load_inputs(args: argparse.Namespace) -> List[str]:
    texts: List[str] = []
    if args.text:
        texts.extend(args.text)
    if args.input_file:
        input_path = Path(args.input_file)
        if input_path.suffix == ".jsonl":
            with input_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if "text" not in record:
                        raise ValueError("JSONL entries must contain a 'text' field")
                    texts.append(record["text"])
        else:
            with input_path.open("r", encoding="utf-8") as handle:
                texts.extend([line.strip() for line in handle if line.strip()])
    return texts


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DETree kNN inference.")
    parser.add_argument("--database-path", type=Path, required=True)
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--text", action="append", help="Direct text input (repeatable).")
    parser.add_argument("--input-file", type=str, help="File with one example per line or JSONL with a 'text' field.")
    parser.add_argument("--output", type=Path, help="Optional JSON file to store predictions.")
    parser.add_argument("--pooling", type=str, default="max")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.97)
    parser.add_argument("--layer", type=int, help="Layer index to use from the database.")
    parser.add_argument("--device", type=str, help="Override torch device (e.g. 'cpu').")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    texts = _load_inputs(args)
    if not texts:
        raise ValueError("No input text provided via --text or --input-file.")

    detector = Detector(
        database_path=args.database_path,
        model_name_or_path=args.model_name_or_path,
        pooling=args.pooling,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        top_k=args.top_k,
        threshold=args.threshold,
        layer=args.layer,
        device=args.device,
    )

    predictions = detector.predict(texts)
    for prediction in predictions:
        print(f"Text: {prediction.text}")
        print(
            "Prediction: {label} (人工概率={p_human:.4f}, AI 概率={p_ai:.4f})".format(
                label=prediction.label,
                p_human=prediction.probability_human,
                p_ai=prediction.probability_ai,
            )
        )
        print("-" * 40)

    if args.output:
        payload = [prediction.__dict__ for prediction in predictions]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
