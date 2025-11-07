import argparse
import json
import logging
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from tqdm import tqdm

from .binoculars_detector import Binoculars
from ..utils import evaluate_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_LOG_PATH = Path(__file__).resolve().parents[3] / "runs" / "val-other_detector.txt"
_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

def load_jsonl(file_path):
    out = []
    with open(file_path, mode='r', encoding='utf-8') as jsonl_file:
        for line in jsonl_file:
            item = json.loads(line)
            out.append(item)
    print(f"Loaded {len(out)} examples from {file_path}")
    return out

def dict2str(metrics):
    out_str=''
    for key in metrics.keys():
        out_str+=f"{key}:{metrics[key]} "
    return out_str

def experiment(args):
    # Initialize Binoculars (experiments in paper use the "accuracy" mode threshold wherever applicable)
    bino = Binoculars(mode="accuracy", max_token_observed=args.tokens_seen)

    logging.info(f"Test in {args.test_data_path}")
    test_data = load_jsonl(args.test_data_path)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.shuffle(test_data)
    predictions = []
    labels = []
    for i, item in tqdm(enumerate(test_data), total=len(test_data)):
        text = item["text"]
        label = item["label"]
        src = item["src"]
        bino_score = -bino.compute_score(text)
        
        if bino_score is None or np.isnan(bino_score) or np.isinf(bino_score):
            bino_score = 0
        if 'human' in src:
            labels.append(0)
        else:
            labels.append(1)
        predictions.append(bino_score)
    metric = evaluate_metrics(labels, predictions)
    print(dict2str(metric))
    with _LOG_PATH.open("a+", encoding="utf-8") as f:
        f.write(f"binoculars  {args.test_data_path}\n")
        f.write(f"{dict2str(metric)}\n")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--test_data_path',
        type=str,
        default='/path/to/RealBench/Deepfake/no_attack/test.jsonl',
        help="Path to the test data. could be several files with ','. Note that the data should have been perturbed.",
    )
    parser.add_argument("--tokens_seen", type=int, default=512, help="Number of tokens seen by the model")
    parser.add_argument('--DEVICE', default="cuda", type=str, required=False)
    parser.add_argument('--seed', default=2023, type=int, required=False)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    experiment(args)


if __name__ == '__main__':
    main()
