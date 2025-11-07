import argparse
import json
import logging
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from tqdm import tqdm

from ..utils import evaluate_metrics

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
    # Initialize RADAR detector model
    detector = transformers.AutoModelForSequenceClassification.from_pretrained("TrustSafeAI/RADAR-Vicuna-7B",device_map="auto")
    tokenizer = transformers.AutoTokenizer.from_pretrained("TrustSafeAI/RADAR-Vicuna-7B")
    detector.eval()

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
        
        # Tokenize input text
        inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Get model output
        with torch.no_grad():
            output_probs = F.log_softmax(detector(**inputs).logits, -1)[:, 0].exp().tolist()

        # Determine the label and append to predictions and labels
        if 'human' in src:
            labels.append(0)
        else:
            labels.append(1)

        predictions.append(output_probs[0])  # Probabilities for AI-generated text
    
    # Compute metrics
    metric = evaluate_metrics(labels, predictions)
    print(dict2str(metric))
    
    # Save results
    with _LOG_PATH.open("a+", encoding="utf-8") as f:
        f.write(f"RADAR  {args.test_data_path}\n")
        f.write(f"{dict2str(metric)}\n")

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--test_data_path',
        type=str,
        default='/path/to/RealBench/DetectRL/Multi_Domain/all_multi_domains_test.jsonl',
        help="Path to the test data. could be several files with ','. Note that the data should have been perturbed.",
    )
    parser.add_argument('--seed', default=2023, type=int, required=False)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    experiment(args)


if __name__ == '__main__':
    main()
