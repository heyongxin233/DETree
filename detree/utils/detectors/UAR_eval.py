import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np; np.random.seed(43)
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from ..utils import evaluate_metrics
os.environ["TOKENIZERS_PARALLELISM"] = "true"

_LOG_PATH = Path(__file__).resolve().parents[3] / "runs" / "val-other_detector.txt"
_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
class PassagesDataset(Dataset):
    def __init__(self, data):
        
        self.passages = data

    def __len__(self):
        return len(self.passages)
         
    def __getitem__(self, idx):
        data_now = self.passages[idx]
        text = data_now['text']
        label = int(data_now['label'])==0
        ids = data_now['id']
        return text, int(label), int(ids)

def load_jsonl(file_path,need_human=True):
    out = []
    with open(file_path, mode='r', encoding='utf-8') as jsonl_file:
        for line in jsonl_file:
            item = json.loads(line)
            if item['src'] =='human' and need_human==False:
                continue
            out.append(item)
    print(f"Loaded {len(out)} examples from {file_path}")
    return out

def dict2str(metrics):
    out_str=''
    for key in metrics.keys():
        out_str+=f"{key}:{metrics[key]} "
    return out_str

def gen_embeddings(data, model, tokenizer):
    device = torch.device("cuda")
    dataset = PassagesDataset(data)
    dataloder = DataLoader(dataset, batch_size=32, num_workers=8, shuffle=False)
    labels, embeddings = [], []
    with torch.no_grad():
        for batch in tqdm(dataloder,total=len(dataloder)):
            texts,label,ids= batch
            encoded_batch = tokenizer.batch_encode_plus(
                                texts,
                                return_tensors="pt",
                                max_length=512,
                                padding='max_length',
                                truncation=True,
                            )
            for key in encoded_batch:
                encoded_batch[key] = encoded_batch[key].unsqueeze(1).to(device)


            now_embeddings = model(**encoded_batch)
            now_embeddings = F.normalize(now_embeddings, p=2, dim=-1)
            embeddings.append(now_embeddings.cpu())
            labels.append(label.cpu())
        labels = torch.cat(labels, dim=0).numpy()
        embeddings = torch.cat(embeddings, dim=0).numpy()

    return embeddings, labels

def run(opt):
    device = torch.device("cuda")
    model = AutoModel.from_pretrained("rrivera1849/LUAR-CRUD", trust_remote_code=True)
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("rrivera1849/LUAR-CRUD")
    database_data = load_jsonl(opt.database_path,need_human=False)
    test_data = load_jsonl(opt.test_dataset_path)
    print("Database Data Size:", len(database_data), "Test Data Size:", len(test_data))
    database_embeddings, database_labels = gen_embeddings(database_data, model, tokenizer)
    test_embeddings, test_labels = gen_embeddings(test_data, model, tokenizer)
    dis = test_embeddings @ database_embeddings.T
    dis = dis.min(axis=1)
    metric = evaluate_metrics(test_labels, dis)
    print(dict2str(metric))
    with _LOG_PATH.open("a+", encoding="utf-8") as f:
        f.write(f"UAR  {opt.test_dataset_path}\n")
        f.write(f"{dict2str(metric)}\n")

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", type=str, default="/path/to/RealBench/MAGE_Unseen/Unseen/5shot/train_0.jsonl")
    parser.add_argument("--test_dataset_path", type=str, default="/path/to/RealBench/MAGE_Unseen/Unseen/5shot/test_0.jsonl")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_argument_parser()
    opt = parser.parse_args(argv)
    run(opt)


if __name__ == "__main__":
    main()
# text = ['The quick brown fox jumps over the lazy dog.','There is a cat on the roof.']
# encoded_batch = tokenizer.batch_encode_plus(
#                         text,
#                         return_tensors="pt",
#                         max_length=512,
#                         padding='max_length',
#                         truncation=True,
#                     )
# for key in encoded_batch:
#     encoded_batch[key] = encoded_batch[key].unsqueeze(1).to(device)

# with torch.no_grad():
#     embeddings = model(**encoded_batch)
#     print(embeddings.shape)
