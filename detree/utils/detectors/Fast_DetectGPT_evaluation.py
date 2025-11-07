import logging
import random
import numpy as np
import torch
import argparse
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from ..utils import evaluate_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def get_sampling_discrepancy_analytic(logits_ref, logits_score, labels):
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
    lprobs_score = torch.log_softmax(logits_score, dim=-1)
    probs_ref = torch.softmax(logits_ref, dim=-1)
    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
    discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(dim=-1).sqrt()
    discrepancy = discrepancy.mean()
    return discrepancy.item()

def get_text_crit(text, args, model_config):
    tokenized = model_config["scoring_tokenizer"](text, return_tensors="pt",
                                  return_token_type_ids=False)
    labels = tokenized.input_ids[:, 1:]
    with torch.no_grad():
        logits_score = model_config["scoring_model"](**tokenized).logits[:, :-1]
        if args.reference_model == args.scoring_model:
            logits_ref = logits_score
        else:
            tokenized = model_config["reference_tokenizer"](text, return_tensors="pt",
                                       return_token_type_ids=False)
            assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
            logits_ref = model_config["reference_model"](**tokenized).logits[:, :-1]
        text_crit = get_sampling_discrepancy_analytic(logits_ref, logits_score, labels)

    return text_crit

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
    # load model
    logging.info(f"Loading reference model of type {args.reference_model}...")
    reference_tokenizer = AutoTokenizer.from_pretrained(args.reference_model)
    reference_model = AutoModelForCausalLM.from_pretrained(args.reference_model,device_map="auto")
    reference_model.eval()
    reference_model
    scoring_tokenizer = AutoTokenizer.from_pretrained(args.scoring_model)
    scoring_model = AutoModelForCausalLM.from_pretrained(args.scoring_model,device_map="auto")
    scoring_model.eval()
    scoring_model

    model_config = {
        "reference_tokenizer": reference_tokenizer,
        "reference_model": reference_model,
        "scoring_tokenizer": scoring_tokenizer,
        "scoring_model": scoring_model,
    }

    logging.info(f"Test in {args.test_data_path}")
    test_data = load_jsonl(args.test_data_path)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.shuffle(test_data)
    predictions = []
    labels = []
    st = time.time()
    for i, item in tqdm(enumerate(test_data), total=len(test_data)):
        if i>=100:  # for debugging, only use the first 100 samples
            break
        text = item["text"]
        label = item["label"]
        src = item["src"]
        text_crit = get_text_crit(text, args, model_config)
        if text_crit is None or np.isnan(text_crit) or np.isinf(text_crit):
            text_crit = 0
        if 'human' in src:
            labels.append(0)
        else:
            labels.append(1)
        predictions.append(text_crit)
    ed = time.time()
    print((ed - st) / 100)
    # metric = evaluate_metrics(labels, predictions)
    # print(dict2str(metric))
    # with open("runs/val-other_detector.txt",'a+') as f:
    #     f.write(f"Fast DetectGPT {args.test_data_path} {args.scoring_model} {args.reference_model}\n")
    #     f.write(f"{dict2str(metric)}\n")



    # logging.info(f"{result}")
    # with open(filename.split(".json")[0] + "_Fast_DetectGPT_data.json", "w") as f:
    #     json.dump(test_data, f, indent=4)

    # with open(filename.split(".json")[0] + "_Fast_DetectGPT_result.json", "w") as f:
    #     json.dump(result, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', type=str, default='/path/to/RealBench/Beemo/Llama_edited/test.jsonl',
                        help="Path to the test data. could be several files with ','. "
                             "Note that the data should have been perturbed.")
    parser.add_argument('--reference_model', type=str, default="EleutherAI/gpt-neo-2.7B")
    parser.add_argument('--scoring_model', type=str, default="EleutherAI/gpt-j-6B")
    parser.add_argument('--DEVICE0', default="cuda:0", type=str, required=False)
    parser.add_argument('--DEVICE1', default="cuda:1", type=str, required=False)
    parser.add_argument('--seed', default=2023, type=int, required=False)
    args = parser.parse_args()

    experiment(args)
