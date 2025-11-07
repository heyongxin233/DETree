import argparse
import json
import logging
import random
import re
from itertools import chain
from pathlib import Path
from typing import Sequence

import numpy as np
import regex
import torch
from cleantext import clean
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..utils import evaluate_metrics

_LOG_PATH = Path(__file__).resolve().parents[3] / "runs" / "val-other_detector.txt"
_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

class MosesPunctNormalizer:

    EXTRA_WHITESPACE = [  # lines 21 - 30
        (r"\r", r""),
        (r"\(", r" ("),
        (r"\)", r") "),
        (r" +", r" "),
        (r"\) ([.!:?;,])", r")\g<1>"),
        (r"\( ", r"("),
        (r" \)", r")"),
        (r"(\d) %", r"\g<1>%"),
        (r" :", r":"),
        (r" ;", r";"),
    ]

    NORMALIZE_UNICODE_IF_NOT_PENN = [(r"`", r"'"), (r"''", r' " ')]  # lines 33 - 34

    NORMALIZE_UNICODE = [  # lines 37 - 50
        ("„", r'"'),
        ("“", r'"'),
        ("”", r'"'),
        ("–", r"-"),
        ("—", r" - "),
        (r" +", r" "),
        ("´", r"'"),
        ("([a-zA-Z])‘([a-zA-Z])", r"\g<1>'\g<2>"),
        ("([a-zA-Z])’([a-zA-Z])", r"\g<1>'\g<2>"),
        ("‘", r"'"),
        ("‚", r"'"),
        ("’", r"'"),
        (r"''", r'"'),
        ("´´", r'"'),
        ("…", r"..."),
    ]

    FRENCH_QUOTES = [  # lines 52 - 57
        ("\u00A0«\u00A0", r'"'),
        ("«\u00A0", r'"'),
        ("«", r'"'),
        ("\u00A0»\u00A0", r'"'),
        ("\u00A0»", r'"'),
        ("»", r'"'),
    ]

    HANDLE_PSEUDO_SPACES = [  # lines 59 - 67
        ("\u00A0%", r"%"),
        ("nº\u00A0", "nº "),
        ("\u00A0:", r":"),
        ("\u00A0ºC", " ºC"),
        ("\u00A0cm", r" cm"),
        ("\u00A0\\?", "?"),
        ("\u00A0\\!", "!"),
        ("\u00A0;", r";"),
        (",\u00A0", r", "),
        (r" +", r" "),
    ]

    EN_QUOTATION_FOLLOWED_BY_COMMA = [(r'"([,.]+)', r'\g<1>"')]

    DE_ES_FR_QUOTATION_FOLLOWED_BY_COMMA = [
        (r',"', r'",'),
        (r'(\.+)"(\s*[^<])', r'"\g<1>\g<2>'),  # don't fix period at end of sentence
    ]

    DE_ES_CZ_CS_FR = [
        ("(\\d)\u00A0(\\d)", r"\g<1>,\g<2>"),
    ]

    OTHER = [
        ("(\\d)\u00A0(\\d)", r"\g<1>.\g<2>"),
    ]

    # Regex substitutions from replace-unicode-punctuation.perl
    # https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/replace-unicode-punctuation.perl
    REPLACE_UNICODE_PUNCTUATION = [
        ("，", ","),
        (r"。\s*", ". "),
        ("、", ","),
        ("”", '"'),
        ("“", '"'),
        ("∶", ":"),
        ("：", ":"),
        ("？", "?"),
        ("《", '"'),
        ("》", '"'),
        ("）", ")"),
        ("！", "!"),
        ("（", "("),
        ("；", ";"),
        ("」", '"'),
        ("「", '"'),
        ("０", "0"),
        ("１", "1"),
        ("２", "2"),
        ("３", "3"),
        ("４", "4"),
        ("５", "5"),
        ("６", "6"),
        ("７", "7"),
        ("８", "8"),
        ("９", "9"),
        (r"．\s*", ". "),
        ("～", "~"),
        ("’", "'"),
        ("…", "..."),
        ("━", "-"),
        ("〈", "<"),
        ("〉", ">"),
        ("【", "["),
        ("】", "]"),
        ("％", "%"),
    ]

    def __init__(
        self,
        lang="en",
        penn=True,
        norm_quote_commas=True,
        norm_numbers=True,
        pre_replace_unicode_punct=False,
        post_remove_control_chars=False,
    ):
        """
        :param language: The two-letter language code.
        :type lang: str
        :param penn: Normalize Penn Treebank style quotations.
        :type penn: bool
        :param norm_quote_commas: Normalize quotations and commas
        :type norm_quote_commas: bool
        :param norm_numbers: Normalize numbers
        :type norm_numbers: bool
        """
        self.substitutions = [
            self.EXTRA_WHITESPACE,
            self.NORMALIZE_UNICODE,
            self.FRENCH_QUOTES,
            self.HANDLE_PSEUDO_SPACES,
        ]

        if penn:  # Adds the penn substitutions after extra_whitespace regexes.
            self.substitutions.insert(1, self.NORMALIZE_UNICODE_IF_NOT_PENN)

        if norm_quote_commas:
            if lang == "en":
                self.substitutions.append(self.EN_QUOTATION_FOLLOWED_BY_COMMA)
            elif lang in ["de", "es", "fr"]:
                self.substitutions.append(self.DE_ES_FR_QUOTATION_FOLLOWED_BY_COMMA)

        if norm_numbers:
            if lang in ["de", "es", "cz", "cs", "fr"]:
                self.substitutions.append(self.DE_ES_CZ_CS_FR)
            else:
                self.substitutions.append(self.OTHER)

        self.substitutions = list(chain(*self.substitutions))

        self.pre_replace_unicode_punct = pre_replace_unicode_punct
        self.post_remove_control_chars = post_remove_control_chars

    def normalize(self, text):
        """
        Returns a string with normalized punctuation.
        """
        # Optionally, replace unicode puncts BEFORE normalization.
        if self.pre_replace_unicode_punct:
            text = self.replace_unicode_punct(text)

        # Actual normalization.
        for regexp, substitution in self.substitutions:
            # print(regexp, substitution)
            text = re.sub(regexp, substitution, str(text))
            # print(text)

        # Optionally, replace unicode puncts BEFORE normalization.
        if self.post_remove_control_chars:
            text = self.remove_control_chars(text)

        return text.strip()

    def replace_unicode_punct(self, text):
        for regexp, substitution in self.REPLACE_UNICODE_PUNCTUATION:
            text = re.sub(regexp, substitution, str(text))
        return text

    def remove_control_chars(self, text):
        return regex.sub(r"\p{C}", "", text)

def _tokenization_norm(text):
    text = text.replace(
        ' ,', ',').replace(
        ' .', '.').replace(
        ' ?', '?').replace(
        ' !', '!').replace(
        ' ;', ';').replace(
        ' \'', '\'').replace(
        ' ’ ', '\'').replace(
        ' :', ':').replace(
        '<newline>', '\n').replace(
        '`` ', '"').replace(
        ' \'\'', '"').replace(
        '\'\'', '"').replace(
        '.. ', '... ').replace(
        ' )', ')').replace(
        '( ', '(').replace(
        ' n\'t', 'n\'t').replace(
        ' i ', ' I ').replace(
        ' i\'', ' I\'').replace(
        '\\\'', '\'').replace(
        '\n ', '\n').strip()
    return text


def _clean_text(text):
    # remove PLM special tokens
    plm_special_tokens = r'(\<pad\>)|(\<s\>)|(\<\/s\>)|(\<unk\>)|(\<\|endoftext\|\>)'
    text = re.sub(plm_special_tokens, "", text)

    # normalize puncuations
    moses_norm = MosesPunctNormalizer()
    text = moses_norm.normalize(text)
    
    # normalize tokenization
    text = _tokenization_norm(text)
    
    # remove specific text patterns, e.g,, url, email and phone number
    text = clean(text,
        fix_unicode=True,               # fix various unicode errors
        to_ascii=True,                  # transliterate to closest ASCII representation
        lower=False,                     # lowercase text
        no_line_breaks=True,           # fully strip line breaks as opposed to only normalizing them
        no_urls=True,                  # replace all URLs with a special token
        no_emails=True,                # replace all email addresses with a special token
        no_phone_numbers=True,         # replace all phone numbers with a special token
        no_numbers=False,               # replace all numbers with a special token
        no_digits=False,                # replace all digits with a special token
        no_currency_symbols=False,      # replace all currency symbols with a special token
        no_punct=False,                 # remove punctuations
        replace_with_punct="",          # instead of removing punctuations you may replace them
        replace_with_url="",
        replace_with_email="",
        replace_with_phone_number="",
        replace_with_number="<NUMBER>",
        replace_with_digit="<DIGIT>",
        replace_with_currency_symbol="<CUR>",
        lang="en"                       # set to 'de' for German special handling
    )
    
    # keep common puncts only
    punct_pattern = r'[^ A-Za-z0-9.?!,:;\-\[\]\{\}\(\)\'\"]'
    text = re.sub(punct_pattern, '', text)
    # remove specific patterns
    spe_pattern = r'[-\[\]\{\}\(\)\'\"]{2,}'
    text = re.sub(spe_pattern, '', text)
    # remove redundate spaces
    text = " ".join(text.split())
    return text

def _rm_line_break(text):
    text = text.replace("\n","\\n")
    text = re.sub(r'(?:\\n)*\\n', r'\\n', text)
    text = re.sub(r'^.{0,3}\\n', '', text)
    text = text.replace("\\n"," ")
    return text

def preprocess(text):
    text = _rm_line_break(text)
    text = _clean_text(text)
    return text


def detect(input_text, tokenizer, model, device='cuda', th=-3.08583984375):
    # Tokenize input text
    tokenize_input = tokenizer(input_text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    tensor_input = torch.tensor(tokenize_input["input_ids"]).to(device)
    
    # Get model output
    outputs = model(tensor_input)
    
    # Calculate score (probability for AI-generated text)
    score = -outputs.logits[0][0].item()  # Negative logit for AI-generated probability

    return score

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
    # Initialize MAGE model
    model_dir = "yaful/MAGE"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).cuda()

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
        
        # preprocess the text
        text = preprocess(text)
        
        # MAGE detection
        score = detect(text, tokenizer, model)

        # Determine the label and append to predictions and labels
        if 'human' in src:
            labels.append(1)
        else:
            labels.append(0)

        predictions.append(score)

    # Compute metrics
    metric = evaluate_metrics(labels, predictions)
    print(dict2str(metric))

    # Save results
    with _LOG_PATH.open("a+", encoding="utf-8") as f:
        f.write(f"MAGE  {args.test_data_path}\n")
        f.write(f"{dict2str(metric)}\n")

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--test_data_path',
        type=str,
        default='/path/to/RealBench/DetectRL/Multi_Attack/all_attacks_llm_test.jsonl',
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
