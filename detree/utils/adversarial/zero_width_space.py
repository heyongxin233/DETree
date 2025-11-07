import argparse
import hashlib
import json
import math
import os
import random
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
zero_width_space = "\u200b"

def ZeroWidthSpaceAttack(text,N=0.3):
    texts = text.split(" ")
    num_spaces = int(N * len(texts))
    indices_to_alter = random.sample(range(len(texts)), num_spaces)
    indices_to_alter = sorted(indices_to_alter)
    for i in indices_to_alter:
        texts[i] += zero_width_space
    return " ".join(texts)

if __name__ == "__main__":
    sample_text = "Before Marian reforms, Roman military forces were exclusively made up of citizens people of property (3500 sesterces, say about 1750 loaves of bread) and capable of supplying their own reserves with food. In 1558, when Pope Clement IV was forced to abandon his government in favor of a more centralized army alliance, they became a major part of the new federation and by 1563 they "
    replaced_text = ZeroWidthSpaceAttack(sample_text)
    print("Original text:", sample_text)
    print("Text after replacement:", replaced_text)