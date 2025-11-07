import re
import nltk
import argparse
import hashlib
import json
import math
import os
import random

sentence_end_regex = r'([.!?。？！])(?=\s|$)'

def InsertParagraphsAttack(text,N=0.5):
    sentences = re.split(sentence_end_regex, text)
    sentences_to_alter = math.ceil((len(sentences) - 1) * N)
    indices_to_alter = random.sample(range(1, len(sentences)), sentences_to_alter)
    sorted_indices_to_alter = sorted(indices_to_alter)
    for i in sorted_indices_to_alter:
        if random.random() < 0.5:
            sentences[i] = sentences[i]+"\n\n  "
        else:
            sentences[i] = sentences[i]+"\n\n"
    return "".join(sentences)

if __name__ == "__main__":
    sample_text = "Before Marian reforms, Roman military forces were exclusively made up of citizens people of property (3500 sesterces, say about 1750 loaves of bread) and capable of supplying their own reserves with food. In 1558, when Pope Clement IV was forced to abandon his government in favor of a more centralized army alliance, they became a major part of the new federation and by 1563 they "
    replaced_text = InsertParagraphsAttack(sample_text)
    print("Original text:", sample_text)
    print("Text after replacement:", replaced_text)

