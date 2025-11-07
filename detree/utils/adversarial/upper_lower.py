import nltk
import math
import os
import random


tokenizer = nltk.tokenize.NLTKWordTokenizer()

def UpperLowerFlipAttack(text, N=0.5):
    indices = [s for s, e in tokenizer.span_tokenize(text) if text[s].isalpha()]
    num_to_flip = math.ceil(len(indices) * N)
    flip_indices = random.sample(indices, num_to_flip)
    text = list(text)
    for i in flip_indices:
        text[i] = text[i].lower() if text[i].isupper() else text[i].upper()
    return "".join(text)

if __name__ == "__main__":
    sample_text = "Before Marian reforms, Roman military forces were exclusively made up of citizens people of property (3500 sesterces, say about 1750 loaves of bread) and capable of supplying their own reserves with food. In 1558, when Pope Clement IV was forced to abandon his government in favor of a more centralized army alliance, they became a major part of the new federation and by 1563 they "
    replaced_text = UpperLowerFlipAttack(sample_text)
    print("Original text:", sample_text)
    print("Text after replacement:", replaced_text)