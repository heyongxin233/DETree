import argparse
import hashlib
import json
import math
import os
import random
import regex as re

mapping = {
            "a": ["а"],
            "A": ["А", "Α"],
            "B": ["В", "Β"],
            "e": ["е"],
            "E": ["Е", "Ε"],
            "c": ["с"],
            "p": ["р"],
            "K": ["К", "Κ"],
            "O": ["О", "Ο"],
            "P": ["Р", "Ρ"],
            "M": ["М", "Μ"],
            "H": ["Н", "Η"],
            "T": ["Т", "Τ"],
            "X": ["Х", "Χ"],
            "C": ["С"],
            "y": ["у"],
            "o": ["о"],
            "x": ["х"],
            "I": ["І", "Ι"],
            "i": ["і"],
            "N": ["Ν"],
            "Z": ["Ζ"],
        }
pattern = re.compile('|'.join(re.escape(char) for char in mapping.keys()))

def HomoglyphAttack(text):
    return pattern.sub(lambda match: random.choice(mapping[match.group()]), text)

def print_unicode(text):
    return " ".join(str(ord(char)) for char in text)
if __name__ == "__main__":
    sample_text = "Before Marian reforms, Roman military forces were exclusively made up of citizens people of property (3500 sesterces, say about 1750 loaves of bread) and capable of supplying their own reserves with food. In 1558, when Pope Clement IV was forced to abandon his government in favor of a more centralized army alliance, they became a major part of the new federation and by 1563 they "
    replaced_text = HomoglyphAttack(sample_text)
    print("Original text:", print_unicode(sample_text))
    print("Text after replacement:", print_unicode(replaced_text))
