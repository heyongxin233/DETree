import argparse
import hashlib
import json
import math
import os
import random
import regex as re


_HERE = os.path.dirname(__file__)
_RESOURCE_PATH = os.path.join(_HERE, "resources", "american_spellings.json")
with open(_RESOURCE_PATH, encoding="utf-8") as fp:
    us_to_gb_spelling = json.load(fp)

def AlternativeSpellingAttack(text):
    matches = list(re.finditer(r"\L<words>", text, words=us_to_gb_spelling.keys()))
    delta = 0
    matches_to_swap = sorted(matches, key=lambda m: m.start())
    for m in matches_to_swap:
        # Get the left and right index of the swap
        left = m.start() + delta
        right = m.end() + delta

        # Get the match and insert it into the text
        gb_spelling = us_to_gb_spelling[m.group()]
        text = text[:left] + gb_spelling + text[right:]

        # Edit delta to account for indexing changes
        delta += len(gb_spelling) - len(m.group())
    
    return text


if __name__ == "__main__":
    sample_text = "Before the Marian reforms, the Roman military forces were exclusively made up of citizens people of property (3500 sesterces, say about 1750 loaves of bread) and capable of supplying their own reserves with food. In 1558, when Pope Clement IV was forced to abandon his government in favor of a more centralized army alliance, they became a major part of the new federation and by 1563 they "
    replaced_text = AlternativeSpellingAttack(sample_text)
    print("Original text:", sample_text)
    print("Text after replacement:", replaced_text)
