import re
import random
import math
articles = ["a ", "an ", "the "]
article_pattern = r'\b(?:' + '|'.join(map(re.escape, articles)) + r')\b'

def ArticleDeletionAttack(text,N=0.5):

    all_indices = [(m.start(), m.group()) for m in re.finditer(article_pattern, text)]
    indices_to_delete = random.sample(all_indices, math.ceil(len(all_indices) * N))
    indices_to_delete = sorted(indices_to_delete)
    result = []
    last_end = 0
    for index, article in indices_to_delete:
        result.append(text[last_end:index])
        last_end = index + len(article)
    result.append(text[last_end:])
    return ''.join(result)

if __name__ == "__main__":
    sample_text = "Before Marian reforms, Roman military forces were exclusively made up of citizens people of property (3500 sesterces, say about 1750 loaves of bread) and capable of supplying their own reserves with food. In 1558, when Pope Clement IV was forced to abandon his government in favor of a more centralized army alliance, they became a major part of the new federation and by 1563 they "
    replaced_text = ArticleDeletionAttack(sample_text)
    print("Original text:", sample_text)
    print("Text after replacement:", replaced_text)
