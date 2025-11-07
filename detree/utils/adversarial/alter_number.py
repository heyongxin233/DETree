import re
import random

number_pattern = re.compile(r'\d+')

def replace_number(match):
    original = match.group()
    while True:
        new_number = ''.join(random.choices('0123456789', k=len(original)))
        if new_number != original:
            return new_number

def AlterNumbersAttack(text):
    return number_pattern.sub(replace_number, text)


if __name__ == "__main__":
    sample_text = "Before the Marian reforms, the Roman military forces were exclusively made up of citizens people of property (3500 sesterces, say about 1750 loaves of bread) and capable of supplying their own reserves with food. In 1558, when Pope Clement IV was forced to abandon his government in favor of a more centralized army alliance, they became a major part of the new federation and by 1563 they "
    replaced_text = AlterNumbersAttack(sample_text)
    print("Original text:", sample_text)
    print("Text after replacement:", replaced_text)
