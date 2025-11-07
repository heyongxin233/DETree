import random

def WhiteSpaceAttack(text,N=0.5):
    texts = text.split(" ")
    spaces_to_alter = int(len(texts) * N)
    indices_to_alter = random.choices(range(len(texts)), k=spaces_to_alter)
    indices_to_alter = sorted(indices_to_alter)
    for i in indices_to_alter:
        texts[i] += " "
    return " ".join(texts)

if __name__ == "__main__":
    sample_text = "Before Marian reforms, Roman military forces were exclusively made up of citizens people of property (3500 sesterces, say about 1750 loaves of bread) and capable of supplying their own reserves with food. In 1558, when Pope Clement IV was forced to abandon his government in favor of a more centralized army alliance, they became a major part of the new federation and by 1563 they "
    replaced_text = WhiteSpaceAttack(sample_text)
    print("Original text:", sample_text)
    print("Text after replacement:", replaced_text)