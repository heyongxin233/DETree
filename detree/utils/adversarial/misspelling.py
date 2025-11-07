import hashlib
import itertools
import json
import os
import random
from pathlib import Path
import re
import nltk
from tqdm import tqdm


def _ensure_nltk_resource(resource_name: str, resource_paths: list[str]) -> None:
    """Download an NLTK resource only if none of the candidate paths exist."""

    for resource_path in resource_paths:
        try:
            nltk.data.find(resource_path)
            return
        except LookupError:
            continue

    nltk.download(resource_name)


_ensure_nltk_resource(
    "punkt_tab",
    [
        "tokenizers/punkt_tab",
        "tokenizers/punkt_tab/english.pickle",
        "tokenizers/punkt",
        "tokenizers/punkt/english.pickle",
    ],
)
_ensure_nltk_resource(
    "averaged_perceptron_tagger_eng",
    [
        "taggers/averaged_perceptron_tagger_eng",
        "taggers/averaged_perceptron_tagger_eng/averaged_perceptron_tagger_eng.pickle",
    ],
)

def merge_dict(dict1, dict2):
    for key in dict1.keys():
        if key in dict2:
            dict1[key] += dict2[key]
    return dict1


sent_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
sspan = lambda x: sent_tokenizer.span_tokenize(x)
word_tokenizer = nltk.tokenize.NLTKWordTokenizer()
wspan = lambda x: word_tokenizer.span_tokenize(x)
current_directory = Path(os.path.dirname(os.path.abspath(__file__)))
json_path = current_directory / "resources" / "misspelling3.json"
with open(json_path, "r") as f:
    corrections = json.load(f)
json_path = current_directory / "resources" / "misspellings.json"
with open(json_path, "r") as f:
    corrections = merge_dict(corrections, json.load(f))
json_path = current_directory / "resources" / "misspellings2.json"
with open(json_path, "r") as f:
    corrections = merge_dict(corrections, json.load(f))

def get_nltk_span_tokens(text):
        return [(ws + s, we + s) for s, e in sspan(text) for ws, we in wspan(text[s:e])]

def misspell(w):
    misspellings = corrections.get(w.lower())
    choice = random.choice(misspellings)
    return choice.capitalize() if w[0].isupper() else choice

def can_misspell(word):
    return word.lower() in corrections

def MisspellingAttack(text):
    word_spans = get_nltk_span_tokens(text)
    all_spans = list(itertools.pairwise(itertools.chain.from_iterable(word_spans)))
    toks = [text[s:e] for s, e in all_spans]

    candidate_spans = [(i, s, e) for i, (s, e) in enumerate(all_spans) if can_misspell(text[s:e])]
    for i, s, e in candidate_spans:
        toks[i] = misspell(text[s:e])
    return "".join(toks)

if __name__ == "__main__":
    sample_text = "Tap tap tap on the door. Bitsy gathered her bravery, approached the closet's door as she knew she must, face her fear as she did each night. Mother, with her hand still raised and her back turned to me, was lifting a dress out of its hanger and turning it over in her hands. \"What are you doing?\" I asked quietly, my voice trembling just like her own, afraid that if we spoke too loudly at this moment their voices would carry through the wall into mine. \"Mother,\" I whispered again. She stopped what she was doing and faced me across the room. Her eyes were puffy from lack of sleep and her hair had been pulled up off her neck in an untidy bun. The lines around her mouth - the ones so carefully curled by her hairdresser earlier today - were hardening now, becoming more pronounced than they'd ever really looked before. If Lily wasn't there for us anymore, why should any mother be? Why even bother trying when all our children left home one after another without warning or explanation? Hadn't anyone taught them anything about life or love? Did they think only happy things happened inside houses? I stood irresolute, staring blankly ahead while Mother slowly lowered herself onto the floor, then began gently rubbing the side of her cheek against the bottom of the dress being examined. When she felt something wet she looked down at her hands and saw bits of dried blood smeared along one knuckle. It took everything within me not to cry out No! But I didn't want to see her fall apart right here, right now, because surely if she gave way once, soon enough the whole house could go crashing down around us. Instead of saying the word aloud, however, which might force matters further forward, I said nothing but simply went and sat beside her, brushing the strands of her hair away from where they clung to her forehead. As I rubbed her head I tried to imagine how the two of us used to sit together under the same roof, how gentle and loving it had always been between us, and how much happier it made me feel knowing she cared deeply for me, no matter what else came along later. So many thoughts raced through my mind and heart: that the entire household may have fallen apart; that perhaps we never would be able to make things better, or maybe never get past this terrible midlife crisis together. All of these anxious questions passed through my brain, but none rose above the others until I realized that I hadn't moved an inch since sitting down next to her. Now I found myself looking directly into her tear-streaked face, searching earnestly for some clue as to why she'd done such a thing. Perhaps I'd convinced myself that she needed help somehow. In fact, it seemed so obvious now that I'm sure it had occurred to me before, probably during my long walk alone toward my old apartment building. Still, I couldn't say exactly why I wanted to know. What would I do if it became necessary? Would I try to drag her outside, lift her bodily from the front hall rug into the living room, and hold her close until someone arrived who could give her aid? Or would I tell her that she should call 911, let them come and take care of her properly? It doesn't matter now, I thought sadly, feeling strangely detached from myself even though I was intimately involved. Whatever happens will happen. We'll find a way. Maybe tonight, maybe tomorrow, or three months from now, the older girls will return. They're both in college now. One is riding high on a full football scholarship. They won't need money, or begrudge us having any ourselves, anyway. There could be absolutely no harm in giving them your new address, anonymously of course, explaining that you've decided to spend a few quiet years here among family friends, henceforth avoiding contact with either of them. And yet, despite the tremendous relief I supposed that knowledge brought me, I also missed those daughters terribly, desperately wished I'd chosen to intervene sooner rather than letting the situation deteriorate as far as it already had. The Girl Who Ran Away (Chapter 3) 6293 Dear Diary, So far I haven't talked about one very important person in all this business: Father. He hasn't run away, nor has he threatened it, but whenever Mom goes near him he gets nervous, his body tensing up until you can almost hear the springs squeaking beneath him. This morning Edie woke me quite early, telling me breathless that Dad sounded sick and crying, and I rushed downstairs ready to rush out after him. Then I remembered that he had gotten hurt last week, got carted off to the hospital in Torrance for stitches, and stayed overnight. By Sunday afternoon Edie told me he'd come home fine, although limping like hell. For weeks afterwards everyone acted blithely unconcerned with whatever ailment he suffered, without-scandentirely irrelevant. Not that it bothered taking current symptomatically serious symptoms setterfrua recent token looks behind the other members' mutations."
    replaced_text = MisspellingAttack(sample_text)
    print("Original text:", sample_text)
    print("Text after replacement:", replaced_text)