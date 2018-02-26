from nltk.corpus import brown
from collections import Counter
import numpy as np

sents = brown.sents()

trigrams = []

for s in sents:
    trigrams.append(('$','$',s[0].lower()))
    if len(s) > 1:
        trigrams.append(('$',s[0].lower(),s[1].lower()))
        for i in range(len(s)-2):
            trigrams.append((s[i].lower(), s[i+1].lower(), s[i+2].lower()))

trigram_counts = Counter(trigrams)
word_counts = Counter(brown.words())

def next(trigram_counts, word_counts, w0, w1):
    candidates = []
    for trigram in trigram_counts.keys():
        if trigram[0] == w0 and trigram[1] == w1:
            candidates.append(trigram[2])
    n = len(brown.words())
    ps = {}
    for candidate in candidates:
        ps[candidate] = (word_counts[candidate]/n)
        t = 0
        for trigram in trigram_counts.keys():
            if trigram[2] == candidate:
                t += trigram_counts[trigram]
        ps[candidate] = ps[candidate] * (trigram_counts[(w0, w1, candidate)] / t)
    return ps

def generate(trigram_counts, word_counts, w0, w1):
    ps = next(trigram_counts, word_counts, w0, w1)
    candidates = list(ps.keys())
    probs = [ps[c] for c in candidates]
    ptotal = sum(probs)
    normprobs = [p/ptotal for p in probs]
    return np.random.choice(candidates, 1, p=normprobs)[0]

for x in range(3):
    w1 = '$'
    w2 = '$'
    sentence = []
    while w2 != '.' and w2 != '!' and w2 != '?':
        w0 = w1
        w1 = w2
        w2 = generate(trigram_counts, word_counts, w0, w1)
        sentence.append(w2)
        print(w2)
    print(sentence)
