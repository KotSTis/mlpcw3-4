from nltk.corpus import brown
from collections import Counter
import numpy as np

sents = brown.tagged_sents()

trigrams = []

for s in sents:
    trigrams.append(('$','$',s[0][1]))
    if len(s) > 1:
        trigrams.append(('$', s[0][1], s[1][1]))
        for i in range(len(s)-2):
            trigrams.append((s[i][1], s[i+1][1], s[i+2][1]))

trigram_counts = Counter(trigrams)
tags = []
for tagged_word in brown.tagged_words():
    tags.append(tagged_word[1])
tag_counts = Counter(tags)
wpt = {}
for tag in set(tags):
    wpt[tag] = []
for tagged_word in brown.tagged_words():
    wpt[tagged_word[1]].append(tagged_word[0].lower())
word_counts = {}
for k in list(wpt.keys()):
    word_counts[k] = Counter(wpt[k])

def next(trigram_counts, tag_counts, t0, t1):
    candidates = []
    for trigram in trigram_counts.keys():
        if trigram[0] == t0 and trigram[1] == t1:
            candidates.append(trigram[2])
    n = len(tags)
    ps = {}
    for candidate in candidates:
        ps[candidate] = (tag_counts[candidate]/n)
        c = 0
        for trigram in trigram_counts.keys():
            if trigram[2] == candidate:
                c += trigram_counts[trigram]
        ps[candidate] = ps[candidate] * (trigram_counts[(t0, t1, candidate)] / c)
    return ps

def generate(trigram_counts, tag_counts, t0, t1):
    ps = next(trigram_counts, tag_counts, t0, t1)
    candidates = list(ps.keys())
    probs = [ps[c] for c in candidates]
    ptotal = sum(probs)
    normprobs = [p/ptotal for p in probs]
    return np.random.choice(candidates, 1, p=normprobs)[0]

def tag2word(t, word_counts, wpt):
    n = 0
    for wl in wpt[t]:
        n+=1
    candidates = list(word_counts[t].keys())
    probs = []
    for candidate in candidates:
        probs.append(word_counts[t][candidate] / n)
    return np.random.choice(candidates, 1, p=probs)[0]

for x in range(5):
    t1 = '$'
    t2 = '$'
    sentence = []
    while t2 != '.':
        t0 = t1
        t1 = t2
        t2 = generate(trigram_counts, tag_counts, t0, t1)
        w = tag2word(t2, word_counts, wpt)
        sentence.append(w)
        print(w)
    print(sentence)
