from __future__ import print_function, division
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import sys
from sklearn.metrics import r2_score
import os
from collections import deque
from nltk.corpus import brown


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size, num_layers):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.input_size = input_size
        self.batch_size = batch_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size, hidden_dim, num_layers, dropout=0.25)
        self.oh2o = nn.Linear(hidden_size + output_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):

        output, hidden = self.rnn(input.view(9, self.batch_size, self.input_size), self.hidden)
        output = output[-1,:,:]
        hidden = hidden[-1,:,:]
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.softmax(output)
        return output

    def initHidden(self):
        return Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size))

def one_hot_encode(corpus_vocab):
    word_to_idx = dict((word, idx) for idx, word in enumerate(corpus_vocab))
    idx_to_word= dict((idx, word) for idx, word in enumerate(corpus_vocab))
    return word_to_idx, idx_to_word, len(corpus_vocab)

def two_hot_encode():
    corpus_full = brow.tagged_words(tagset='universal')
    corpus_vocab = [g[0] for g in corpus_full]
    tags = set([g[1] for g in corpus_full])

    word_to_idx = dict((word, idx) for idx, word in enumerate(corpus_vocab))
    idx_to_word= dict((idx, word) for idx, word in enumerate(corpus_vocab))

    tag_to_idx = dict((tag, idx) for idx, tag in enumerate(tags))
    idx_to_tag= dict((idx, tag) for idx, tag in enumerate(tags))

    return word_to_idx, idx_to_word, len(corpus_vocab), tag_to_idx, idx_to_tag, len(tags)

def train(epochs, num_layers, batch_size):
    sents = brown.sents()
    w_to_idx, idx_to_w, vocab_size = one_hot_encode(brown.words())
    minibatches = proc_sent(sents,w_to_idx, vocab_size, batch_size)

    model = RNN(vocab_size, vocab_size*1.2, vocab_size, batch_size, num_layers)
    loss_function = nn.NLLLoss(reduce=False)
    optimizer = optim.Adam(model.parameters())

    for epoch in range(epochs):
        loss_list = []
        for minibatch, minibatch_ys in minibatches:
            if len(minibatch) != batch_size:
                continue
            loss = 0.
            hidden = rnn.initHidden()
            opt.zero_grad()
            prediction = model(minibatch, hidden)

            for b in range(batch_size):
                loss += loss_function(prediction[:,b,:], minibatch_ys[b])
            loss.backward()
            optimizer.step()
            loss_list.extend(loss/batch_size)

    #producing 5 sentences
    model.batch_size =  1
    for i in range(5):
        lastnine = []
        for i in range(9):
            start_point.append(np.zeros(vocab_size))
        while word not in ['.','!','?']:
            hidden = rnn.initHidden()
            prediction = model(input, hidden)
            argmax = max(xrange(len(prediction)), key=values.__getitem__)
            word = idx_to_w[argmax]
            lastnine = lastnine[:-8].append(word)
            sentence[i].append(word)
        print(sentence)

def proc_sent(sents,w_to_idx, vocab_size, batch_size):
    minibatches = []
    minibatch = []
    minibatch_ys = []
    for sent in sents:

        if len(minibatch) == batch_size:
            minibatches.append(( autograd.Variable(minibatch, requires_grad=True), autograd.Variable(minibatch_ys) ))
            minibatch = []
            minibatch_ys = []

        start_point = deque(maxlen=9)
        for i in range(9):
            start_point.append(np.zeros(vocab_size, dtype=int))

        for word in range(len(sent)-1):
            temp = np.zeros(vocab_size, dtype=int)
            temp[w_to_idx[ sent[word] ]] = 1
            start_point.append(temp)
            minibatch.append(torch.LongTensor(start_point))

            temp_y = np.zeros(vocab_size, dtype=int)
            temp_y[w_to_idx[ sent[word+1] ]] = 1
            y_tens = torch.FloatTensor(temp_y)
            minibatch_ys.append(y_tens)
    return minibatches

def proc_sent_two_hot(sents,w_to_idx, vocab_size, batch_size):
    minibatches = []
    minibatch = []
    minibatch_ys = []
    for sent in sents:

        if len(minibatch) == batch_size:
            minibatches.append(( autograd.Variable(minibatch, requires_grad=True), autograd.Variable(minibatch_ys) ))
            minibatch = []
            minibatch_ys = []

        start_point = deque(maxlen=9)
        for i in range(9):
            start_point.append(np.zeros(vocab_size, dtype=int))

        for word in range(len(sent)-1):
            temp = np.zeros(vocab_size, dtype=int)
            temp[w_to_idx[ sent[word] ]] = 1
            start_point.append(temp)
            minibatch.append(torch.LongTensor(start_point))

            temp_y = np.zeros(vocab_size, dtype=int)
            temp_y[w_to_idx[ sent[word+1] ]] = 1
            y_tens = torch.FloatTensor(temp_y)
            minibatch_ys.append(y_tens)
    return minibatches

if __name__ == '__main__':
    #train(int(sys.argv[1]), int(sys.argv[2]). int(sys.argv[3]), int(sys.argv[4]))
    train(1,2,50)
