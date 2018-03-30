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
import time


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size, num_layers):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.input_size = input_size
        self.batch_size = batch_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(int(input_size), int(hidden_size), num_layers=int(self.num_layers), dropout=0.25)
        self.oh2o = nn.Linear(hidden_size + output_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output, hidden = self.rnn(input.view(9, self.batch_size, self.input_size), hidden)
        output = output[-1,:,:]
        hidden = hidden[-1,:,:]
        output_combined = torch.cat((hidden, output), 1)
        output = self.oh2o(output_combined)
        output = self.softmax(output)
        return output

    def initHidden(self):
        return autograd.Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size))

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
    words = [word.lower() for sent in sents for word in sent]
    print(len(words))
    w_to_idx, idx_to_w, vocab_size = one_hot_encode(set(words))
    print('got the w_t_idx')
    #minibatches = proc_sent(sents,w_to_idx, vocab_size, batch_size)
    #print('made the minibatches')
    print(vocab_size)
    model = RNN(vocab_size, int(vocab_size), vocab_size, batch_size, num_layers)
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(epochs):
        loss_list = []
        k = 0
        l = 0
        minibatch = []
        minibatch_ys = []
        running_loss = []
        for sent in sents:
            start = time.time()
            start_point = deque(maxlen=9)
            for i in range(9):
                start_point.append(np.zeros(vocab_size, dtype=int))

            for word in range(len(sent)-1):
                k += 1
                word = word.lower()
                if len(minibatch) == batch_size:
                    l += 1
                    break
                temp = np.zeros(vocab_size, dtype=int)
                temp[w_to_idx[ sent[word] ]] = 1
                start_point.append(temp)
                minibatch.append(torch.FloatTensor(start_point))

                temp_y = np.zeros(vocab_size, dtype=int)
                temp_y[w_to_idx[ sent[word+1] ]] = 1
                y_tens = torch.FloatTensor(temp_y)
                minibatch_ys.append(y_tens)
            if len(minibatch) == batch_size:
                input =  autograd.Variable(torch.stack(minibatch), requires_grad=True)

                targets =  autograd.Variable(torch.stack(minibatch_ys))
                minibatch = []
                minibatch_ys = []
                hidden = model.initHidden()
                optimizer.zero_grad()
                predictions = model(input, hidden)
                y_hat = predictions.type(torch.FloatTensor)
                loss = loss_function(y_hat, targets)
                loss.backward()
                optimizer.step()
                running_loss.append(loss.data[0])

                #print( i, end='\r')
                if l % 10 == 9:
                    print('[%d, %5d] loss: %.5f' % (epoch + 1, k+1, np.mean(running_loss)), end=' ')
                    print(running_loss)
                    running_loss = []

    #producing 5 sentences
    model.batch_size =  1
    for i in range(5):
        sentence = []
        for i in range(9):
            start_point.append(np.zeros(vocab_size))
        while word not in ['.','!','?']:
            hidden = model.initHidden()
            input = autograd.Variable(torch.FloatTensor(start_point))
            prediction = model(input, hidden)
            argmax = max(xrange(len(prediction)), key=prediction.__getitem__)
            word = idx_to_w[argmax]
            one_hot_word = np.zeros(vocab_size)
            one_hot_word[argmax] = 1
            start_point.append(one_hot_word)
            print(word)
            sentence.append(word)
        print(sentence)

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
            minibatch.append(torch.FloatTensor(start_point))

            temp_y = np.zeros(vocab_size, dtype=int)
            temp_y[w_to_idx[ sent[word+1] ]] = 1
            y_tens = torch.FloatTensor(temp_y)
            minibatch_ys.append(y_tens)
    return minibatches

if __name__ == '__main__':
            #(epochs,       num_layers,       batch_size)
    train(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
