# -*- coding: utf-8 -*-

import argparse
import sys, time

import chainer
import chainer.optimizers
import chainer.serializers
import chainer.functions as F
from chainer import Variable
from chainer import cuda
import numpy as np
from model import LetterClassifyer

def argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode')
    parser.add_argument('file')
    parser.add_argument('--embed', default=200, type=int)
    parser.add_argument('--vocab', default=3000, type=int)
    parser.add_argument('--hidden', default=1000, type=int)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--model', default="model")
    parser.add_argument('--use-gpu', action='store_true', default=False)
    parser.add_argument('--unchain', action='store_true', default=False)
    args = parser.parse_args()
    return args

# ファイルから1文字単位の列とラベルを取得
def letter_list(fname):
    with open(fname) as f:
        for l in f:
            body = l[:-3]
            val = int(l[-2])
            x = list(''.join(body.split()))
            x.append('</s>')
            yield x, val
def letter_list_text(t):
    x = list(''.join(t.split()))
    x.append('</s>')
    return x
# 
class Vocabulary:
    def __init__(self, fname):
        self.fname = fname
        self.l2i = {}
        self.i2l = []
        if not fname is None:
            self.load_vocab()
    def stoi(self, letter):
        if letter in self.l2i:
            return self.l2i[letter]
        return self.l2i['<unk>']
    def itos(self, id):
        if id < len(self.i2l):
            return self.i2l[id]
        return '<unk>'
            
    def append_letter(self, l):
        if l in self.l2i:
            return
        self.i2l.append(l)
        id = len(self.i2l) -1
        self.l2i[l] = id
    def load_vocab(self):
        self.append_letter('<unk>')
        self.append_letter('<s>')
        self.append_letter('</s>')
        with open(self.fname) as f:
            for line in f:
                nline = line[:-3]
                for l in nline:
                    self.append_letter(l)

    def save_vocab(self, filename):
        with open(filename, 'w') as f:
            for l in self.i2l:
                f.write(l + "\n")
    @staticmethod
    def load_from_file(filename):
        vocab = Vocabulary(None)
        with open(filename) as f:
            for l in f:
                l = l[:-1]
                vocab.append_letter(l)
        return vocab

def forward(src_batch, t, model, is_training, vocab, xp):
    batch_size = len(src_batch)
    src_len = len(src_batch[0])
    src_stoi = vocab.stoi
    x_batch = [Variable(xp.asarray([[src_stoi(x)]], dtype=xp.int32)) for x in src_batch[0]]
    y = model.forward(x_batch)
    if is_training:
        t = Variable(xp.asarray([t], dtype=xp.int32))
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        return y, acc, loss
    else:
        return y
        
def train(args):
    if args.use_gpu:
        xp = cuda.cupy
        cuda.get_device(0).use()
    else:
        xp = np
    vocab = Vocabulary(args.file)
    m = LetterClassifyer(args.vocab, args.embed, args.hidden)
    m.zerograds()
    if args.use_gpu:
        m.to_gpu()
    time_t = 10
    for e in range(args.epoch):
        opt = chainer.optimizers.Adam(alpha=0.001)
        opt.setup(m)
        opt.add_hook(chainer.optimizer.GradientClipping(5.0))
        print("epoch: %d" % e)
        i =0
        total_acc = 0
        e_acc = 0.0
        for x_batch, y in letter_list(args.file):
            x_batch = [x_batch]
            output, acc, loss = forward(x_batch, y, m, True, vocab, xp)
            total_acc += acc
            e_acc += acc
            if i % time_t == 0:
                if i != 0:
                    total_acc /= time_t
                print("time: %d, accuracy %f loss %f" % (i, total_acc.data, loss.data))
                total_acc = 0
                # print("".join(x_batch[0]))
                # print(",".join([str(vocab.stoi(x)) for x in x_batch[0]]))
            loss.backward()
            if args.unchain:
                loss.unchain_backward()
            opt.update()
            i += 1
            sys.stdout.flush()
        chainer.serializers.save_hdf5(args.model + ".hdf5", m)
        vocab.save_vocab(args.model + ".vocab")
        e_acc /= i
        print("total acc: %f" %  e_acc.data)

def eval(args):
    if args.use_gpu:
        xp = cuda.cupy
        cuda.get_device(0).use()
    else:
        xp = np
    vocab = Vocabulary.load_from_file("%s.vocab" % args.model)
    m = LetterClassifyer(args.vocab, args.embed, args.hidden)
    chainer.serializers.load_hdf5("%s.hdf5" % args.model, m)
    if args.use_gpu:
        m.to_gpu
    x_batch = [letter_list_text(args.file)]
    output = forward(x_batch, None, m, False, vocab, xp)
    print(output.data)
    print("hyp: %d" % np.argmax(output.data)) # label

def main():
    args = argument()
    if args.mode == 'train':
        train(args)
    else:
        eval(args)

if __name__ == '__main__':
    main()
#
