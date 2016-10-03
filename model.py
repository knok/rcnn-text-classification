# -*- coding: utf-8 -*-

import chainer.functions as F
import chainer.links as L
import chainer
from chainer import Chain
import numpy as np
import six

class LetterClassifyer(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size, class_size=2):
        super(LetterClassifyer, self).__init__(
            embed = L.EmbedID(vocab_size, embed_size),
            fc1 = L.Linear(embed_size*2, hidden_size*2),
            fc2 = L.Linear(hidden_size*2, class_size)
            )
    def forward(self, x_list):
        cl_list = []
        for x in x_list:
            wvec = self.embed(x)
            cl_list.append(wvec)
        cr_list = []
        for x in reversed(x_list):
            wvec = self.embed(x)
            cr_list.append(wvec)
        xi_list = []
        for cl, cr in zip(cl_list, cr_list):
            xi_list.append(F.concat((cl, cr)))
        yi_list = []
        for xi in xi_list:
            yi_list.append(F.tanh(self.fc1(xi)))
        y3 = yi_list[0]
        for yi in yi_list[1:]:
            y3 = F.maximum(yi, y3)
        y4 = self.fc2(y3)
        return y4
    
