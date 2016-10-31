import numpy as np
import cPickle as pickle
import gym, argparse, os, chainer
from datetime import datetime
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import copy
import time

class CNN(object):
    def __init__(self):
        super(Conv_NN, self).__init__(
            conv_1=L.Convolution2D(4, 4, 8, stride=2, pad=1),
            conv_2=L.Convolution2D(4, 4, 8, stride=1, pad=1)
        )

    def __call__(self, x_data):
        output = Variable(x_data)
        output = F.relu(self.conv_1(output))
        output = F.relu(self.conv_2(output))
        # note that no pooling layers are included, since translation is important for most games
        return output
        
class Policy_net(object):
    """docstring for policy"""
    def __init__(self, arg):
        pass
        

class Value_net(object):
    """docstring for Value"""
    def __init__(self, arg):
        pass


        