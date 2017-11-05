# -*- coding: utf-8 -*-
'''
all.txt的标签是前3000为1，后3000为0
把数据打乱，数据和标签分别存储
'''
import codecs
import jieba
import re
import h5py
import numpy as np
import random

def loadTrainingInfo(path):
    '''载入分词训练数据字典和概率'''
    print('load training info from %s'%path)
    fd = open(path, 'r')
    lines = fd.readlines()
    fd.close()
    vocab = {}
    indexVocab = [0 for i in range(len(lines))]
    for line in lines:
        rst = line.strip().split('\t')
        if len(rst) < 2: continue
        char, index = rst[0], int(rst[1])
        vocab[char] = index
        indexVocab[index] = char
    return (vocab, indexVocab)

def loadTrainingData(path):
    '''载入分词训练输入样本'''
    print('load training data from %s'%path)
    fd = h5py.File(path,'r')
    X = fd['X'][:]
    fd.close()
    return X

def shuffle(maxlen):
    training_data_filePath = "./data/traindata/training-" + str(maxlen) + ".data"
    trainingData = loadTrainingData(training_data_filePath)
    X = trainingData
    Y = []
    for i in range(3000):
        Y.append(1)
    for i in range(3000):
        Y.append(0)
    X = X.reshape(-1, maxlen)
    # 打乱数据
    index = [i for i in range(6000)]
    random.shuffle(index)
    X = X[index]
    Y = np.array(Y)
    Y = Y[index]
    fd = h5py.File('./data/shuffledata/training-shuffle-' + str(maxlen) + '.data', 'w')
    fd.create_dataset('X', data=X)
    fd.create_dataset('Y', data=Y)
    fd.close()

#---------------------打乱数据-------------------#
shuffle(25)
#---------------------END-------------------#