# -*- coding: utf-8 -*-
'''
10折交叉训练数据
'''
import codecs
import jieba
import re
import h5py
from numpy import *
import numpy as np
import random
import time
from gensim.models import Word2Vec
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers import Bidirectional, TimeDistributed
from keras.callbacks import EarlyStopping
import sys

def loadTrainingData(path):
    '''载入分词训练输入样本'''
    fd = h5py.File(path,'r')
    X = fd['X'][:]
    Y = fd['Y'][:]
    fd.close()
    return X,Y

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

# 标准输出重定向至文件
savedStdout = sys.stdout
file = open('out.txt', 'w+')
sys.stdout = file

# 加载词库
training_info_filePath = "./data/training.info"
modelPath = "./data/keras_model"
weightPath = "./data/keras_model_weights"
word2vec_model_file = "./data/word2vec.model"
trainingInfo = loadTrainingInfo(training_info_filePath)
print ('Load vocab Done!')
print ('Training model...')
start_time = time.time()
(vocab, indexVocab) = trainingInfo

# 加载词向量信息，作为Embedding层的权重
vocabSize = len(vocab) + 1
w2vModel = Word2Vec.load(word2vec_model_file)
embeddingDim = w2vModel.vector_size
embeddingUnknown = [0 for i in range(embeddingDim)]
embeddingWeights = np.zeros((vocabSize + 1, embeddingDim))
for word, index in vocab.items():
    if word in w2vModel:
        e = w2vModel[word]
    else:
        e = embeddingUnknown
    embeddingWeights[index, :] = e

# 训练
for i in range(10):
    #i的序号为测试集的序号，其余为训练集
    Train_X = []
    Train_Y = []
    temp = 0
    for j in range(10):
        if i==j:
            continue
        path = './data/10/'+ str(j) + '.data'
        x, y = loadTrainingData(path)
        Train_X.append(x)
        Train_Y.append(y)
    test_path = './data/10/' + str(i) +'.data'
    Test_X, Test_Y = loadTrainingData(test_path)
    Train_X = np.array(Train_X)
    Train_Y = np.array(Train_Y)
    Train_X = Train_X.reshape(5400,-1)
    Train_Y = Train_Y.reshape(5400)
    model = Sequential()
    model.add(Embedding(output_dim=embeddingDim, input_dim=vocabSize + 1,
                        input_length=128, mask_zero=True, weights=[embeddingWeights]))
    model.add(Bidirectional(LSTM(output_dim=100, return_sequences=False), merge_mode='sum'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))
    # print(model.summary())
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
    early_stopping = EarlyStopping(monitor="val_acc", patience=3)
    result = model.fit(Train_X, Train_Y, batch_size=128,
                       epochs=100,
                       validation_data=(Test_X, Test_Y),
                       callbacks=[early_stopping])
    sys.stdout.flush()
sys.stdout.close()
