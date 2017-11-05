# -*- coding: utf-8 -*-
'''
把training-shuffle.data切分成大小相同的十块数据
'''
import codecs
import jieba
import re
import h5py
import numpy as np
import random

def loadTrainingData(path):
    '''载入分词训练输入样本'''
    print('load training data from %s'%path)
    fd = h5py.File(path,'r')
    X = fd['X'][:]
    Y = fd['Y'][:]
    fd.close()
    return X,Y

def saveTrainingData(path, trainingData):
    '''保存分词训练输入样本'''
    print('save training data to %s' % path)
    # 采用hdf5保存大矩阵效率最高
    fd = h5py.File(path, 'w')
    (X, Y) = trainingData
    fd.create_dataset('X', data=X)
    fd.create_dataset('Y', data=Y)
    fd.close()

def splitdata(maxlen):
    path = './data/shuffledata/training-shuffle-'+str(maxlen)+'.data'
    X, Y = loadTrainingData(path)
    for i in range(10):
        x = X[i*600 : (i+1)*600]
        y = Y[i * 600: (i + 1) * 600]
        print(np.array(x).shape)
        print(np.array(y).shape)
        saveTrainingData('./data/seg/'+str(maxlen)+'/'+str(i)+'.data', (x,y))

splitdata(150)