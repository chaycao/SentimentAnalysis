# -*- coding: utf-8 -*-


import os
import codecs
import jieba
import re
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import time
from gensim.models import Word2Vec
import multiprocessing
import multiprocessing
import time
import h5py
import numpy as np
from sklearn import model_selection
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers import Bidirectional, TimeDistributed
from keras.callbacks import EarlyStopping
from gensim.models import Word2Vec


'''
把6000的各个文件变成一个文本
'''
def generate():
    filepath = './6000/neg/'
    outpath = 'neg_test.txt'
    pathDir = os.listdir(filepath)
    outFile = codecs.open(outpath, 'w', 'utf-8')
    for allDir in pathDir:
        child = os.path.join('%s%s' % (filepath, allDir))
        inFile = codecs.open(child, 'r', 'utf-8')
        for eachLine in inFile:
            eachLine = eachLine.strip()
            if len(eachLine) != 0 :
                outFile.write(eachLine)
        outFile.write('\n')
        outFile.flush()
        inFile.close()
    outFile.close()

# generate()

def cut():
    inFile = codecs.open('./data/neg.txt','r','utf-8')
    outFile = codecs.open('./data/neg_cut.txt','w','utf-8')
    for line in inFile:
        seg_list = jieba.cut(line, cut_all=False)
        out = re.sub(r"\s{2,}", " ", " ".join(seg_list))
        outFile.write(out+"\n")
    inFile.close()
    outFile.close()


'''
neg:
max: 1761
avg:107.459

pos:
max: 717
avg:66.265
'''
def analyze():
    file = codecs.open('pos_cut.txt','r','utf-8')
    max = 0
    count = 0
    for line in file:
        length = len(line.split(" "))
        if (length > max):
            max = length
        count += length
    print('max: ' + str(max))
    print('avg:' + str(count/3000))
    file.close()

def merge():
    file_pos = codecs.open('pos_cut.txt', 'r', 'utf-8')
    file_neg = codecs.open('neg_cut.txt', 'r', 'utf-8')
    outFile = codecs.open('all.txt', 'w', 'utf-8')
    for line in file_pos:
        line = line.strip()
        if (len(line) > 0):
            line = line[0:128]
            print(len(line))
            outFile.write(line + '\n')
    for line in file_neg:
        line = line.strip()
        if (len(line) > 0):
            line = line[0:128]
            print(len(line))
            outFile.write(line + '\n')

def word2vec():
    start_time = time.time()
    input_file = 'all.txt'
    output_model_file = 'word2vec.model'
    output_vector_file = 'word2vec.vector'

    model = Word2Vec(LineSentence(input_file), size=128, window=5, min_count=5,
                     workers=multiprocessing.cpu_count())

    # trim unneeded model memory = use(much) less RAM
    # model.init_sims(replace=True)
    model.save(output_model_file)
    model.wv.save_word2vec_format(output_vector_file, binary=False)

    end_time = time.time()
    print("used time : %d s" % (end_time - start_time))

#-------------------------- 词典统计 --------------------------------------------------#

retain_unknown = 'retain-unknown'
retain_empty = 'retain-empty' # 用于填充

def saveTrainingData(path, trainingData):
    '''保存分词训练输入样本'''
    print('save training data to %s' % path)
    # 采用hdf5保存大矩阵效率最高
    fd = h5py.File(path, 'w')
    X = trainingData
    fd.create_dataset('X', data=X)
    fd.close()

def sent2vec2(sent, vocab, num):
    charVec = []
    for char in sent:
        if char in vocab:
            charVec.append(vocab[char])  # 字在vocabIndex中的索引
        else:
            charVec.append(vocab[retain_unknown])
    # 填充到指定长度
    while len(charVec) < num:
        charVec.append(vocab[retain_empty])
    return charVec

def doc2vec(fname, vocab):
    #一次性读入文件，注意内存
    fd = codecs.open(fname, 'r', 'utf-8')
    lines = fd.readlines()
    fd.close()
    #样本集
    X = []
    #遍历行
    for line in lines:
        #按空格分割
        words = line.strip().split()
        #每行的分词信息
        chars = [] # 存一个个的词
        for word in words:
            chars.append(word)
        #将句子转成词向量，长度短的，填充到指定长度
        lineVecX = sent2vec2(chars, vocab, 128)
        # 理论上说，X应该都是128维的
        X.extend(lineVecX)
    return X

def vocabAddChar(vocab, indexVocab, index, char):
    if char not in vocab:
        vocab[char] = index
        indexVocab.append(char)
        index += 1
    return index

def genVocab(fname, delimiters = [' ', '\n']):
    fd = codecs.open(fname, 'r', 'utf-8')
    lines = fd.readlines()
    fd.close()
    vocab = {}      # 词: 在indexVocab中的索引
    indexVocab = [] # 词
    index = 0
    for line in lines:
        words = line.strip().split()
        if len(words) <= 0: continue
        #遍历所有词
        #如果为分隔符则无需加入字典
        for word in words:
            if word not in delimiters:
                index = vocabAddChar(vocab, indexVocab, index, word)
    #加入未登陆新词和填充词
    vocab[retain_unknown] = len(vocab)
    vocab[retain_empty] = len(vocab)
    indexVocab.append(retain_unknown)
    indexVocab.append(retain_empty)
    #返回字典与索引
    return vocab, indexVocab

def load(fname):
    print ('train from file', fname)
    delims = [' ', '\n']
    vocab, indexVocab = genVocab(fname)
    X = doc2vec(fname, vocab)
    print (len(X))
    return X, (vocab, indexVocab)

def saveTrainingInfo(path, trainingInfo):
    '''保存分词训练数据字典和概率'''
    print('save training info to %s'%path)
    fd = open(path, 'w')
    (vocab, indexVocab) = trainingInfo
    for char in vocab:
        fd.write(str(char.encode('utf-8')) + '\t' + str(vocab[char]) + '\n')
    fd.close()

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


def dictGenerate():
    start_time = time.time()
    input_file = "./data/all.txt"
    training_info_filePath = "./data/training.info"
    training_data_filePath = "./data/training.data"
    X, (vocab, indexVocab) = load(input_file)
    # TrainInfo：词向量和词典的相关情况
    saveTrainingInfo(training_info_filePath, (vocab, indexVocab))
    # TrainData：将字表示为向量和标记
    saveTrainingData(training_data_filePath, X)
    end_time = time.time()
    print("used time : %d s" % (end_time - start_time))

dictGenerate()