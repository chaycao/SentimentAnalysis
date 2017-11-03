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
from keras.callbacks import EarlyStopping,Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import sys
from keras import backend as K

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
#savedStdout = sys.stdout
#file = open('out.txt', 'w+')
#sys.stdout = file

# 加载词库
training_info_filePath = "./data/training.info"
modelPath = "./data/keras_model"
weightPath = "./data/keras_model_weights"
word2vec_model_file = "./data/word2vec-100.model"
trainingInfo = loadTrainingInfo(training_info_filePath)
print ('Load vocab Done!')
print ('Training model...')
start_time = time.time()
(vocab, indexVocab) = trainingInfo

# 加载词向量信息，作为Embedding层的权重
vocabSize = len(vocab) + 1
w2vModel = Word2Vec.load(word2vec_model_file)
embeddingDim = w2vModel.vector_size
print('词向量：'+embeddingDim)
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
    print("------- "+str(i+1)+"次 -------")
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
    print(model.summary())

    # 自定义f1值
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def f1(y_true, y_pred):
        def precision(y_true, y_pred):
            """Precision metric.

            Only computes a batch-wise average of precision.

            Computes the precision, a metric for multi-label classification of
            how many selected items are relevant.
            """
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision
        def recall(y_true, y_pred):
            """Recall metric.

            Only computes a batch-wise average of recall.

            Computes the recall, a metric for multi-label classification of
            how many relevant items are selected.
            """
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall
        precision = precision(y_true, y_pred)
        recall = recall(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall))


    class Metrics(Callback):
        def on_train_begin(self, logs={}):
            self.val_f1s = []
            self.val_recalls = []
            self.val_precisions = []
        def on_epoch_end(self, epoch, logs={}):
            val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
            val_targ = self.model.validation_data[1]
            _val_f1 = f1_score(val_targ, val_predict)
            _val_recall = recall_score(val_targ, val_predict)
            _val_precision = precision_score(val_targ, val_predict)
            self.val_f1s.append(_val_f1)
            self.val_recalls.append(_val_recall)
            self.val_precisions.append(_val_precision)
            print (" — val_f1: % f — val_precision: % f — val_recall % f" % (_val_f1, _val_precision, _val_recall))
            return

    metrics = Metrics()

    model.compile(loss="binary_crossentropy", optimizer='adam', metrics=[recall, precision, f1])
    
    early_stopping = EarlyStopping(monitor="val_f1", patience=3, mode='max')
    result = model.fit(Train_X, Train_Y, batch_size=128,
                       epochs=100,
                       validation_data=(Test_X, Test_Y),
                       callbacks=[early_stopping])
    #sys.stdout.flush()
#sys.stdout.close()
