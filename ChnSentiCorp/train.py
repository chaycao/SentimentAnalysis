# -*- coding: utf-8 -*-

import multiprocessing
import time
import h5py
import numpy as np
import random
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
from pretreat import loadTrainingInfo, loadTrainingData
'''----------------------训练模型-----------------------------------------'''
training_info_filePath = "./data/training.info"
training_data_filePath = "./data/training.data"
modelPath = "./data/keras_model"
weightPath = "./data/keras_model_weights"
word2vec_model_file = "./data/word2vec.model"

print ('Loading vocab...')
start_time = time.time()
trainingInfo = loadTrainingInfo(training_info_filePath)
trainingData = loadTrainingData(training_data_filePath)
print("Loading used time : ", time.time() - start_time)
print ('Done!')

print ('Training model...')
start_time = time.time()

(vocab, indexVocab) = trainingInfo
X = trainingData
y = []
for i in range(0, 3000):
    y.append(1)
for i in range(0, 2999):
    y.append(-1)
X = X.reshape(-1, 128)
# 打乱数据
index = [i for i in range(5999)]
random.shuffle(index)
X = X[index]
y = np.array(y)
y = y[index]
print(X.size)
train_X, test_X, train_y, test_y = model_selection.train_test_split(X, y, train_size=0.9, random_state=1)
train_X = np.array(train_X)
train_y = np.array(train_y)
test_X = np.array(test_X)
test_y = np.array(test_y)

outputDims = 2
Y_train = np_utils.to_categorical(train_y, outputDims)
Y_test = np_utils.to_categorical(test_y, outputDims)
batchSize = 128
vocabSize = len(vocab) + 1
wordDims = 100
maxlen = 128
hiddenDims = 100

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

print("x_shape:" + str(train_X.shape))
print("y_shape:" + str(Y_train.shape))
# LSTM
model = Sequential()
model.add(Embedding(output_dim=embeddingDim, input_dim=vocabSize + 1,
                    input_length=maxlen, mask_zero=True, weights=[embeddingWeights]))
model.add(Bidirectional(LSTM(output_dim=hiddenDims, return_sequences=False), merge_mode='sum'))
model.add(Dropout(0.5))
model.add(Dense(outputDims, activation="softmax"))
model.add(Dropout(0.5))
print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

early_stopping = EarlyStopping(monitor="val_acc", patience=3)

result = model.fit(train_X, Y_train, batch_size=batchSize,
                   epochs=100,
                   validation_data=(test_X, Y_test),
                   callbacks=[early_stopping])

j = model.to_json()
fd = open(modelPath, 'w')
fd.write(j)
fd.close()

model.save_weights(weightPath)

print("Training used time : ", time.time() - start_time)
print ('Done!')