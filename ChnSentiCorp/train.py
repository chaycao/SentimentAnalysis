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
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


'''----------------------训练模型-----------------------------------------'''
training_info_filePath = "./data/training.info"
training_data_filePath = "./data/training.data"
modelPath = "./data/keras_model"
weightPath = "./data/keras_model_weights"
word2vec_model_file = "./data/word2vec-128.model"

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
    y.append(0)
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

outputDims = 1
# Y_train = np_utils.to_categorical(train_y, outputDims)
# Y_test = np_utils.to_categorical(test_y, outputDims)
Y_train = train_y
Y_test = test_y

def create_model():
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
    # LSTM

    model = Sequential()
    model.add(Embedding(output_dim=embeddingDim, input_dim=vocabSize + 1,
                        input_length=maxlen, mask_zero=True, weights=[embeddingWeights]))
    model.add(Bidirectional(LSTM(output_dim=hiddenDims, return_sequences=False), merge_mode='sum'))
    model.add(Dropout(0.5))
    model.add(Dense(outputDims, activation="sigmoid"))
    print(model.summary())
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
    return model

early_stopping = EarlyStopping(monitor="val_acc", patience=3)

# result = model.fit(train_X, Y_train, batch_size=batchSize,
#                    epochs=100,
#                    validation_data=(test_X, Y_test),
#                    callbacks=[early_stopping])
seed = 7
start_time = time.time()
estimator = KerasClassifier(build_fn=create_model, nb_epoch=10, batch_size=128, verbose=1)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, y, cv=kfold) # 如果get_parameters方法报错,查看 https://github.com/fchollet/keras/pull/5121/commits/01c6b7180116d80845a1a6dc1f3e0fe7ef0684d8
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
end_time = time.time()
print ("用时: ", end_time - start_time)


# j = model.to_json()
# fd = open(modelPath, 'w')
# fd.write(j)
# fd.close()

# model.save_weights(weightPath)

print("Training used time : ", time.time() - start_time)
print ('Done!')