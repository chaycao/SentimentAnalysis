# -*- coding: utf-8 -*-

import codecs
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

def histSegLen():
    '''
    样本单词长度
    '''
    file = codecs.open('./data/seg/all_cut.txt','r','utf-8')
    data = []
    i = 0
    for line in file:
        data.append(line.split(" ").__len__())
    data = pd.DataFrame(data)
    plt.xlabel('样本长度（个）')
    plt.ylabel('数量（个）')
    plt.title('数据集中样本长度情况')
    plt.hist(data, bins=50, range=(0,250), edgecolor='black')
    plt.savefig('./figure/seglen.png', dpi=600)
    plt.show()
    
def lineWordMaxLenF1():
    '''
    MaxLen对F1的影响
    '''
    f1 = [
        0.88257,
        0.89997,
        0.90666,
        0.90143,
        0.90237,
        0.90257,
        0.90182,
        0.90183,
        0.89733,
        0.90015]
    f1= np.array(f1)
    maxlen = np.linspace(25,250,10)
    plt.figure()
    plt.plot(maxlen, f1, '-o')
    plt.xlim((0, 275))
    plt.xticks(np.linspace(0, 250, 11))
    plt.xlabel('MaxLen')
    plt.ylabel('F1')
    plt.title('MaxLen对F1的影响')
    plt.savefig('./figure/MaxLen.png', dpi=600)
    plt.show()

def lineWordMaxLenTime():
    '''
    MaxLen对F1的影响
    '''
    f1 = [
        436,
        818,
        1337,
        1487,
        1716,
        3237,
        3411,
        4578,
        13696,
        12516]
    f1= np.array(f1)
    maxlen = np.linspace(25,250,10)
    plt.figure()
    plt.plot(maxlen, f1, '-o')
    plt.xlim((0, 275))
    plt.xticks(np.linspace(0, 250, 11))
    plt.xlabel('MaxLen')
    plt.ylabel('时间(s)')
    plt.title('MaxLen对时间的影响')
    plt.savefig('./figure/MaxLenTime.png', dpi=600)
    plt.show()

lineWordMaxLenTime()