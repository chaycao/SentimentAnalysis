# -*- coding: utf-8 -*-
import os
import codecs
import jieba
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate():
    '''
    把6000个文件变成一个文本
    :return:
    '''
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

def cut():
    '''
    对pos.txt neg.txt，分词，去停用词
    '''
    neg_inFile = codecs.open('./data/seg/neg.txt','r','utf-8')
    neg_outFile = codecs.open('./data/seg/neg_cut.txt','w','utf-8')
    stoplist = codecs.open('./data/stopwords/stopwords.txt','r','utf-8')
    stoplist = set(w.strip() for w in stoplist)
    for line in neg_inFile:
        seg_list = jieba.cut(line, cut_all=False)
        seg_list = [word for word in list(seg_list) if word not in stoplist]
        out = re.sub(r"\s{2,}", " ", " ".join(seg_list))
        neg_outFile.write(out+"\n")
    neg_inFile.close()
    neg_outFile.close()

    pos_inFile = codecs.open('./data/seg/pos.txt', 'r', 'utf-8')
    pos_outFile = codecs.open('./data/seg/pos_cut.txt', 'w', 'utf-8')
    for line in pos_inFile:
        seg_list = jieba.cut(line, cut_all=False)
        seg_list = [word for word in list(seg_list) if word not in stoplist]
        out = re.sub(r"\s{2,}", " ", " ".join(seg_list))
        pos_outFile.write(out + "\n")
    pos_inFile.close()
    pos_outFile.close()

# cut()

def merge():
    file_pos = codecs.open('./data/seg/pos_cut.txt', 'r', 'utf-8')
    file_neg = codecs.open('./data/seg/neg_cut.txt', 'r', 'utf-8')
    outFile = codecs.open('./data/seg/all_cut.txt', 'w', 'utf-8')
    for line in file_pos:
        line = line.strip()
        if (len(line) > 0):
            outFile.write(line + '\n')
    for line in file_neg:
        line = line.strip()
        if (len(line) > 0):
            outFile.write(line + '\n')
merge()
def cutLen(maxlen):
    '''
    把正负样本合并,大于maxlen的部分去除
    :param maxlen:
    :return:
    '''
    file_pos = codecs.open('pos_cut.txt', 'r', 'utf-8')
    file_neg = codecs.open('neg_cut.txt', 'r', 'utf-8')
    outFile = codecs.open('all-' + maxlen + '.txt', 'w', 'utf-8')
    for line in file_pos:
        line = line.strip()
        if (len(line) > 0):
            line = line[0:maxlen]
            print(len(line))
            outFile.write(line + '\n')
    for line in file_neg:
        line = line.strip()
        if (len(line) > 0):
            line = line[0:maxlen]
            print(len(line))
            outFile.write(line + '\n')
