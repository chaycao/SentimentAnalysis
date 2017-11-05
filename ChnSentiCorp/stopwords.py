# -*- coding: utf-8 -*-
import os
import codecs
"""
    合并文本文件
"""
mergefiledir = './data/stopwords'
filenames = os.listdir(mergefiledir)
file = codecs.open('./data/stopwords/stopwords.txt', 'w', 'utf-8')

for filename in filenames:
    filepath = mergefiledir + '\\' + filename
    for line in codecs.open(filepath, 'r', 'utf-8'):
        file.writelines(line)
    file.write('\n')

"""
    去重
"""
lines = codecs.open('./data/stopwords/stopwords.txt', 'r', 'utf-8')
newfile = codecs.open('./data/stopwords/stopwords_new.txt', 'w', 'utf-8')
new = []
for line in lines.readlines():
    if line not in new:
        new.append(line)
        newfile.writelines(line)

file.close()
newfile.close()