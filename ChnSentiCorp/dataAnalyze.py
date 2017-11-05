# -*- coding: utf-8 -*-

import codecs
import pandas as pd
import numpy as np
import matplotlib.pylab as plt


file = codecs.open('./data/seg/all_cut.txt','r','utf-8')
data = []
i = 0
for line in file:
    data.append(line.split(" ").__len__())
data = pd.DataFrame(data)
plt.hist(data, bins=50, range=(0,250), edgecolor='black')
plt.show()