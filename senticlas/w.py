from collections import Counter
import numpy as np
import time
import sys
import numpy as np
import random
f=open(r'E:\pywork\work\senticlas\review_sentiment\train2.rlabelclass','r')
raw=f.read()
f.close()
file_names=raw.split('\n')

file_names=[(f.split(' ')[0], f.split(' ')[1] )for f in file_names if len(f)>0]
random.shuffle(file_names)
labels=[s for (f,s) in file_names] #文本的情感标记
files=[f for (f,s) in file_names]   #文件名
dirname=r"E:\pywork\work\senticlas\review_sentiment\train2\\"
reviews = list()
for fn in range(len(files)):
    try:
        f= open(dirname+files[fn],mode='r')#,encoding="utf-8")
        raw=f.read()
        f.close()
#         raw=re.sub(r'[\r\n\u3000]','',raw)
        reviews.append(raw)
#         print(reviews)
    except:
        continue
maxlen = 100 #截断词数
min_count = 5 #出现次数少于该值的词扔掉。这是最简单的降维方法
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding
from keras.layers import LSTM
 
#建立模型
model = Sequential()
model.add(Embedding(len(reviews), 256, input_length=maxlen))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
 
batch_size = 128
train_num = 15000
f=open(r'E:\pywork\work\senticlas\review_sentiment\test2.rlabelclass','r')
raw=f.read()
f.close()
file_names=raw.split('\n')

file_names=[(f.split(' ')[0], f.split(' ')[1] )for f in file_names if len(f)>0]
random.shuffle(file_names)
labls=[s for (f,s) in file_names] #文本的情感标记
files=[f for (f,s) in file_names]   #文件名
dirname=r"E:\pywork\work\senticlas\review_sentiment\test2\\"
re = list()
for fn in range(len(files)):
    try:
        f= open(dirname+files[fn],mode='r')#,encoding="utf-8")
        raw=f.read()
        f.close()
#         raw=re.sub(r'[\r\n\u3000]','',raw)
        re.append(raw)
#         print(reviews)
    except:
        continue
model.fit(reviews,labels, batch_size = batch_size, nb_epoch=30)
 
model.evaluate(re,labls, batch_size = batch_size)
