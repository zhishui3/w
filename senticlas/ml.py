import nltk
from nltk.corpus import movie_reviews
documents = [(list(movie_reviews.words(fileid)), category) for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]  #转化为词列表的影评，与标签，组成二元组
documents = [(list(movie_reviews.words(fileid)), category) for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]  #转化为词列表的影评，与标签，组成二元组
import jieba
dirname=r"review_sentiment\train2\\"
f=open(r'review_sentiment\train2.rlabelclass','r')
raw=f.read()
f.close()
file_names=raw.split('\n')

file_names=[(f.split(' ')[0], f.split(' ')[1] )for f in file_names if len(f)>0]
files=[f for (f,s) in file_names]   #文件名
target=[]   # 句子的人工情感标记
fn=0
fm=0 #正确读入文件数量
for fn in range(len(files)):
    try:
        f= open(dirname+files[fn],mode='r')#,encoding="utf-8")
        raw=f.read()
        f.close()
        raw=re.sub(r'[\r\n\u3000]','',raw)
        words=jieba.cut(raw)
        target.append(words)
        fm+=1
        if fm>2000:
            break
    except:
        continue
        
import random
random.shuffle(documents)	
# dwords = ' '.join(documents)
all_words = nltk.FreqDist(w.lower() for w in target)
all_words=all_words.most_common(2000)	#词频表按频率排序
dirname=r"review_sentiment\test2\\"
f=open(r'review_sentiment\test2.rlabelclass','r')
raw=f.read()
f.close()
file_names=raw.split('\n')

file_names=[(f.split(' ')[0], f.split(' ')[1] )for f in file_names if len(f)>0]
files=[f for (f,s) in file_names]   #文件名
target_test=[]   # 句子的人工情感标记
fn=0
fm=0 #正确读入文件数量
for fn in range(len(files)):
    try:
        f= open(dirname+files[fn],mode='r')#,encoding="utf-8")
        raw=f.read()
        f.close()
        raw=re.sub(r'[\r\n\u3000]','',raw)
        words=jieba.cut(raw)
        target_test.append(words)
        fm+=1
        if fm>2000:
            break
    except:
        continue
f=open(r'stopwords.txt','r',encoding='utf-8')
raw=f.read()
f.close()
punc=['(',')','，','。','？','“','”','‘','’','：','；','【','】','…','《','■','》','_','<','>','！',',',  '!', '.',  '、', '~', '）', '（', ';', ':']
sent_mark=['，','。','？','！',',',  '!', '.']
stop_words=raw.split()


word_features  =[w for (w,f) in all_words if w not in stop_words]

import numpy as np
features = np.zeros([len(documents),len(word_features)],dtype=float)
for n in range(len(documents)):
        document_words = set(documents[n][0])
        for  m in range(len(word_features)):
                if word_features[m] in document_words:
                        features[n,m] = 1 # 文件-词集矩阵


train_set=features

test_set=features


from sklearn.svm import SVC
svclf = SVC(kernel ='linear')#‘rbf’,‘linear’, ‘poly’, ‘sigmoid’, ‘precomputed’
svclf.fit(train_set,target)  
pred = svclf.predict(test_set)

print(sum([1 for n in range(len(target_test)) if pred[n]==target_test[n]])/len(target_test))

from sklearn.neighbors import KNeighborsClassifier
knnclf = KNeighborsClassifier(n_neighbors=7)#default with k=5  
knnclf.fit(train_set,target)  
pred = knnclf.predict(test_set)
print(sum([1 for n in range(len(target_test)) if pred[n]==target_test[n] ])/len(target_test))
