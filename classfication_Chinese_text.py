import nltk
from nltk.corpus import movie_reviews
from pylab import plot,show
from numpy import array
from numpy.random import rand
from scipy.cluster.vq import kmeans,vq,whiten
import numpy as np
import random
import os
import codecs
import jieba
import re

documents=[]
terms=['']*1000000
dirname1=r"E:\语言信息处理\2017_autumn_nlp\corpus for classifier\Sogou\C000008\\"
files=os.listdir(dirname1)
m=0

for fn in files:
        try:
                f= codecs.open(dirname1+fn,mode='r')
                text=f.readlines()
                f.close()
                voca=[]
                for t in text:
                        words=re.sub(r'[\r\t\n\u3000]','',t)
                        words=list(jieba.cut(words))
                        voca=voca+words
                        terms[m:m+len(words)]=words
                        m=m+len(words)
                documents.append([voca,'c08'])
        except:
                continue
dirname1=r"E:\语言信息处理\2017_autumn_nlp\corpus for classifier\Sogou\C000010\\"
files=os.listdir(dirname1)

for fn in files:
        try:
                f= codecs.open(dirname1+fn,mode='r')
                text=f.readlines()
                f.close()
                voca=[]
                for t in text:
                        words=re.sub(r'[\r\t\n\u3000]','',t)
                        words=list(jieba.cut(words))
                        voca=voca+words
                        terms[m:m+len(words)]=words
                        m=m+len(words)
                documents.append([voca,'c10'])
        except:
                continue        
terms=terms[:m]
random.shuffle(documents)	
#raise Exception
all_words = nltk.FreqDist(terms)	 #建立全部影评的词频表
all_words=all_words.most_common(2000)			#词频表按频率排序
#stopwords = nltk.corpus.stopwords.words('english')
word_features  =[w for (w,f) in all_words]# if w not in stopwords] 		#特征词为词频表中前2000词

def document_features(document):		#建立特征提取器，标定该影评是否有特征词
	document_words = set(document)
	features = {}
	for word in word_features:
		features['contains(%s)' % word] = (word in document_words)
	return features

featuresets = [(document_features(d), c) for (d,c) in documents]
train_set=featuresets[:1000]
test_set=featuresets[1000:]
from nltk.classify import MaxentClassifier

me_classifier = MaxentClassifier.train(train_set, algorithm='iis', trace=0, max_iter=1, min_lldelta=0.5)
print('Maxentropy=',nltk.classify.accuracy(me_classifier, test_set))

by_classifier = nltk.NaiveBayesClassifier.train(train_set)
print('NaiveBayes=',nltk.classify.accuracy(by_classifier, test_set))

from nltk.classify import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier.train(train_set, binary=True, entropy_cutoff=0.8, depth_cutoff=50, support_cutoff=30)
print('DecisionTree=',nltk.classify.accuracy(dt_classifier, test_set))

		#建立特征提取器，标定该影评是否有特征词

features = np.zeros([len(documents),len(word_features)],dtype=float)
for n in range(len(documents)):
        document_words = set(documents[n][0])
        for  m in range(len(word_features)):
                if word_features[m] in document_words:
                        features[n,m] = 1

target=[c for (d,c) in documents]
train_set1=features[:1000,:]
target_train=target[:1000]
test_set1=features[1000:,:]
target_test=target[1000:]

from sklearn.svm import SVC
svclf = SVC(kernel ='linear')
svclf.fit(train_set1, target_train)  
pred_svc = svclf.predict(test_set1)
print('SVM=',sum(pred_svc==target_test)/len(target_test))

from sklearn.neighbors import KNeighborsClassifier
knnclf = KNeighborsClassifier(n_neighbors=7)#default with k=5  
knnclf.fit(train_set1, target_train)  
pred_knn = knnclf.predict(test_set1);
print('KNN_eu=',sum(pred_knn==target_test)/len(target_test))

from Bio import kNN
from scipy import spatial
model = kNN.train(train_set1, target_train, 7)
dd=[kNN.classify(model, t, distance_fn=spatial.distance.cosine) for t in test_set1]
print('KNN_cos=',sum(np.array(dd)==np.array(target_test))/len(target_test))
