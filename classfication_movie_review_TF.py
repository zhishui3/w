import nltk
from nltk.corpus import movie_reviews
from pylab import plot,show
from numpy import array
from numpy.random import rand
from scipy.cluster.vq import kmeans,vq,whiten
import numpy as np
import random
import collections

documents = [(list(movie_reviews.words(fileid)), category) for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]  #转化为词列表的影评，与标签，组成二元组
random.shuffle(documents)	

all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())	 #建立全部影评的词频表
all_words=all_words.most_common(2000)			#词频表按频率排序
stopwords = nltk.corpus.stopwords.words('english')
word_features  =[w for (w,f) in all_words if w not in stopwords] 		#特征词为词频表中前2000词


features = np.zeros([len(documents),len(word_features)],dtype=float)
for n in range(len(documents)):
        document_words = documents[n][0]
        pdf=collections.Counter( document_words)
        for  m in range(len(word_features)):
                if word_features[m] in document_words:
                        features[n,m] = pdf[word_features[m]]

target=[c for (d,c) in documents]
train_set1=features[:1500,:]
target_train=target[:1500]
test_set1=features[1500:,:]
target_test=target[1500:]

from sklearn.svm import SVC
svclf = SVC(kernel ='linear',probability=True)
svclf.fit(train_set1, target_train)  
pred_svc = svclf.predict(test_set1)
print('SVM=',sum(pred_svc==target_test)/len(target_test))


from Bio import kNN
from scipy import spatial
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(7,)
model = kNN.train(train_set1, target_train, 7)
dd=[kNN.classify(model, t, distance_fn=spatial.distance.cosine) for t in test_set1]
print('KNN_cos=',sum(np.array(dd)==np.array(target_test))/len(target_test))

from sklearn.neighbors import KNeighborsClassifier
knnclf = KNeighborsClassifier(n_neighbors=7)#default with k=5  
knnclf.fit(train_set1, target_train)  
pred_knn = knnclf.predict(test_set1);
print('KNN_eu=',sum(pred_knn==target_test)/len(target_test))

from sklearn.naive_bayes import GaussianNB
nbclf = GaussianNB()
nbclf.fit(train_set1, target_train)  
pred_nb = nbclf.predict(test_set1);
print('NB=',sum(pred_nb==target_test)/len(target_test))

from sklearn import tree
dtclf = tree.DecisionTreeClassifier()
dtclf.fit(train_set1, target_train)  
pred_dt = dtclf.predict(test_set1)
print('DT=',sum(pred_dt==target_test)/len(target_test))

print(" results of SOW ")
features = np.zeros([len(documents),len(word_features)],dtype=float)
for n in range(len(documents)):
        document_words = set(documents[n][0])
        for  m in range(len(word_features)):
                if word_features[m] in document_words:
                        features[n,m] = 1
target=[c for (d,c) in documents]
train_set1=features[:1500,:]
target_train=target[:1500]
test_set1=features[1500:,:]
target_test=target[1500:]

from sklearn.svm import SVC
svclf = SVC(kernel ='linear',probability=True)
#svclf = SVC(kernel ='linear')#‘rbf’,‘linear’, ‘poly’, ‘sigmoid’, ‘precomputed’
svclf.fit(train_set1, target_train)  
pred_svc = svclf.predict(test_set1)
print('SVM=',sum(pred_svc==target_test)/len(target_test))


from Bio import kNN
from scipy import spatial
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(7,)
model = kNN.train(train_set1, target_train, 7)
dd=[kNN.classify(model, t, distance_fn=spatial.distance.cosine) for t in test_set1]
print('KNN_cos=',sum(np.array(dd)==np.array(target_test))/len(target_test))

from sklearn.neighbors import KNeighborsClassifier
knnclf = KNeighborsClassifier(n_neighbors=7)#default with k=5  
knnclf.fit(train_set1, target_train)  
pred_knn = knnclf.predict(test_set1);
print('KNN_eu=',sum(pred_knn==target_test)/len(target_test))

from sklearn.naive_bayes import GaussianNB
nbclf = GaussianNB()
nbclf.fit(train_set1, target_train)  
pred_nb = nbclf.predict(test_set1);
print('NB=',sum(pred_nb==target_test)/len(target_test))

from sklearn import tree
dtclf = tree.DecisionTreeClassifier()
dtclf.fit(train_set1, target_train)  
pred_dt = dtclf.predict(test_set1)
print('DT=',sum(pred_dt==target_test)/len(target_test))
