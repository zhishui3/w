import nltk
from nltk.corpus import movie_reviews
from pylab import plot,show
from numpy import array
from numpy.random import rand
from scipy.cluster.vq import kmeans,vq,whiten
import numpy as np
import random
import collections

documents = [(' '.join(movie_reviews.words(fileid)), category) for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]  #转化为词列表的影评，与标签，组成二元组
random.shuffle(documents)	
documents_words=[w for (w,t) in documents]

from sklearn import feature_extraction  
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer  

vectorizer=CountVectorizer(min_df=100,stop_words='english')#该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频  
transformer=TfidfTransformer()#该类会统计每个词语的tf-idf权值

tfidf=transformer.fit_transform(vectorizer.fit_transform(documents_words))#fit_transform计算tf-idf，fit_transform将文本转为词频矩阵  

word=vectorizer.get_feature_names()#获取词袋模型中的所有词语  

features=tfidf.toarray()#将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
print(features.shape)

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
model = kNN.train(train_set1, target_train, 7)
dd=[kNN.classify(model, t, distance_fn=spatial.distance.cosine) for t in test_set1]
print('KNN_cos=',sum(np.array(dd)==np.array(target_test))/len(target_test))

from sklearn.neighbors import KNeighborsClassifier
knnclf = KNeighborsClassifier(n_neighbors=7)#default with k=5  
knnclf.fit(train_set1, target_train)  
pred_knn = knnclf.predict(test_set1);
print('KNN_eu=',sum(pred_knn==target_test)/len(target_test))



