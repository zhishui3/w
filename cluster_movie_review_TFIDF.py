import nltk
from nltk.corpus import movie_reviews
from pylab import plot,show
from numpy import array
from numpy.random import rand
from scipy.cluster.vq import kmeans,vq,whiten
import numpy as np
import random

documents = [(' '.join(movie_reviews.words(fileid)), category) for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]  #转化为词列表的影评，与标签，组成二元组
random.shuffle(documents)	
documents_words=[w for (w,t) in documents]

from sklearn import feature_extraction  
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer  

vectorizer=CountVectorizer(min_df=20,stop_words='english')#该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频  
transformer=TfidfTransformer()#该类会统计每个词语的tf-idf权值  
tfidf=transformer.fit_transform(vectorizer.fit_transform(documents_words))#fit_transform计算tf-idf，fit_transform将文本转为词频矩阵  
word=vectorizer.get_feature_names()#获取词袋模型中的所有词语  
features=tfidf.toarray()#将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重  


target=[c for (d,c) in documents]

data=whiten(features)
centroids,_ = kmeans(data,2)
idx,_ = vq(data,centroids)

target1=[1 if x =='pos' else 0 for x in target]
a=sum(target1==idx)/len(target1)
print('scipy_eu=',max(a,1-a))

from nltk.cluster import KMeansClusterer,cosine_distance
clus=KMeansClusterer(2,cosine_distance)
results=clus.cluster(data,True,trace=False)
a=sum(np.array(target1)==results)/len(target1)
print('nltk_cosdis=',max(a,1-a))

from Bio.Cluster import kcluster
clusterid, error, nfound = kcluster(data,2,dist='c')
a=sum(target1==clusterid)/len(target1)
print('Bio_cos=',max(a,1-a))
