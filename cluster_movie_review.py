import nltk
from nltk.corpus import movie_reviews
from pylab import plot,show
from numpy import array
from numpy.random import rand
from scipy.cluster.vq import kmeans,vq,whiten
import numpy as np
import random

documents = [(list(movie_reviews.words(fileid)), category) for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]  #转化为词列表的影评，与标签，组成二元组
random.shuffle(documents)	

all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())	 #建立全部影评的词频表
all_words=all_words.most_common(2000)			#词频表按频率排序
stopwords = nltk.corpus.stopwords.words('english')
word_features  =[w for (w,f) in all_words if w not in stopwords] 		#特征词为词频表中前2000词

features = np.zeros([len(documents),len(word_features)],dtype=float)
for n in range(len(documents)):
        document_words = set(documents[n][0])
        for  m in range(len(word_features)):
                if word_features[m] in document_words:
                        features[n,m] = 1

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
