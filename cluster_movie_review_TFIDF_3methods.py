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

import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.datasets import make_moons  
from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN

X,y = make_moons(n_samples=200,noise=0.05,random_state=0)#创建半月形数据  
plt.scatter(X[:,0],X[:,1])  
plt.show()  
f,(ax1,ax2) = plt.subplots(1,2,figsize=(8,3))  
#原型聚类KMeans模型  
km=KMeans(n_clusters=2,random_state=0)  
y_km = km.fit_predict(X)  
ax1.scatter(X[y_km==0,0],X[y_km==0,1],c='lightblue',marker='o',s=40,label='cluster 1')  
ax1.scatter(X[y_km==1,0],X[y_km==1,1],c='red',marker='s',s=40,label='cluster 2')  
ax1.set_title('K-means clustering')  
#层次聚类凝聚模型  
ac=AgglomerativeClustering(n_clusters=2,affinity='euclidean',linkage='complete')#全连接，欧式距离计算联合矩阵  
y_ac= ac.fit_predict(X)  
ax2.scatter(X[y_ac==0,0],X[y_ac==0,1],c='lightblue',marker='o',s=40,label='cluster 1')  
ax2.scatter(X[y_ac==1,0],X[y_ac==1,1],c='red',marker='s',s=40,label='cluster 2')  
ax2.set_title('Agglomerative clustering')  
plt.legend()  
plt.show()  
#密度聚类DBSCAN，成功对半月形数据进行分类  
db =DBSCAN(eps=0.2,min_samples=5,metric='euclidean')#欧式距离，样本点数量5，半径0.2  
y_db = db.fit_predict(X)  
plt.scatter(X[y_db==0,0],X[y_db==0,1],c='lightblue',marker='o',s=40,label='cluster 1')  
plt.scatter(X[y_db==1,0],X[y_db==1,1],c='red',marker='s',s=40,label='cluster 2')  
plt.legend()  
plt.show() 
