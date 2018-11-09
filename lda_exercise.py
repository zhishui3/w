import lda
import numpy as np
import matplotlib.pyplot as plt
from sklearn import feature_extraction  
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer  

documents=['Human machine interface for ABC computer applications',
'A survey of user opinion of computer system response time',
'The EPS user interface management system',
'System and human system engineering testing of EPS',
'Relation of user perceived response time to error measurement',
'The generation of random, binary, ordered trees',
'The intersection graph of paths in trees',
'Graph minors IV: Widths of trees and well-quasi-ordering',
'Graph minors: A survey']

stopwords=['a','the','of','to','for','and']
vectorizer=CountVectorizer(min_df=2,stop_words=stopwords)#该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频  
transformer=TfidfTransformer()#该类会统计每个词语的tf-idf权值
tf=vectorizer.fit_transform(documents)
tfidf=transformer.fit_transform(tf)#fit_transform计算tf-idf，fit_transform将文本转为词频矩阵  
vocab=vectorizer.get_feature_names()

titles=['c1','c2','c3','c4','c5','m1','m2','m3','m4']
model = lda.LDA(n_topics=4, n_iter=150, random_state=1)
model.fit(tf)

model.doc_topic_
model.topic_word_

plt.plot(model.topic_word_.transpose())
plt.xlabel(vocab)
plt.legend(['topic1','topic2','topic3','topic4'])
plt.show()

plt.plot(model.doc_topic_.transpose())
plt.legend(titles)
plt.xlabel(['topic1','topic2','topic3','topic4'])
plt.show()


# SVD
from scipy import linalg

u,s,v=linalg.svd(tf.toarray())
s4=s
s4[4:]=0
d4=linalg.diagsvd(s4,len(u),len(v))

a4=np.dot(u,np.dot(d4,v))

us_sp=np.dot(u,d4)
vs=np.dot(d4,v)
vs_sp=vs.transpose()
