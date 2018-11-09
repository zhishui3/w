# keywords extraction

# prepare data
import os
import codecs
import thulac
import re
from sklearn import feature_extraction  
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer  
import numpy as np
from scipy import spatial
from scipy.cluster.vq import kmeans,vq,whiten

documents=[]
terms=['']*1000000
textnames=[]


f=open(r'E:\语言信息处理/stopwords-utf8.txt','r',encoding='utf-8')
raw=f.read()
f.close()
stopwords=list(set(raw.split()+['(',')','，','。','？','“','”','‘','’','：','；','【','】','…','《','■','》','_','<','>','！']))
candpos=['n','np','ns','ni','nz','v','a','d','j','x']


thu1= thulac.thulac()


m=0
dirname1=r"E:\语言信息处理\2017_autumn_nlp\forkeywords\political_text\\"
files=os.listdir(dirname1)
candidates=[[]]*len(files)
fm=0

for fn in files:
        f= codecs.open(dirname1+fn,mode='r')#,encoding="utf-8")
        text=f.readlines()
        f.close()
        voca=[]

        for t in text:
                
                words=re.sub(r'[\r\t\n\u3000]','',t)
                words=thu1.cut(words)
                cw=[w[0] for w in words if w[0] not in stopwords and w[-1] in candpos ]
                candidates[fm]=candidates[fm]+cw

                words=[w[0] for w in words if re.match('^\D',w[0])]
                voca=voca+words
                terms[m:m+len(words)]=words
                m=m+len(words)

        documents.append(' '.join(voca))
        textnames.append(dirname1+fn)
        candidates[fm]=list(set(candidates[fm]))
        fm+=1

candidates[fm:]=[]


# compute feature

vectorizer=CountVectorizer(min_df=5,stop_words=stopwords)#定义计数器  
transformer=TfidfTransformer()#该类会统计每个词语的tf-idf权值

tf=vectorizer.fit_transform(documents) #计算TF
tfidf=transformer.fit_transform(tf)#fit_transform计算tf-idf，fit_transform将文本转为词频矩阵 
word=vectorizer.get_feature_names()#获取词袋模型中的所有词语  
tfidfmat=tfidf.toarray()
tfmat=tf.toarray()

# step 2

## TFIDF
K=5
for n in range(np.shape(tfidfmat)[0]):
    data=[(w,tfidfmat[n,word.index(w)]) for w in candidates[n] if w in word]
    data.sort(key=lambda x:x[1],reverse=True)    
    print("keywords {}".format([w for (w,t) in data[:K]]))

import lda

model = lda.LDA(n_topics=fm, n_iter=150, random_state=1)
model.fit(tf)

# 显示每个话题的前10词

# 显示前10个文本的前K个话题

doc_topic = model.doc_topic_
topic_word=model.topic_word_
for i in range(len(doc_topic)):
        topic_most_pr = doc_topic[i].argsort()
        keywords=[topic_word[topic_most_pr[n]].argmax() for n in range(K)] ##话题中概率最大的词
        print('*keywords {}'.format([word[n] for n in keywords]))
    

### cluster candidates words by topic/svd

from scipy import spatial
from scipy.cluster.vq import kmeans,vq,whiten

word_topic=topic_word.transpose()# 词-话题向量

for n in range(len(doc_topic)):
        keywords=[]
        data=[(w,word.index(w))  for w in candidates[n] if w in word]
        cand_vec=word_topic[[w[1] for w in data],:]# 候选词-话题向量
        centroids,_ = kmeans(whiten(cand_vec),K)
        for i in range(K):
                min_dist=100
                near_word=-1
                for j in range(len(cand_vec)):
                        a=np.dot(centroids[i,:],cand_vec[j,:])
                        if a<=min_dist and j not in keywords:
                                min_dist=a
                                near_word=j
                keywords.append(near_word)
        keywords=[data[w][0] for w in keywords]
        print('*keywords {}'.format(keywords))

    
