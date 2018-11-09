import os
import codecs
import jieba
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


Sogou_class=['08','10']
m=0
for sc in Sogou_class:
        dirname1=r"E:\pywork\nlp\Sogou\Sogou\C0000"+sc+"\\"
        files=os.listdir(dirname1)

        fm=0
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
                        documents.append(' '.join(voca))
                        textnames.append(dirname1+fn)
                        fm+=1
                except:
                        continue
                if fm>1000:
                        break
 
stopwords=['(',')','，','。','？','“','”','‘','’','：','；','【','】','…','<','>','《','》','！']

vectorizer=CountVectorizer(min_df=10,stop_words=stopwords)#定义计数器  
transformer=TfidfTransformer()#该类会统计每个词语的tf-idf权值

tf=vectorizer.fit_transform(documents) #计算TF
tfidf=transformer.fit_transform(tf)#fit_transform计算tf-idf，fit_transform将文本转为词频矩阵 
vocab=vectorizer.get_feature_names()

import lda

model = lda.LDA(n_topics=20, n_iter=150, random_state=1)
model.fit(tf)

topic_words=model.topic_word_

# 显示每个话题的前10词
n = 10
for i, t in enumerate(topic_words):
    topicwords = np.array(vocab)[np.argsort(-t)][:n]
    print('*Topic {}\n- {}'.format(i, ' '.join(topicwords)))


# 显示前10个文本的前K个话题
k=10
doc_topic = model.doc_topic_
for i in range(10):
    topic_most_pr = doc_topic[i].argsort()
    print("doc: {} topic: {}".format(textnames[i], topic_most_pr[:k]))

