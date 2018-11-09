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

f=open(r'E:\语言信息处理/stopwords-utf8.txt','r',encoding='utf-8')
raw=f.read()
f.close()
stopwords=list(set(raw.split()+['(',')','，','。','？','“','”','‘','’','：','；','【','】','…','《','■','》','_','<','>','！']))
candpos=['n','np','ns','ni','nz','v','a','d','j','x']

dirname1=r"E:\forkeywords\abstracts\\"
files=os.listdir(dirname1)
candidates=[[]]*len(files)
fm=0
thu1= thulac.thulac()
vectorizer=CountVectorizer(min_df=5,stop_words=stopwords)#定义计数器  
K=3

for fn in files:
        
    f = open(dirname1+fn)
    raw=f.read()
    f.close()

    sents=re.sub(r'[\r\t\n\u3000]','',raw)
    sents=re.split('[。？…！]',sents)

    sentences=[]
    m=0
    vocab=['']*len(raw)
    for s in sents:
        words=thu1.cut(s)
        cw=[w[0].lower() for w in words if w[0] not in stopwords and w[-1] in candpos ]
        candidates[fm]=candidates[fm]+cw
        
        words=[w[0].lower() for w in words if re.match('^\D',w[0])]
        vocab[m:m+len(words)]=words
        m=m+len(words)

        sentences.append(' '.join(words))
## candidates
    candidates[fm]=list(set( candidates[fm]))

## compute feature
    tf=vectorizer.fit_transform(sentences) #计算TF
    word=vectorizer.get_feature_names()#获取词袋模型中的所有词语  
    tfmat=tf.toarray()
    transformer=TfidfTransformer()#该类会统计每个词语的tf-idf权值

    tfidf=transformer.fit_transform(tf)#fit_transform计算tf-idf，fit_transform将文本转为词频矩阵 
    tfidfmat=tfidf.toarray()

    counts=np.sum(tfmat,axis=0)

    sent_weigth=np.zeros([ tfmat.shape[0]],dtype=int)

    for n in range( tfmat.shape[0]):
        sent_weigth[n]=sum([ counts[word.index(w)] for w in sentences[n].split(' ') if w in word])
        
    ss=[(n,sent_weigth[n]) for n in range(tfmat.shape[0])]
    ss.sort(key=lambda x:x[1],reverse=True)
    ss=[s[0]for s in ss[:K]]
   
    f = open(dirname2+fn)
    abstract=f.read()
    f.close()
    abstract=re.sub(r'[\r\t\n\u3000]','',abstract)
    print("Abstract{}".format(abstract))

    print("Auto_abstract {}".format([sents[s] for s in ss]))

    print('___________________')

