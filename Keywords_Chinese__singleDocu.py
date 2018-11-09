import os
import codecs
import thulac
import re
from sklearn import feature_extraction  
from sklearn.feature_extraction.text import CountVectorizer  
import numpy as np
from scipy import spatial
from scipy.cluster.vq import kmeans,vq,whiten

f=open(r'C:\Documents and Settings\Administrator\桌面\2017_nlp_autumn\forkeywords/stopwordsCH.txt','r',encoding='utf-8')
raw=f.read()
f.close()
stopwords=list(set(raw.split()+['(',')','，','。','？','“','”','‘','’','：','；','【','】','…','《','■','》','_','<','>','！']))
candpos=['n','np','ns','ni','nz','v','a','d','j','x']

dirname1=r"C:\Documents and Settings\Administrator\桌面\2018_spring\forkeywords\political_text\\"
dirname2=r"C:\Documents and Settings\Administrator\桌面\2018_spring\forkeywords\keywords_title\\"
files=os.listdir(dirname1)
candidates=[[]]*len(files)
fm=0
thu1= thulac.thulac()
vectorizer=CountVectorizer(min_df=5,stop_words=stopwords)#定义计数器  


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
    
    f = open(dirname2+fn)
    title_keywords=f.read()
    f.close()
    print("Title_keywords {}:{}".format(fn,title_keywords))
## determine keywords based on term frequency 
    counts=np.sum(tfmat,axis=0)
    data=[(w,counts[word.index(w)]) for w in candidates[fm] if w in word]
    data.sort(key=lambda x:x[1],reverse=True)   
    print("keywords_TF {}".format([w for (w,t) in data[:5]]))

# determine keywords based on term frequency distribution

    std_tf=np.std(tfmat,axis=0)
    data=[(w,std_tf[word.index(w)]) for w in candidates[fm] if w in word]
    data.sort(key=lambda x:x[1],reverse=True)   
    print("keywords_TFdistr {}".format([w for (w,t) in data[:5]]))

# determine keywords based on term frequency + location

    local=[vocab.index(w) for w in vocab]
    local=np.array(local)/len(vocab)
    
    data=[(w,std_tf[word.index(w)]-local[word.index(w)]) for w in candidates[fm] if w in word]
    data.sort(key=lambda x:x[1],reverse=True)   
    print("keywords_TF+loca {}".format([w for (w,t) in data[:5]]))

    print('___________________')




