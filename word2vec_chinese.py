from gensim.models import word2vec
import jieba
import re 
import os
import numpy as np


dirname=r"C:\Documents and Settings\Administrator\桌面\2017_nlp_autumn\Sogou\C000008\\"
files=os.listdir(dirname)
sentences=['']*1000000
m=0

for fn in files:
    try:
        f= open(dirname+fn,mode='r')#,encoding="utf-8")
        raw=f.read()
        f.close()
        raw=re.sub(r'[\r\n\u3000]','',raw)
        sents=re.split('[。？…！]',raw) #文本切分为句子，依据结束标记和符号        
        sents=[s.strip() for s in sents if len(s.strip())>0]
        for s in sents:
            words=list(jieba.cut(s))
            words=[w.strip() for w in words if len(w.strip())>0]
            sentences[m]=words
            m+=1
    except:
        continue
sentences[m:]=[]

model=word2vec.Word2Vec(sentences,  min_count=5,size=50)            

# 提取词向量
model['已经']

# 计算相似度,Compute cosine similarity between two docvecs
d=model.similarity(u"好", u"行")
print(d)

# 计算最相似的10个词及其相似度，This method computes cosine similarity between a simple mean
## of the projection weight vectors of the given words and the vectors for each word in the model.

model.most_similar(u"好")

model.most_similar(positive=['进入','之后'],negative=['正在'])


#一个简单的适合短句的句子相似度

sent_vec=np.zeros([len(sentences),model.layer1_size])

for n in range(len(sentences)):
    sv=np.zeros([1,model.layer1_size])
    for w in sentences[n]:
        if w in model:
            sv=sv+model[w] #累加句子中每个词的向量，作为整句的向量
    sent_vec[n,:]=sv

from scipy import spatial

def sent_most_simlar(sent_id):
    print(sentences[sent_id])
    sent_most_siml_id=-1
    cos_simla=0
    for n in range(len(sentences)):
        if n == sent_id:
            continue
        cs=1-spatial.distance.cosine(sent_vec[n,:],sent_vec[sent_id,:]) # 计算两个句子的余弦相似度
        if cs>=cos_simla:
            cos_simla=cs
            sent_most_siml_id=n
    print(sent_most_siml_id, cos_simla,sentences[sent_most_siml_id])
    
