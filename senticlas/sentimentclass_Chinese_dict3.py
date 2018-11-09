import os
import codecs
import thulac
import re
import numpy as np
from sklearn.svm import SVC
import nltk
import random
from sklearn import feature_extraction  
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer  


# 读入情感词典

f= codecs.open(r'D:\FeigeDownload\lesson14\BosonNLP_sentiment_score.txt',mode='r',encoding='utf-8')
text=f.readlines()
f.close()
senDict ={}
for s in text:
    s=re.sub(r'[\r\t\n\u3000]','',s)
    s=s.split(' ')
    if len(s)==2:
        senDict[s[0]] = np.float_(s[1])

# 否定词
notlist='不、没、无、非、莫、弗、勿、毋、未、否、别、无、休、难道'
notDict=notlist.split('、')

#程度词
f= codecs.open(r'D:\FeigeDownload\lesson14\degreeAdv.txt',mode='r',encoding='utf-8')
text=f.readlines()
f.close()
degreeDict ={}
for s in text:
    s=re.sub(r'[\r\n\u3000\ufeff]','',s)
    s=re.split('\t',s)
    if len(s)==2:
        degreeDict[s[0]] = np.float_(s[1])




f=open(r'E:\语言信息处理\stopwords_utf8.txt','r',encoding='utf-8')
raw=f.read()
f.close()
punc=['(',')','，','。','？','“','”','‘','’','：','；','【','】','…','《','■','》','_','<','>','！',',',  '!', '.',  '、', '~', '）', '（', ';', ':']
sent_mark=['，','。','？','！',',',  '!', '.']
stopwords=raw.split()


# 读入文本

def score_sent(sent):
    not_num=0
    degree=1
    score=0
    for w in sent:
        if w in notDict:
            not_num+=1
        if w in degreeDict:
            degree=degree*degreeDict[w]
        if w in senDict:
            score=score+senDict[w]
    score=(-1)**not_num*degree*score
    return score

f=open(r'D:\FeigeDownload\review_sentiment\train2.rlabelclass','r')
raw=f.read()
f.close()
file_names=raw.split('\n')

file_names=[(f.split(' ')[0], f.split(' ')[1] )for f in file_names if len(f)>0]
random.shuffle(file_names)
file_sent=[s for (f,s) in file_names] #文本的情感标记
files=[f for (f,s) in file_names]   #文件名

thu1= thulac.thulac()
dirname=r"D:\FeigeDownload\sentiment\review_sentiment\train2\\"
sent_doc=[] #句子的情感分值
target=[]   # 句子的人工情感标记
fn=0
fm=0 #正确读入文件数量

for fn in range(len(files)):
    try:
        f= open(dirname+files[fn],mode='r')#,encoding="utf-8")
        raw=f.read()
        f.close()
        raw=re.sub(r'[\r\n\u3000]','',raw)
        sents=re.split('[，。？！,!.\t]',raw) #文本切分为句子，依据结束标记和符号
        sents=[s for s in sents if len(s.strip())>0]

        sents_score=[]
        
        for s in sents:
            words=thu1.cut(s)
            words=[w[0] for w in words]
            words=[w for w in words if (w in senDict or w in notDict or w in degreeDict ) and w not in stopwords ]
            sents_score.append(score_sent(words)) #计算每个句子的情感分值

        #sents_score[-1]=sents_score[-1]*len(sents_score)/2
        sent_doc.append(sum(sents_score))   #每个句子的情感分值的和，作为文本的情感分值
        #sent_doc.append(sents_score[-1])
        
        target.append(file_sent[fn])
        fm+=1
        if fm>2000:
            break
    except:
        continue
sent_doc=np.sign(np.array(sent_doc))

target=[ int(t) for t in target]
print(sum(sent_doc==target)/len(target))

