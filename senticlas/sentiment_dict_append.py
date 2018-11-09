import os
import codecs
import re
import numpy as np
import nltk
import random
from nltk.corpus import movie_reviews
from nltk.corpus import wordnet as wd

## 读入种子情感词典

f= codecs.open(r'E:\语言信息处理\2017_autumn_nlp\corpus for classifier\sentiment\sentiment dictionary\正面情感词语（英文）.txt',mode='r')#,encoding='utf-8'
text=f.readlines()
f.close()
sentDict_pos =[]
for s in text:
    s=re.sub(r'[\r\n\u3000\ufeff\t]','',s)
    sentDict_pos.append(s)

f= codecs.open(r'E:\语言信息处理\2017_autumn_nlp\corpus for classifier\sentiment\sentiment dictionary\负面情感词语（英文）.txt',mode='r')#,encoding='utf-8'
text=f.readlines()
f.close()
sentDict_neg =[]
for s in text:
    s=re.sub(r'[\r\n\u3000\ufeff\t]','',s)
    sentDict_neg.append(s)

##读入语料库

documents = [' '.join(movie_reviews.words(fileid)) for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]

good = wd.synset('good.a.01')
bad=wd.synset('regretful.a.01')
for doc in documents:
    sentences=re.split('[.,?!;]',doc)  #切分成小句
    for sent in sentences:
        sent_tagged=nltk.pos_tag(sent.split(' '))  #词性标注
        candidates=[w[0] for w in sent_tagged if re.match('^(JJ)',w[1])  and len(w[0])>0] ## 只选择形容词做候选词
        for cand in  candidates:
            if cand in sentDict_neg or cand in sentDict_pos:
                continue
            cand_synsets=wd.synsets(cand.lower())
            cand_pos=[pos for pos in cand_synsets if '.a.01' in pos .name()]
            if len(cand_pos)>0:
                good_score=good.path_similarity(cand_pos[0])
                bad_score=bad.path_similarity(cand_pos[0])
                if good_score is None or bad_score is None:
                    continue
                if good_score>bad_score:
                    sentDict_pos.append(cand)
                else:
                    sentDict_neg.append(cand)
