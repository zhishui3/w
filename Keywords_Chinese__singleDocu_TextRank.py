import os
import codecs
import thulac
import re
import numpy as np
from textrank4zh import TextRank4Keyword, TextRank4Sentence

f=open(r'E:\语言信息处理/stopwords-utf8.txt','r',encoding='utf-8')
raw=f.read()
f.close()
stopwords=list(set(raw.split()+['(',')','，','。','？','“','”','‘','’','：','；','【','】','…','《','■','》','_','<','>','！']))
candpos=['n','np','ns','ni','nz','v','a','d','j','x']

dirname1=r"E:\语言信息处理\2017_autumn_nlp\forkeywords\political_text\\"
dirname2=r"E:\语言信息处理\2017_autumn_nlp\forkeywords\keywords_title\\"
files=os.listdir(dirname1)
candidates=[[]]*len(files)
fm=0
tr4w = TextRank4Keyword()

for fn in files:
        
    f = open(dirname1+fn)
    raw=f.read()
    f.close()
    docu=re.sub(r'[\r\t\n\u3000]','',raw)
    
    tr4w.analyze(text=docu, lower=True, window=3)

    
    f = open(dirname2+fn)
    title_keywords=f.read()
    f.close()
    print("Title_keywords {}:{}".format(fn,title_keywords))

    data=[w['word'] for w in  tr4w.get_keywords(num=5,word_min_len = 2)]
    
    print("keywords_TextRank+loca {}".format(data))

    data= tr4w.get_keyphrases(keywords_num=10,min_occur_num = 2)

    print("keyphreases_TextRank+loca {}".format(data))

    print('___________________')




