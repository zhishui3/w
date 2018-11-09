import os
import codecs
import jieba
import re

from summarise import Summariser
from textrank4zh import TextRank4Keyword, TextRank4Sentence


tr4s = TextRank4Sentence()
dirname1=r"C:\forkeywords\political_text\\"
dirname2=r"C:\forkeywords\abstracts\\"

files=os.listdir(dirname1)


for fn in files:
        
    f = open(dirname1+fn)
    raw=f.read()
    f.close()
    docu=re.sub(r'[\r\t\n\u3000]','',raw)

    f = open(dirname2+fn)
    abstract=f.read()
    f.close()
    print("Abstract{}".format(abstract))
   
    tr4s.analyze(text=docu, lower=True, source = 'all_filters')
    data=[w['sentence'] for w in tr4s.get_key_sentences(num=3)]
    print("Auto-abstract{}".format(data))

    print('_________________')
