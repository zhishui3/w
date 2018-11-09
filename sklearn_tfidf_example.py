# coding:utf-8  
 
from sklearn import feature_extraction  
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer  
  

corpus=["他 来自 北京 清华大学",#文本0，词之间以空格隔开
        "他 来自 网易 杭研 大厦",#文本1  
        "他 硕士 毕业 于 中国 科学院",#文本2  
        "我 毕业 于 中国 传媒大学"]#文本3  
vectorizer=CountVectorizer(min_df=1, token_pattern='(?u)\\b\\w+\\b')#该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频  
transformer=TfidfTransformer()#该类会统计每个词语的tf-idf权值
tf=vectorizer.fit_transform(corpus) #计算词频,稀疏矩阵
#print('tf_matrix:',tf.todense())
tfidf=transformer.fit_transform(tf)#fit_transform计算tf-idf，稀疏矩阵
#print('tfidf_matrix:',tfidf.todense())

word=vectorizer.get_feature_names()#获取词袋模型中的所有词语  
weight=tfidf.toarray()#将tf-idf转化为稠密矩阵，元素a[i][j]表示j词在i类文本中的tf-idf权重  
from collections import OrderedDict as od
#dd = od()
dd = dict()
for i in range(len(weight)):
    print (u"-------文本",i,u"中的词语的tf-idf权重------"  )
    for j in range(len(word)):  
        print (word[j],weight[i][j] )
        dd[weight[i][j]]=word[j]
#print(word)
#a = [dd[v] for v in sorted(dd.keys())]
#sorted(dd.keys())
result = sorted(dd.items(), key=lambda d:d[1], reverse = True)
result = result[:7]
print(result)