import nltk
from nltk.corpus import movie_reviews

import numpy as np

documents = [' '.join(movie_reviews.words(fileid)) for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]  #转化为词列表的影评，与标签，组成二元组

from sklearn import feature_extraction  
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer  

vectorizer=CountVectorizer(min_df=20,stop_words='english')#该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频  
transformer=TfidfTransformer()#该类会统计每个词语的tf-idf权值
tf=vectorizer.fit_transform(documents)
tfidf=transformer.fit_transform(tf)#fit_transform计算tf-idf，fit_transform将文本转为词频矩阵  
vocab=vectorizer.get_feature_names()

import lda

model = lda.LDA(n_topics=4, n_iter=150, random_state=1)
model.fit(tf)

topic_words=model.topic_word_

# 显示每个话题的前10词
n = 10
for i, t in enumerate(topic_words):
    topicwords = np.array(vocab)[np.argsort(-t)][:n]
    print('*Topic {}\n- {}'.format(i, ' '.join(topicwords)))


# 显示前10个文本的前K个话题
k=4
doc_topic = model.doc_topic_
for i in range(10):
    topic_most_pr = doc_topic[i].argsort()
    print("doc: {} topic: {}".format(i, topic_most_pr[:k]))

