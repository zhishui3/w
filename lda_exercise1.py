import numpy as np
from scipy import linalg
import lda


a=np.array([[1, 1, 1, 0, 0], [
3, 3, 3, 0, 0], [
4, 4, 4, 0, 0], [
5, 5, 5, 0, 0], [
0, 2, 0, 4, 4], [
0, 0, 0, 5, 5], [
0, 1, 0, 2, 2]])

u,s,v=linalg.svd(a)
s2=s
s2[2:]=0
d2=linalg.diagsvd(s2,len(u),len(v))

a2=np.dot(u,np.dot(d2,v))

us_sp=np.dot(u,d2)
vs=np.dot(d2,v)
vs_sp=vs.transpose()

from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=2)
svd.fit(a)  
us_skl=svd.fit_transform(a)

model = lda.LDA(n_topics=2, n_iter=150, random_state=1)
model.fit(a)

