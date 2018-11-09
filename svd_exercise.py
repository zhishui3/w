import numpy as np
from scipy import linalg

a=np.array([[3, 4, 1],[2, 2, 3],[0,  1,  3],[4,  4,  1]])
u,s,v=linalg.svd(a)
s2=s
s2[2:]=0
d2=linalg.diagsvd(s2,len(u),len(v))

a2=np.dot(u,np.dot(d2,v))


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
us_skl=svd.fit_transform(a)#US前两列

vt_skl=svd.components_#V的前两行
