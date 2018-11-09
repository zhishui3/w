from pylab import plot,show
from numpy import array
from scipy.cluster.hierarchy import linkage, dendrogram


x=array([ [ 0.7,0.4],
                   [ 0.9,0.3],
                   [ 0.8,0.6],
                   [ 0.7,0.3],
                   [ 1.0,0.5],
                   [ 1.0,0.6]])

plot(x[:,0],x[:,1],'o')
show()

Z=linkage(x,method='single')
dendrogram(Z)
show()

