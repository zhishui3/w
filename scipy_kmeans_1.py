from numpy import array
from scipy.cluster.vq import vq, kmeans, whiten
from pylab import plot,show
from numpy import vstack,array

data  = array([[ 1.9,0.8], 
                  [ 1.5,0.5],
                   [ 1.8,0.6],
                [ 1.4,0.8],
               [ 1.9,0.3],
               [ 1.8,0.7],
                   [ 2.0,0.5],
                [ 1.7,0.9],
                   [ 1.6,0.4],
                   [ 1.4,0.6],
                   [ 1.5,0.5],
                   [ 0.7,1.8],
                   [ 0.9,1.9],
                   [ 0.8,1.2],
                   [ 0.7,1.4],
                   [ 0.9,1.3],
                   [ 1.5,0.7],
                   [ 0.7,1.3],
                   [ 1.0,1.5],
                   [ 0.8,1.6],
                   [ 0.7,0.8],
                   [ 0.9,0.9],
                   [ 0.8,0.2],
                   [ 0.7,0.4],
                   [ 0.9,0.3],
                   [ 0.8,0.6],
                   [ 0.7,0.3],
                   [ 1.0,0.5],
                   [ 1.0,0.6],                   
                   [ 1.0,1.0]])

centroids,_ = kmeans(data,2)
idx,_ = vq(data,centroids)
plot(data[idx==0,0],data[idx==0,1],'ob',     data[idx==1,0],data[idx==1,1],'or')
plot(centroids[:,0],centroids[:,1],'sg',markersize=8)
show()

wd=whiten(data)
centroids,_ = kmeans(wd,2)
idx,_ = vq(wd,centroids)
#plot(data[idx==0,0],data[idx==0,1],'ob',     data[idx==1,0],data[idx==1,1],'or')
plot(wd[idx==0,0],wd[idx==0,1],'ob',     wd[idx==1,0],wd[idx==1,1],'or')
plot(centroids[:,0],centroids[:,1],'sg',markersize=8)
show()

