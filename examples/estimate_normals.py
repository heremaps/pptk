import points
import numpy
import time

# read points from file
X = numpy.fromfile(\
	'/Users/victorlu/Projects/PointKdTree1Split/TestData/icp/data.xyz',\
	dtype = numpy.float32).reshape((-1,3))
P = points.points(X.shape[0])
P[:] = X
del X

# estimate normal at each point
start_time = time.time()
normals = []
for (distances,indices) in P.nearest(P,64):
	nns = P[indices] - numpy.mean(P[indices],axis=0).T
	C = numpy.dot(nns.T, nns)
	W,V = numpy.linalg.eigh(C)
	normals.append(V[:, numpy.argmin(W)].T)
print 'Time to estimate normals: %f' % (time.time() - start_time)

# concatenate normals into 3 column matrix
C = numpy.concatenate(normals)
C = (C + 1) / 2
