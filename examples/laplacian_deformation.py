import points
from scipy import sparse
import numpy as np

# set parameters
N = 10000	# number of points
k = 8			# neighborhood size for calculating Laplacian

# construct point cloud P
numpy.random.seed(0)
xz = points.points(np.random.rand(N,2))
y = np.multiply(\
	1-np.cos(10*np.pi*xz[:,0]),\
	1-np.cos(10*np.pi*xz[:,1]))
xz = 2*(xz-0.5)
P = np.hstack((xz[:,0],y,xz[:,1]))

# constrain points x > 0.8 to y = 1 and points x < -0.8 to y = 0
QLindices = np.flatnonzero(P[:,0]<-0.8)
QRindices = np.flatnonzero(P[:,0]>0.8)
QL = P[QLindices]
QR = P[QRindices]
QR[:,1] += 1
Qindices = hstack((QLindices,QRindices))
Q = vstack((QL,QR))
M = len(Qindices)

# compute laplacian vectors L and save neighbor indices I
L = points.points(numpoints=N,dim=3) 
I = -1*np.ones(N,k)
for i,indices in enumerate(P.nearest(P,k)):
	L[i] = P[i] - np.mean(P[indices],axis=0)
	I[i] = indices

# construct matrix A (in Ax = b)
A = sparse.lil_matrix((3*(N+M),3*N)) 
for i,inds in enumerate(I):	# first 3N rows of A
	cols = (np.tile(3*np.reshape(inds,(k,1)),(1,3))+\
		np.arange(3).reshape(3,1)).flatten()	
	d = L[i].T
	D = np.matrix([\
		[d[0],    0, d[2],-d[1],0,0,0],\
		[d[1],-d[2],    0, d[0],0,0,0],\
		[d[2], d[1],-d[0],    0,0,0,0]])
	C = np.zeros((3*k+3,7)).view(np.matrix)
	for j,ind in enumerate(inds):
		v = P[ind].T
		C[3*j:3*(j+1)] = [\
			[v[0],    0, v[2],-v[1],1,0,0],\
			[v[1],-v[2],    0, v[0],0,1,0],\
			[v[2], v[1],-v[0],    0,0,0,1]]
	Ai = D*np.inv(C.T*C)*C.T-\
		np.hstack((-np.identity(3),np.tile(np.identity(3),(1,k))/k))
	A[3*i+0,cols] = Ai[0]
	A[3*i+1,cols] = Ai[1]
	A[3*i+2,cols] = Ai[2]
for i,ind in enumerate(Qindices):	# remaining 3M rows of A 
	A[3*N+3*i+np.arange(3),3*ind+np.arange(3)] = 1

# construct vector b (in Ax = b)
b = np.vstack((np.zeros((3*N,1)),Q.reshape(3*M,1)))

# solve linear system Ax = b
Pnew = sparse.linalg.lsqr(A.tocsr(),b).reshape(N,3)
