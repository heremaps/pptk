import points
import numpy as np

def rotation_matrix(a,b,c):
	# return matrix that first rotates by 
	# radian angles a,b,c along axes x,y,z
	Rx = np.array([\
		[1,0,0],\
		[0,np.cos(a),-np.sin(a)],\
		[0,np.sin(a),np.cos(a)]])
	Ry = np.array([\
		[np.cos(b),0,np.sin(b)],\
		[0,1,0],\
		[-np.sin(b),0,np.cos(b)]])
	Rz = np.array([\
		[np.cos(c),-np.sin(c),0],\
		[np.sin(c),np.cos(c),0],\
		[0,0,1]])
	return np.float32(np.dot(Rz,np.dot(Ry,Rx)))

def icp(X,Y,N,max_iters=50,err_tol=0.001,verbose=False):
	# find the R and t that aligns X into Y, N
	R = np.float32(np.eye(3))
	t = np.float32(np.zeros(3))
	Q = X.copy()
	for i in xrange(max_iters):
		# return 1-nn in V
		V = np.vstack(Y[Y.nbhds(Q,k=1)].evaluate())
		# set up linear system Ax=B
		A = np.hstack((np.cross(V,N),N))
		B = np.sum(np.multiply(V-Q,N),axis=1)
		err = np.mean(np.square(B))
		if verbose:
			print '%d: err=%e'%(i,err)
		if err < err_tol:
			break
		[[a],[b],[c],[tx],[ty],[tz]] = \
			np.linalg.lstsq(A,B)[0].tolist()
		if verbose:
			print '    (a,b,c)=(%f,%f,%f), t=(%f,%f,%f)'%\
				(a,b,c,tx,ty,tz)
		# update R and t
		M = rotation_matrix(a,b,c)
		R = np.dot(M,R)
		t = np.dot(M,t)+\
			np.float32(np.array([tx,ty,tz]))
		# update Q
		Q = np.dot(X,R.T)+t
	return (R,t)
