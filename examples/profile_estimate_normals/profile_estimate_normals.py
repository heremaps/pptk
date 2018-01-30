execfile('../setvars.py')
import points
import numpy as np
import time
import sys

if __name__=='__main__':
	num_chunks = int(sys.argv[1])
	print 'num_chunks=%d'%num_chunks
	P = points.fromfile('../cropped.xyz')
	I = P.nbhds(k=64)
	S = P[I]
	X = S-points.MEAN(S,axis=0)
	W,V = points.EIGH(points.DOT(X.T,X)).unzip()
	N = V[:,points.ARGMIN(W)]
	t = time.time()
	NN = N.evaluate(num_chunks=num_chunks)
	t = time.time() - t
	print t
