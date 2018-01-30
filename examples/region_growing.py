import points
import numpy as np
import networkx as nx
import expr

def estimate_normals(P,k,verbose=False):
	N = np.zeros(P.shape).view(np.matrix)
	for i,indices in enumerate(P.nbhds(k=k,verbose=verbose)):
		X = P[indices]-np.mean(P[indices],axis=0)
		W,V = np.linalg.eigh(X.T*X)
		N[i] = V[:,np.argmin(W)].T
	return N

def estimate_normals_batched(P,k,verbose=False):
	S = P[P.nbhds(k=k,verbose=verbose)]
	X = S-points.MEAN(S,axis=0)
	W,V = points.EIGH(points.DOT(X.T,X)).unzip()
	return V[:,points.ARGMIN(W)].evaluate()

def estimate_normals_and_curvatures(P,k,verbose=False):
	N = np.zeros(P.shape).view(np.matrix)
	curvature = np.zeros(P.shape[0])
	for i,indices in enumerate(P.nhbrhoods(k=k,verbose=verbose)):
		X = P[indices]-np.mean(P[indices],axis=0)
		W,V = np.linalg.eigh(X.T*X)
		mindex = np.argmin(W)
		N[i] = V[:,mindex].T
		curvature[i] = W[mindex]/np.sum(W)
	return N,curvature

def orient_normals(P,N,k=16,verbose=False):
	# todo: on bunny, gives small isolated patches of incorrectly
	# oriented normals.
	N = N.copy()
	J = [j for indices in P.nhbrhoods(k=k,verbose=verbose) \
		for j in indices]
	W = numpy.asarray(1.0-np.abs(np.sum(np.multiply(
		N[J],np.repeat(N,k,axis=0)),axis=1)).T).flatten()
	I = [i for j in range(P.shape[0]) for i in [j]*k]
	E = zip(I,J,W)
	G = nx.Graph()
	G.add_weighted_edges_from(E)
	T = nx.minimum_spanning_tree(G)
	for S in nx.connected_component_subgraphs(T):
		for (i,j) in nx.dfs_edges(S):
			if (N[i]*N[j].T)[0,0] < 0.0:
				N[j] = -N[j]
	return N

def grow_regions(P,N,c,eps1,eps2,verbose=False):
	numPoints = P.shape[0]
	labels = numpy.zeros(numPoints)
	seedable = c<eps2
	nhbrhoods = [inds for inds in P.nhbrhoods(k=64,verbose=verbose)]
	for i in range(numPoints):
		if labels[i]>0:
			continue
		labels[i] = i+1
		seeds = [i]
		while seeds:
			s = seeds.pop(0)
			nhbr_indices = nhbrhoods[s][labels[nhbrhoods[s]]==0]
			d = np.abs(np.asarray((P[nhbr_indices]-P[s])*N[s].T).flatten())
			nhbr_indices = nhbr_indices[d<eps1]
			labels[nhbr_indices]=i+1
			seeds.extend(\
				nhbr_indices[seedable[nhbr_indices]].tolist())
	newlabels = {l:i for i,l in enumerate(numpy.unique(labels))}
	labels = np.array([newlabels[l] for l in labels])
	return labels
