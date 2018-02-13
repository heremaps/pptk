import multiprocessing as mp
import time
import pickle

def _partition(n,p):
	n = int(n)
	p = int(p)
	c = n/p
	l = n%p
	R = [c*i for i in range(p-l)]
	R.extend([R[-1]+c])
	R.extend([R[-1]+(c+1)*(i+1) for i in range(l)])
	return R

def _harness(op, pipe, index):
	xs = pipe.recv()
	#t = time.time()
	result = [op(x) for x in xs]
	#print 'process %d finished in %f seconds.' % (index,time.time()-t)
	pipe.send(result)

def parmap(op, xs, numprocs=mp.cpu_count()):
	numprocs = max(1,min(min(mp.cpu_count,len(xs)),numprocs))
	splits = _partition(len(xs),numprocs)
	pipes = [mp.Pipe() for i in range(numprocs)]
	jobs = [mp.Process(target=_harness,\
		args=(op,pipes[i][1],i)) \
		for i in range(numprocs)]
	#t = time.time()
	for i,j in enumerate(jobs):
		j.start()
		#print 'started job %d' % i
	for i in range(numprocs):
		pipes[i][0].send(xs[splits[i]:splits[i+1]])
	outputs = []
	for i in range(numprocs):
		outputs.extend(pipes[i][0].recv())
		#print time.time() - t
	for i,j in enumerate(jobs):
		j.join()
		#print 'joined job %d' % i
	return outputs

