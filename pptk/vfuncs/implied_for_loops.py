import time
import numpy as np
import foo

def f(x):
	return np.mean(x)

def gen_input(k,n):
	return [np.float32(np.random.rand(k,k)) for i in xrange(n)]

def run_test_1(x):
	t = time.time()
	r = map(f,x)
	t = time.time() - t
	return (t,r)

def run_test_2(x):
	t = time.time()
	r = foo.listofmeans(x)
	t = time.time() - t
	return (t,r)

def run_test_3(x):
	f = foo.computemean
	t = time.time()
	r = map(f,x)
	t = time.time() - t
	return (t,r)

def run_test_4(x):
	f = foo.computemean
	r = [None]*len(x)
	t = time.time()
	for i,v in enumerate(x):
		r[i] = foo.computemean(v)
	t = time.time() - t
	return (t,r)

def run_test_5(x):
	f = np.core._methods._mean
	t = time.time()
	r = map(f,x)
	t = time.time() - t
	return (t,r)

def run_test_6(x):
	t = time.time()
	r = foo.listofmeanspy(x)
	t = time.time() - t
	return (t,r)

def run_test_7(x):
	t = time.time()
	r = foo.xplusx(x)
	t = time.time() - t
	return (t,r)


