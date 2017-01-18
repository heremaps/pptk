import unittest
execfile('../setvars.py')
import points
import numpy

class TestPointsCreation(unittest.TestCase): 

	def test_one_point_a(self):
		P = points.points([1,2,3])
		self.assertTrue(type(P)==points.points)
		self.assertTrue(P.dtype=='float32')
		self.assertTrue(P.shape==(1,3))
		self.assertTrue(P.tolist()==[[1,2,3]])

	def test_one_point_b(self):
		P = points.points([[1,2,3]])
		self.assertTrue(type(P)==points.points)
		self.assertTrue(P.dtype=='float32')
		self.assertTrue(P.shape==(1,3))
		self.assertTrue(P.tolist()==[[1,2,3]])

	def test_two_points(self):
		P = points.points([[1,2,3],[2,3,4]])
		self.assertTrue(type(P)==points.points)
		self.assertTrue(P.dtype=='float32')
		self.assertTrue(P.shape==(2,3))
		self.assertTrue(P.tolist()==[[1,2,3],[2,3,4]])

	def test_from_string(self):
		P = points.points('1,2,3;2,3,4')
		self.assertTrue(type(P)==points.points)
		self.assertTrue(P.dtype=='float32')
		self.assertTrue(P.shape==(2,3))
		self.assertTrue(P.tolist()==[[1,2,3],[2,3,4]])

	def test_1d_point(self):
		P = points.points(1)
		self.assertTrue(type(P)==points.points)
		self.assertTrue(P.dtype=='float32')
		self.assertTrue(P.shape==(1,1))
		self.assertTrue(P.tolist()==[[1]])
	
	def test_zeros(self):
		P = points.zeros(10,3)
		self.assertTrue(type(P)==points.points)
		self.assertTrue(P.dtype=='float32')
		self.assertTrue(P.shape==(10,3))
		self.assertTrue(numpy.all(P==0))
	
	def test_rand(self):
		P = points.rand(10,3)
		self.assertTrue(type(P)==points.points)
		self.assertTrue(P.dtype=='float32')
		self.assertTrue(P.shape==(10,3))

	def test_empty(self):
		P = points.empty(10,3)
		self.assertTrue(type(P)==points.points)
		self.assertTrue(P.dtype=='float32')
		self.assertTrue(P.shape==(10,3))

	def test_view(self):
		M = numpy.matrix('1,2,3')
		with self.assertRaises(TypeError):
			M.view(points.points)
		P = points.points('1,2,3')
		Q = P.view()
		self.assertTrue(type(P)==points.points)
		self.assertTrue(P.dtype=='float32')
		self.assertTrue(P.shape==(1,3))
		self.assertTrue(P.tolist()==[[1,2,3]])

	def test_copy_from_ndarray(self):
		X = numpy.ndarray(shape=(4,5),dtype='float32')
		X[:] = 1
		with self.assertRaises(TypeError):
			points.points(X,copy=False)
		P = points.points(X)
		self.assertTrue(type(P)==points.points)
		self.assertTrue(P.dtype=='float32')
		self.assertTrue(P.shape==(4,5))
		self.assertTrue(P.tolist()==X.tolist())
		P[0,0] = 0
		self.assertTrue(X[0,0]!=0)
	
	def test_copy_from_matrix(self):
		X = numpy.matrix('1,2;2,3',dtype='float32')
		with self.assertRaises(TypeError):
			points.points(X,copy=False)
		P = points.points(X)
		self.assertTrue(type(P)==points.points)
		self.assertTrue(P.dtype=='float32')
		self.assertTrue(P.shape==(2,2))
		self.assertTrue(P.tolist()==X.tolist())
		P[0,0] = 0
		self.assertTrue(X[0,0]!=0)

class TestIndexing(unittest.TestCase):

	def setUp(self):
		self.P = points.points('1,2,3;2,3,4')
		P = self.P.copy()
		self.Q = P.view()

	def test_get_row(self):
		x = self.P[0]
		self.assertTrue(type(x)==points.points)
		self.assertTrue(x.shape==(1,3))
		self.assertTrue(x.tolist()==[[1,2,3]])

	def test_get_item(self):
		x = self.P[0:2,0]
		self.assertTrue(type(x)==points.points)
		self.assertTrue(x.shape==(2,1))
		self.assertTrue(x.tolist()==[[1],[2]])

	def test_set_row_i(self):
		self.P[0] = 1
		self.assertTrue(self.P.tolist()==[[1,1,1],[2,3,4]])
		self.Q[0] = 1
		self.assertTrue(self.Q.tolist()==[[1,1,1],[2,3,4]])

	def test_set_row_ii(self):
		self.P[0] = [0,1,2]
		self.assertTrue(self.P.tolist()==[[0,1,2],[2,3,4]])
		self.Q[0] = [0,1,2]
		self.assertTrue(self.Q.tolist()==[[0,1,2],[2,3,4]])

	def test_set_row_iii(self):
		self.P[0] = [[0,1,2]]
		self.assertTrue(self.P.tolist()==[[0,1,2],[2,3,4]])
		self.Q[0] = [[0,1,2]]
		self.assertTrue(self.Q.tolist()==[[0,1,2],[2,3,4]])

	def test_set_all_i(self):
		self.P[:] = 1
		self.assertTrue(self.P.tolist()==[[1,1,1],[1,1,1]])
		self.Q[:] = 1
		self.assertTrue(self.Q.tolist()==[[1,1,1],[1,1,1]])

	def test_set_all_ii(self):
		self.P[:] = [[0],[1]]
		self.assertTrue(self.P.tolist()==[[0,0,0],[1,1,1]])
		self.Q[:] = [[0],[1]]
		self.assertTrue(self.Q.tolist()==[[0,0,0],[1,1,1]])

	def test_set_all_iii(self):
		self.P[:] = [[4,5,6]]
		self.assertTrue(self.P.tolist()==[[4,5,6],[4,5,6]])
		self.Q[:] = [[4,5,6]]
		self.assertTrue(self.Q.tolist()==[[4,5,6],[4,5,6]])

	def test_set_item(self):
		self.P[0:2,0] = [[0],[1]]
		self.assertTrue(self.P.tolist()==[[0,2,3],[1,3,4]])
		self.Q[0:2,0] = [[0],[1]]
		self.assertTrue(self.Q.tolist()==[[0,2,3],[1,3,4]])

	def test_augmented_assignments(self):
		self.P[0] += 1
		self.assertTrue(self.P.tolist()==[[2,3,4],[2,3,4]])
		self.P[0] -= 1
		self.assertTrue(self.P.tolist()==[[1,2,3],[2,3,4]])
		self.P[0] *= 2
		self.assertTrue(self.P.tolist()==[[2,4,6],[2,3,4]])
		self.P[0] /= 2
		self.assertTrue(self.P.tolist()==[[1,2,3],[2,3,4]])
		self.P[0] += self.P[1]
		self.assertTrue(self.P.tolist()==[[3,5,7],[2,3,4]])
		self.P[0] -= self.P[1]
		self.assertTrue(self.P.tolist()==[[1,2,3],[2,3,4]])
		self.P[0] /= self.P[1]
		ans = numpy.array([[0.5,2.0/3,0.75],[2,3,4]],dtype=numpy.float32)
		self.assertTrue(self.P.tolist()==ans.tolist())

		self.Q[0] += 1
		self.assertTrue(self.Q.tolist()==[[2,3,4],[2,3,4]])
		self.Q[0] -= 1
		self.assertTrue(self.Q.tolist()==[[1,2,3],[2,3,4]])
		self.Q[0] *= 2
		self.assertTrue(self.Q.tolist()==[[2,4,6],[2,3,4]])
		self.Q[0] /= 2
		self.assertTrue(self.Q.tolist()==[[1,2,3],[2,3,4]])
		self.Q[0] += self.Q[1]
		self.assertTrue(self.Q.tolist()==[[3,5,7],[2,3,4]])
		self.Q[0] -= self.Q[1]
		self.assertTrue(self.Q.tolist()==[[1,2,3],[2,3,4]])
		self.Q[0] /= self.Q[1]
		ans = numpy.array([[0.5,2.0/3,0.75],[2,3,4]],dtype=numpy.float32)
		self.assertTrue(self.Q.tolist()==ans.tolist())
	
	def test_operations(self):
		X = self.P[0] + self.P[1]
		self.assertTrue(X.tolist()==[[3,5,7]])
		X = self.P[0] - self.P[1]
		self.assertTrue(X.tolist()==[[-1,-1,-1]])
		X = numpy.multiply(self.P[0],self.P[1])
		self.assertTrue(X.tolist()==[[2,6,12]])

		P = points.empty(2)
		P[:] = [[1,2,3],[2,3,4]]
		X = P[0] + P[1]
		self.assertTrue(X.tolist()==[[3,5,7]])
		X = P[0] - P[1]
		self.assertTrue(X.tolist()==[[-1,-1,-1]])
		X = numpy.multiply(P[0],P[1])
		self.assertTrue(X.tolist()==[[2,6,12]])

	def test_boolean_indexing(self):
		# ideally want: X = self.P[self.P[:,0]<2]
		X = self.P[numpy.asarray(self.P[:,0]<2).flatten()]
		self.assertTrue(X.tolist()==[[1,2,3]])

class TestOperations(unittest.TestCase):
	def setUp(self):
		self.P = points.points('1,2,3;2,3,4')

	def test_mul(self):
		X = self.P.T * self.P
		self.assertTrue(X.tolist()==[[5,8,11],[8,13,18],[11,18,25]])
		X = self.P * self.P.T
		self.assertTrue(X.tolist()==[[14,20],[20,29]])

class TestWriteLocking(unittest.TestCase):
	def setUp(self):
		self.P = points.points('1,2,3;2,3,4')
	
	def test_view(self):
		X = self.P.view(numpy.ndarray)
		with self.assertRaises((RuntimeError,ValueError)):
			X[:] = 1
	
	def test_asarray(self):
		X = numpy.asarray(self.P)
		with self.assertRaises((RuntimeError,ValueError)):
			X[:] = 1
	
	def test_invalid_set_value(self):
		with self.assertRaises(ValueError):
			self.P[0] = '1,2,3'
		self.assertTrue(self.P.flags.writeable==False)

class TestUpdateSystem(unittest.TestCase):

	def setUp(self):
		self.P = points.points('1,2,3;2,3,4')
		self.Q = points.points('0,0,0')
		points._last_modified.clear()

	def test_reset(self):
		points._last_modified.clear()
		self.assertTrue(points._last_modified.get(self.P._memloc)==None)
		self.assertTrue(self.P._last_updated==None)

	def test_modify_then_query(self):
		self.P[0,0]=6
		self.assertTrue(points._last_modified.get(self.P._memloc)!=None)
		self.assertTrue(self.P._last_updated==None)

		[x for x in self.P.nbhds(self.Q,k=1)]
		self.assertTrue(self.P._last_updated!=None)
		self.assertTrue(self.P._last_updated > points._last_modified.get(self.P._memloc))

	def test_directly_query(self):
		[x for x in self.P.nbhds(self.Q,k=1)]
		self.assertTrue(points._last_modified.get(self.P._memloc)!=None)
		self.assertTrue(self.P._last_updated!=None)
		self.assertTrue(self.P._last_updated > points._last_modified.get(self.P._memloc))

	def test_create_view(self):
		X = self.P[:,1:]
		self.assertTrue(X._memloc==self.P._memloc)
		self.assertTrue(X._memsize==self.P._memsize)

	def test_create_copy(self):
		X = self.P.copy()
		self.assertTrue(X._memloc!=self.P._memloc)
	
	def test_create(self):
		self.assertTrue(self.P._last_updated==None)
		self.assertTrue(self.P._memsize==24)

class TestQueries(unittest.TestCase):

	def setUp(self):
		numpy.random.seed(0)
		self.P = points.points(numpy.random.rand(100,3))
		self.Q = points.points(numpy.random.rand(5,3))

	def test_check_inputs(self):
		with self.assertRaises(ValueError):
			self.P.nbhds(points.zeros(0,2),k=1)
		with self.assertRaises(TypeError):
			self.P.nbhds(points.zeros(0,3),k='a')
		with self.assertRaises(TypeError):
			self.P.nbhds(points.zeros(0,3),r='a')

	def test_knearest(self):
		k = 10
		numqueries = self.Q.shape[0]
		I = numpy.ndarray(shape=(numqueries,k))
		for i,q in enumerate(self.Q):
			d = numpy.sum(numpy.square(self.P-q),axis=1).T
			I[i,:] = numpy.argsort(d)[0,0:k]

		R = numpy.vstack(x for x in self.P.nbhds(self.Q,k=k))
		self.assertTrue((I==R).all())

	def test_rnear(self):
		pass
	
	def test_r_k_nearest(self):
		pass

if __name__=='__main__':
	unittest.main()
