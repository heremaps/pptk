import numpy
import time
import ctypes
import multiprocessing
from ..kdtree import kdtree
import parmap
import expr

from expr import MEAN
from expr import SUM 
from expr import PROD 
from expr import ALL
from expr import ANY
from expr import MIN
from expr import MAX
from expr import ARGMIN
from expr import ARGMAX
from expr import EIGH
from expr import DOT
from expr import TRANSPOSE
__all__ = ['points','expr','MEAN','SUM','PROD','ALL','ANY','MIN','MAX','ARGMIN','ARGMAX','EIGH','DOT','TRANSPOSE']
_last_modified = dict()

def _memaddr(obj):
	return ctypes.addressof(\
		obj.ctypes.data_as(ctypes.POINTER(ctypes.c_float)).contents)

def zeros(numPoints,dim=3,dtype=numpy.float32):
	obj = numpy.ndarray.__new__(points,(numPoints,dim),dtype=dtype)
	obj[:] = 0
	obj.flags.writeable = False
	return obj

def ones(numPoints,dim=3,dtype=numpy.float32):
	obj = numpy.ndarray.__new__(points,(numPoints,dim),dtype=dtype)
	obj[:] = 1
	obj.flags.writeable = False;
	return obj

def empty(numPoints,dim=3,dtype=numpy.float32):
	obj = numpy.ndarray.__new__(points,(numPoints,dim),dtype=dtype)
	obj.flags.writeable = False
	return obj

def rand(numPoints,dim=3,dtype=numpy.float32):
	return dtype(points(numpy.random.rand(numPoints,dim)))

def fromfile(filename,dtype=numpy.float32):
	x = numpy.fromfile(filename,dtype)
	if x.shape[0] % 3 != 0:
		raise ValueError('Invalid points file: %s' % filename)
	p = empty(x.shape[0]/3,3)
	p[:] = x.reshape((x.shape[0]/3,3))
	return p

class points (numpy.matrix):
	def __new__(cls,data,dtype=numpy.float32):
		if not isinstance(data,numpy.ndarray):
			if isinstance(data,str):
				data = \
					numpy.matrixlib.defmatrix._convert_from_string(data)
			data = numpy.array(data,dtype=dtype)
		if not dtype:
			dtype = data.dtype
		obj = numpy.ndarray.__new__(\
			cls,data.shape,dtype=dtype)
		obj[:] = data
		return obj

	def __array_finalize__(self,obj):
		self._last_updated = None
		self._tree = None

		if obj is not None and type(obj) != points:
			# arrived at here via view() of a non-points object
			raise TypeError('Detected attempt at creating points-type'\
				'view on non-points object.')
			
		if obj is None:
			# arrived at here via __new__
			self._memsize = self.size * self.dtype.itemsize
			self._memloc = _memaddr(self)
		elif _memaddr(self) < obj._memloc or \
				_memaddr(self) >= obj._memloc + obj._memsize:
			# arrived at here via copy()
			self._memsize = self.size * self.dtype.itemsize
			self._memloc = _memaddr(self)
		else:
			# arrived at here via slicing/indexing
			# or view() of a points object
			self._memsize = obj._memsize
			self._memloc = obj._memloc

		# note: matrix.__array_finalize__ sets the shape attribute of
		# self, which in turn calls points.__array_finalize__.
		# call matrix.__array_finalize__ here at the end to ensure
		# self's attributes have been fully specified.
		numpy.matrix.__array_finalize__(self,obj)

	def __init__(self,*args,**kwargs):
		self.flags.writeable = False

	def _record_modify_time(self):
		_last_modified[self._memloc] = time.time()

	def _update_kd_tree(self):
		# if there is no recorded last modify time for self._memloc,
		# then self has either not been modified yet since creation,
		# or _last_modified dictionary has been cleared. Either way,
		# the k-d tree needs updating; we set the last modify time to
		# the current time to trigger this.
		if _last_modified.get(self._memloc) is None:
			_last_modified[self._memloc] = time.time()

		# note: None < x, for any number x
		build_time = None
		if self._last_updated < _last_modified[self._memloc]:
			#print 'updating k-d tree'
			# note: do not need to explicitly call __del__()
			# as it is automatically called when overwritten
			build_time = time.time()
			self._tree = kdtree._build(self)
			build_time = time.time() - build_time
			self._last_updated = time.time()
		return build_time

	def nbhds(self,queries=None,k=1,r=None,verbose=False):
		return expr.nbhds_op(self,queries,k,r)

	# override methods that modify object content to
	# record timestamp, signalling need for k-d tree update

	# inplace arithmetic methods
	# e.g. +=, -=, *=, /=, //=, %=, **=, <<=, >>=, &=, ^=, |=
	def __iadd__(self,other):
		self._record_modify_time();
		if self.base is not None:
			self.base.flags.writeable = True
		self.flags.writeable = True
		obj = numpy.matrix.__iadd__(self,other) # why need to return?
		self.flags.writeable = False
		if self.base is not None:
			self.base.flags.writeable = False
		return obj
	def __isub__(self,other):
		self._record_modify_time();
		if self.base is not None:
			self.base.flags.writeable = True
		self.flags.writeable = True
		obj = numpy.matrix.__isub__(self,other)
		self.flags.writeable = False
		if self.base is not None:
			self.base.flags.writeable = False
		return obj
	def __imul__(self,other):
		self._record_modify_time();
		if self.base is not None:
			self.base.flags.writeable = True
		self.flags.writeable = True
		obj = numpy.matrix.__imul__(self,other)
		self.flags.writeable = False
		if self.base is not None:
			self.base.flags.writeable = False
		return obj
	def __idiv__(self,other):
		self._record_modify_time();
		if self.base is not None:
			self.base.flags.writeable = True
		self.flags.writeable = True
		obj = numpy.matrix.__idiv__(self,other)
		self.flags.writeable = False
		if self.base is not None:
			self.base.flags.writeable = False
		return obj
	def __itruediv__(self,other):
		self._record_modify_time();
		if self.base is not None:
			self.base.flags.writeable = True
		self.flags.writeable = True
		obj = numpy.matrix.__itruediv__(self,other)
		self.flags.writeable = False
		if self.base is not None:
			self.base.flags.writeable = False
		return obj
	def __ifloordiv__(self,other):
		self._record_modify_time();
		if self.base is not None:
			self.base.flags.writeable = True
		self.flags.writeable = True
		obj = numpy.matrix.__ifloordiv__(self,other)
		self.flags.writeable = False
		if self.base is not None:
			self.base.flags.writeable = False
		return obj
	def __imod__(self,other):
		self._record_modify_time();
		if self.base is not None:
			self.base.flags.writeable = True
		self.flags.writeable = True
		obj = numpy.matrix.__imod__(self,other)
		self.flags.writeable = False
		if self.base is not None:
			self.base.flags.writeable = False
		return obj
	def __ipow__(self,other):
		self._record_modify_time();
		if self.base is not None:
			self.base.flags.writeable = True
		self.flags.writeable = True
		obj = numpy.matrix.__ipow__(self,other)
		self.flags.writeable = False
		if self.base is not None:
			self.base.flags.writeable = False
		return obj
	def __ilshift__(self,other):
		self._record_modify_time();
		if self.base is not None:
			self.base.flags.writeable = True
		self.flags.writeable = True
		obj = numpy.matrix.__ilshift__(self,other)
		self.flags.writeable = False
		if self.base is not None:
			self.base.flags.writeable = False
		return obj
	def __irshift__(self,other):
		self._record_modify_time();
		if self.base is not None:
			self.base.flags.writeable = True
		self.flags.writeable = True
		obj = numpy.matrix.__irshift__(self,other)
		self.flags.writeable = False
		if self.base is not None:
			self.base.flags.writeable = False
		return obj
	def __iand__(self,other):
		self._record_modify_time();
		if self.base is not None:
			self.base.flags.writeable = True
		self.flags.writeable = True
		obj = numpy.matrix.__iand__(self,other)
		self.flags.writeable = False
		if self.base is not None:
			self.base.flags.writeable = False
		return obj
	def __ixor__(self,other):
		self._record_modify_time();
		if self.base is not None:
			self.base.flags.writeable = True
		self.flags.writeable = True
		obj = numpy.matrix.__ixor__(self,other)
		self.flags.writeable = False
		if self.base is not None:
			self.base.flags.writeable = False
		return obj
	def __ior__(self,other):
		self._record_modify_time();
		if self.base is not None:
			self.base.flags.writeable = True
		self.flags.writeable = True
		obj = numpy.matrix.__ior__(self,other)
		self.flags.writeable = False
		if self.base is not None:
			self.base.flags.writeable = False
		return obj
	
	# indexing and slicing operator
	def __getitem__(self,key):
		if isinstance(key,expr.expression):
			return expr.index_op(\
				expr._make_expression(self),key,slice(None,None,None))
		elif isinstance(key,tuple) and len(key)==2 and (\
				isinstance(key[0],expr.expression) and \
				isinstance(key[1],slice) \
			or \
				isinstance(key[0],slice) and \
				isinstance(key[1],expr.expression)):
			return expr.index_op(\
				expr._make_expression(self),key[0],key[1])
		else:
			return numpy.matrix.__getitem__(self,key)
	def __setitem__(self,key,value):
		if self.base is not None:
			self.base.flags.writeable = True
		self.flags.writeable = True
		try:
			numpy.matrix.__setitem__(self,key,value)
		finally:
			self.flags.writeable = False
			if self.base is not None:
				self.base.flags.writeable = False
		self._record_modify_time();
	def __delitem__(self,key):
		self.flags.writeable = True
		numpy.matrix.__delitem__(self,key)
		self.flags.writeable = False
		self._record_modify_time();
	def __getslice__(self,i,j):
		return numpy.matrix.__getslice__(self,i,j)
	def __setslice__(self,i,j,sequence):
		if self.base is not None:
			self.base.flags.writeable = True
		self.flags.writeable = True
		try:
			numpy.matrix.__setslice__(self,i,j,sequence)
		finally:
			self.flags.writeable = False
			if self.base is not None:
				self.base.flags.writeable = False
		self._record_modify_time();
	def __delslice__(self,i,j):
		self.flags.writeable = True
		numpy.matrix.__delslice__(self,i,j)
		self.flags.writeable = False
		self._record_modify_time();
	

