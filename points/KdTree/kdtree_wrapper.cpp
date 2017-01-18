#include <iostream>
#include <limits>
#include "Python.h"
#include "arrayobject.h"
#include "PointKdTree.h"
#include "BuilderParallel.h"
#include "TraverserBatchedDF.h"
#include "TraverserDF.h"
#include "python_util.h"
#include "../KdTree/UsefulHeaders/vltools/timer.h"
using namespace vltools;
using namespace std;
using namespace pointkd;

struct ArrayStruct {
	const void * data;
	float scalar;	
	npy_intp m;
	npy_intp n;
	npy_intp row_stride;
	npy_intp col_stride;
	int type_num;
};

bool extract_array (ArrayStruct & item, PyObject * obj) {
	// Checks the following conditions:
	// 1. obj is a PyArrayObject
	// 2. obj's ndim is either 1 or 2
	// If all the above conditions hold, then array properties are
	// extracted and function returns true.
	if (!PyArray_Check(obj)) {
		PyErr_SetString(PyExc_TypeError, "Encountered non-array type");
		return false;
	}
	PyArrayObject * arr = (PyArrayObject*)obj;
	int ndim = PyArray_NDIM(arr);
	if (ndim != 1 && ndim != 2) {
		PyErr_SetString(PyExc_ValueError, "Array ndim neither 1 nor 2");
		return false;
	}
	if (ndim == 1) {
		item.m = 1;
		item.n = PyArray_DIM(arr, 0);
		item.row_stride = 0;
		item.col_stride = PyArray_STRIDE(arr, 0) / sizeof(float);
	} else {
		item.m = PyArray_DIM(arr, 0);
		item.n = PyArray_DIM(arr, 1);
		item.row_stride = PyArray_STRIDE(arr, 0) / sizeof(float);
		item.col_stride = PyArray_STRIDE(arr, 1) / sizeof(float);
	}
	if (item.m == 1)
		item.row_stride = 0;
	if (item.n == 1)
		item.col_stride = 0;
	item.data = PyArray_DATA(arr);
	item.type_num = PyArray_TYPE(arr);
	return true;
}

template <typename T>
void copyToVector(vector<T> & v, const ArrayStruct arr) {
	// assumes arr.type_num is consistent with T
	v.reserve(arr.m * arr.n);
	for (npy_intp i = 0; i < arr.m; i++)
		for (npy_intp j = 0; j < arr.n; j++)
			v.push_back(
				*((const T *)arr.data + 
				arr.row_stride * i + 
				arr.col_stride * j));
}

PyObject * listOfLists(const vector<int> & indices, 
	const vector<float> & distances, const int k) {
	// assumes indices.size() == distances.size()
	// and k divides indices.size()
	Py_ssize_t num_lists = (Py_ssize_t)indices.size() / k;
	PyObject * indices_list = PyList_New(num_lists);
	PyObject * distances_list = PyList_New(num_lists);
	for (Py_ssize_t i = 0; i < num_lists; i++) {
		const int * ptr_indices = &indices[i * k];
		const float * ptr_distances = &distances[i * k];
		Py_ssize_t num_neighbors = (Py_ssize_t)k;
		for (int j = 0; j < k ; j++) {
			if (ptr_indices[j] == -1) {
				num_neighbors = j;
				break;
			}
		}
		PyObject * indices_sublist = PyList_New(num_neighbors);
		PyObject * distances_sublist = PyList_New(num_neighbors);
		for (Py_ssize_t j = 0; j < num_neighbors; j++) {
			PyList_SetItem(indices_sublist, j, 
				PyInt_FromLong((long)ptr_indices[j]));
			PyList_SetItem(distances_sublist, j, 
				PyFloat_FromDouble((double)ptr_distances[j]));
		}
		PyList_SetItem(indices_list, i, indices_sublist);
		PyList_SetItem(distances_list, i, distances_sublist);
	}
	PyObject * out_tuple = PyTuple_New(2);
	PyTuple_SetItem(out_tuple, 0, indices_list);
	PyTuple_SetItem(out_tuple, 1, distances_list);

	return out_tuple;
}

PyObject * listOfArrays(const vector<int> & indices, 
	const vector<float> & distances, const int k) {
	Py_ssize_t num_arrays = (Py_ssize_t)indices.size() / k;
	PyObject * indices_list = PyList_New(num_arrays);
	PyObject * distances_list = PyList_New(num_arrays);
	for (Py_ssize_t i = 0; i < num_arrays; i++) {
		const int * ptr_indices = &indices[i * k];
		const float * ptr_distances = &distances[i * k];
		npy_intp num_neighbors = (npy_intp)k;
		for (int j = 0; j < k ; j++) {
			if (ptr_indices[j] == -1) {
				num_neighbors = j;
				break;
			}
		}
		PyObject * indices_array = 
			PyArray_EMPTY(1, &num_neighbors, NPY_INT32, false);
		PyObject * distances_array = 
			PyArray_EMPTY(1, &num_neighbors, NPY_FLOAT32, false);
		copy(ptr_indices, ptr_indices + num_neighbors, 
			(int*)PyArray_DATA((PyArrayObject*)indices_array));
		copy(ptr_distances, ptr_distances + num_neighbors, 
			(float*)PyArray_DATA((PyArrayObject*)distances_array));
		PyList_SetItem(indices_list, i, indices_array);
		PyList_SetItem(distances_list, i, distances_array);
	}
	PyObject * out_tuple = PyTuple_New(2);
	PyTuple_SetItem(out_tuple, 0, indices_list);
	PyTuple_SetItem(out_tuple, 1, distances_list);

	return out_tuple;
}

PyObject * k_nearest (PyObject * obj_queries, 
	const PointKdTree<float> * tree, int k, float dMax) {
	// assumes k > 0 and dMax > 0
	vector<int> indices;
	vector<float> distances;
	if (obj_queries == Py_None) {
		// all k-nearest
		TraverserBatchedDF<float> trav(*tree);
		trav.allNearest(k, indices, distances, dMax);
	} else if (PyArray_Check(obj_queries)) {
		ArrayStruct arrStruct;
		if (!extract_array(arrStruct, obj_queries)) {
			return NULL;
		}
		PyArrayObject * arr = (PyArrayObject *)obj_queries;
		if (arrStruct.type_num == NPY_FLOAT32) {
			// query with explicitly specified query points
			if (arrStruct.n != 3) {
				PyErr_SetString(PyExc_ValueError, 
					"Require 3-d query points");
				return NULL;
			}
			vector<float> queries;
			copyToVector(queries, arrStruct); 
			TraverserDF<float> trav(*tree);
			trav.nearest(queries, k, indices, distances, dMax);
		} else if (arrStruct.type_num == NPY_INT32) {
			// query with array of indices
			vector<int> queryIndices;
			copyToVector(queryIndices, arrStruct);
			if (!check_indices(queryIndices, (npy_intp)tree->getNumPoints(), 0)) {
				PyErr_SetString(PyExc_IndexError, "Index out of bounds");
				return NULL;
			}
			fix_negative_indices(queryIndices, (npy_intp)tree->getNumPoints());
			TraverserDF<float> trav(*tree);
			trav.nearestSelf(queryIndices, k, indices, distances, dMax);
		} else {
			PyErr_Format(PyExc_TypeError, 
				"Querying with array dtype %s unsupported",
				PyArray_DESCR(arr)->typeobj->tp_name);
			return NULL;
		}
	} else if (PyList_Check(obj_queries)) {
		// query with list of indices
		vector<int> queryIndices;
		if (!extract_indices(queryIndices, obj_queries, 0)) {
			PyErr_SetString(PyExc_IndexError, "Invalid list of indices");
			return NULL;
		}
		if (!check_indices(queryIndices, (npy_intp)tree->getNumPoints(), 0)) {
			PyErr_SetString(PyExc_IndexError, "Index out of bounds");
			return NULL;
		}
		fix_negative_indices(queryIndices, (npy_intp)tree->getNumPoints());
		TraverserDF<float> trav(*tree);
		trav.nearestSelf(queryIndices, k, indices, distances, dMax);
	} else if (PySlice_Check(obj_queries)) {
		PyErr_SetString(PyExc_NotImplementedError, 
			"Slice-based query not yet implemented");
		return NULL;
	} else {
		PyErr_Format(PyExc_TypeError, 
			"Unsupported query type %s", 
			obj_queries->ob_type->tp_name);
		return NULL;
	}

	// store results in tuple (indices, distances)
	// note: for 4-nn on 330309 points, 
	// generating a list of lists takes about 0.38+0.59
	// generating a list of arrays takes about 0.38+0.20
	PyObject * result = listOfArrays(indices, distances, k);
	//PyObject * result = listOfLists(indices, distances, k);

	return result;
}

void delete_kdtree (PyObject * obj) {
	delete (PointKdTree<float>*)PyCapsule_GetPointer(obj, NULL);
}

static PyObject * _build (PyObject * self, PyObject * args, PyObject * kwargs) {
	static char * keywords[] = {(char*)"data", (char*)"numprocs", 
		(char*)"maxleafsize", (char*)"emptysplit", NULL};
	PyObject * data = NULL;
	int numProcs = -1;
	int maxLeafSize = 16;
	float emptySplit = 0.2f;
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|ii", keywords,
		&data, &numProcs, &maxLeafSize, &emptySplit)) {
		PyErr_SetString(PyExc_TypeError, "Failed parsing arguments");
		return NULL;
	}
	ArrayStruct X;
	if (!extract_array(X, data)) {
		return NULL;
	}
	if (X.type_num != NPY_FLOAT32) {
		PyErr_SetString(PyExc_TypeError, "Point dtype must be float32");
		return NULL;
	}
	if (X.n != 3) {
		PyErr_SetString(PyExc_ValueError, "Currently only supports 3-d points");
		return NULL;
	}
	// if not contiguous, copy array to stl vector
	const float * points;
	vector<float> buf;
	if (!PyArray_ISCONTIGUOUS(data)) {
		buf.reserve(X.m * X.n);
		for (npy_intp i = 0; i < X.m; i++)
			for (npy_intp j = 0; j < X.n; j++)
				buf.push_back(*((const float *)X.data + X.row_stride * i + X.col_stride * j));
		points = &buf[0];
	} else
		points = (const float *)X.data;
	// build k-d tree
	PointKdTree<float> * tree = new PointKdTree<float>();
	BuilderParallel<float> builder(*tree);
	builder.build(points, X.m, X.n, maxLeafSize, emptySplit, numProcs);
	// construct capsule object
	PyObject * capsule = PyCapsule_New((void*)tree, NULL, &delete_kdtree);
	return capsule;
}

static PyObject * _query(
	PyObject * self, PyObject * args, PyObject * kwargs)
{
	const float flt_infinity = numeric_limits<float>::infinity();
	PyObject * obj_kdtree;
	PyObject * obj_queries;
	PyObject * obj_k = NULL;
	PyObject * obj_dMax = NULL;
	int numProcs = -1;
	static char * keywords[] = {(char*)"kdtree", (char*)"queries", 
		(char*)"k", (char*)"dmax", (char*)"numprocs", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|OOi", keywords, 
		&obj_kdtree, &obj_queries, &obj_k, &obj_dMax, &numProcs)) {
		PyErr_SetString(PyExc_TypeError, "Failed parsing arguments");
		return NULL;
	}
	// check option obj_k and set k
	int k;
	if (obj_k == NULL) {
		k = 1;
	} else if (obj_k == Py_None) {
		k = -1;
	} else if (PyInt_Check(obj_k)) {
		k = (int)PyInt_AsLong(obj_k);
		if (k <= 0) {
			PyErr_Format(PyExc_ValueError, 
				"Encountered non-positive k=%d", k);
			return NULL;
		}
	} else {
		PyErr_Format(PyExc_TypeError, "Encountered %s k. "
			"Require int or None, ", obj_k->ob_type->tp_name);
		return NULL;
	}
	// check option obj_dMax and set dMax
	float dMax;
	if (obj_dMax == NULL) {
		dMax = flt_infinity;
	} else if (obj_dMax == Py_None) {
		dMax = flt_infinity;
	} else if (PyFloat_Check(obj_dMax)) {
		dMax = (float)PyFloat_AsDouble(obj_dMax);
		if (dMax <= 0.0f) {
			PyErr_SetString(PyExc_ValueError, 
				"Encountered non-positive dmax");
			return NULL;
		}
	} else {
		PyErr_Format(PyExc_TypeError, "Encountered %s dmax. "
			"Require float or None.", obj_dMax->ob_type->tp_name);
		return NULL;
	}
	// get pointer to k-d tree
	const PointKdTree<float> * tree;
	if (PyCapsule_CheckExact(obj_kdtree)) {
		tree = (PointKdTree<float> *)PyCapsule_GetPointer(obj_kdtree, NULL);
	} else {
		PyErr_SetString(PyExc_TypeError,
			"First arg must be capsule of k-d tree pointer");
		return NULL;
	}
	// toggle between various query types
	// note: at this point, k is either -1 or positive
	// and dMax is either finite positive or inf
	if (k > 0) {
		// k-nearest
		return k_nearest(obj_queries, tree, k, dMax);
	} else if (dMax != flt_infinity) {
		// r-near
		PyErr_SetString(PyExc_NotImplementedError, 
			"r-near not yet supported");
		return NULL;
	} else {
		PyErr_SetString(PyExc_ValueError, 
			"Invalid combination of k and dmax options");
		return NULL;
	}
}

static PyMethodDef methods_table[] = {
	{"_build", (PyCFunction)_build, METH_KEYWORDS, "build k-d tree"},
	{"_query", (PyCFunction)_query, METH_KEYWORDS, "performs either k-nearest or r-near"},
	{NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initkdtree(void) {
	Py_InitModule("kdtree", methods_table);
	import_array();
}
