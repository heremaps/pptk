#include "Python.h"
#include "arrayobject.h"
#include "PointKdTree.h"
#include "TraverserDF.h"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <vector>
#include <limits>
#include <omp.h>
#include <iostream>


using namespace vltools;
using namespace Eigen;
using namespace pointkd;
using namespace std;

typedef Matrix<float,Dynamic,3,RowMajor> Vectors;

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

void estimate_normals(Vectors & normals,
	const vector<float> & points, const std::size_t k, 
	float d_max = numeric_limits<float>::infinity())
{
	size_t numPoints = points.size() / 3;
	Map<const Vectors> P(&points[0], numPoints, 3);
	// Vectors P(numPoints, 3);
	PointKdTree<float> tree(points, 3);
	TraverserDF<float> trav(tree);
	normals.resize(numPoints, NoChange);
	omp_set_num_threads(omp_get_num_procs());
	std::cout << omp_get_num_procs() << std::endl;
	#pragma omp parallel for schedule(static, 1000)
	for (int i = 0; i < numPoints; i++) {
		if (i%10000==0)std::cout << omp_get_thread_num() << ": " << i << std::endl;
		vector<int> indices;
		vector<float> distances;
		vector<float> q(P.data() + i * 3, P.data() + (i + 1) * 3);
		trav.nearest(q, k, indices, distances, d_max);
		int num_neighbors;
		for (num_neighbors = 0; num_neighbors < k; num_neighbors++)
			if (indices[num_neighbors] == -1) break;
		Vectors X(num_neighbors, 3);
		for (size_t j = 0; j < num_neighbors; j++)
			X.row(j) = P.row(indices[j]);
		X.rowwise() -= X.colwise().mean();
		Matrix3f C = X.transpose() * X;
		SelfAdjointEigenSolver<Matrix3f> es(C);
		normals.row(i) = es.eigenvectors().col(0).transpose();
	}
}

static PyObject * estimate_normals_wrapper(PyObject * self, PyObject * args) {
	PyObject * p =NULL;
	int k;
	float d_max;
	if (!PyArg_ParseTuple(args, "Oif", &p, &k, &d_max)) {
		PyErr_SetString(PyExc_RuntimeError, "Failed to parse inputs");
		return NULL;
	}
	
	ArrayStruct arr;
	if (!extract_array(arr, p) || arr.type_num!=NPY_FLOAT32) {
		PyErr_SetString(PyExc_TypeError, "First argument must be 2-d array of float32s");
		return NULL;
	}
	
	std::vector<float> points;
	copyToVector(points, arr);
	
	Vectors normals;
	estimate_normals(normals, points, k, d_max);
	npy_intp dims[] = {arr.m, 3};
	PyObject * n = PyArray_EMPTY(2, dims, NPY_FLOAT32, false);
	copy(normals.data(), normals.data() + normals.size(), (float*)PyArray_DATA((PyArrayObject*)n));
	return n;
}

static PyMethodDef methods[] = {
	{"estimate_normals", estimate_normals_wrapper, METH_VARARGS, "estimate normals at 3d points"},
	{NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initestimate_normals(void) {
	(void) Py_InitModule("estimate_normals", methods);
	import_array();
}
// int main(int argc, char * argv[]) {
	// if (argc != 2 && argc != 3) {
		// cout << "usage: estimate_normals <xyz> [out]" << endl;
		// return EXIT_FAILURE;
	// }
	// std::vector<float> points;
	// readpoints(points, argv[1], 3);
	// cout << "number of points: " << points.size() / 3 << endl;
	// Vectors normals;
	// double t = getTime();
	// estimate_normals(normals, points, 64);
	// t = getTime() - t;
	// cout << "time to estimate normals: " << t << endl;
	// if (argc == 3) {
		// vector<float> v_normals(normals.data(),
			// normals.data() + normals.size());
		// writepoints(argv[2], v_normals, 3);
		// cout << "wrote " << normals.rows() <<
			// " normal vectors to " << argv[2] << endl;
	// }
	// return EXIT_SUCCESS;
// }
