#include <Python.h>
#include "arrayobject.h"
//#define NPY_NO_DEPRECATED_API
//#define NPY_1_7_API_VERSION
#include <iostream>
using namespace std;

static PyObject * printlist(PyObject * self, PyObject * args) {
	cout << "printlist: not implemented" << endl;
	PyObject * x;
	cout << PyTuple_Check(args) << endl;
	if (!PyArg_ParseTuple(args,"O",&x)) {
		PyErr_SetString(PyExc_TypeError, "printlist: could not"
			" parse arguments");
		return NULL;
	}
	if (!PyList_Check(x)) {
		PyErr_SetString(PyExc_TypeError, "printlist: requires list");
		return NULL;
	}
	Py_ssize_t n = PyList_Size(x);
	for (Py_ssize_t i = 0; i < n; i++) {
		PyObject * item = PyList_GetItem(x,i);
		cout << item->ob_type->tp_name << ": ";
		if (PyNumber_Check(item)) {
			cout << PyFloat_AsDouble(PyNumber_Float(item));
		}
		cout << endl;
	}
	Py_RETURN_NONE;
}

static PyObject * printtuple(PyObject * self, PyObject * args) {
	cout << "printtuple: not implemented" << endl;
	Py_RETURN_NONE;
}

static PyObject * makelist(PyObject * self, PyObject * args) {
	PyObject * x = PyList_New(3);
	PyObject * first_item = PyList_GetItem(x,0);
	cout << first_item << endl;

	PyObject * num_one = PyFloat_FromDouble(1);
	PyObject * num_two = PyFloat_FromDouble(2);
	PyObject * num_three = PyFloat_FromDouble(3);
	PyList_SetItem(x, 0, num_one);
	PyList_SetItem(x, 1, num_two);
	PyList_SetItem(x, 2, num_three);
	return x;
}

static PyObject * maketuple(PyObject * self, PyObject * args) {
	PyObject * x = PyTuple_New(3);
	PyTuple_SetItem(x, 0, PyFloat_FromDouble(1));
	PyTuple_SetItem(x, 1, PyFloat_FromDouble(2));
	PyTuple_SetItem(x, 2, PyFloat_FromDouble(3));
	return x;
}

static PyObject * makendarray(PyObject * self, PyObject * args) {
	npy_intp dims = 3;
	PyArrayObject * x = (PyArrayObject*)
		PyArray_EMPTY(1, &dims, NPY_FLOAT32, 0);
	cout << "array size: " << PyArray_DIMS(x)[0] << endl;
	cout << "stride size: " << PyArray_STRIDES(x)[0] << endl;
	float * data = (float*)PyArray_DATA(x);
	data[0] = 1.0f;
	data[1] = 2.0f;
	data[2] = 3.0f;
	return (PyObject*)x;
	Py_RETURN_NONE;
}

static PyObject * listofmeans(PyObject * self, PyObject * args) {
	PyObject * x;
	if (!PyArg_ParseTuple(args, "O", &x)) {
		PyErr_SetString(PyExc_TypeError, "failed to parse arguments");
		return NULL;
	}
	if (!PyList_Check(x)) {
		PyErr_SetString(PyExc_TypeError, "requires list");
		return NULL;
	}
	Py_ssize_t n = PyList_Size(x);
	PyObject * results = PyList_New(n);
	for (Py_ssize_t i = 0; i < n; i++) {
		PyObject * item = PyList_GetItem(x, i); 
		if (!PyArray_Check(item)) {
			PyErr_SetString(PyExc_TypeError, "requires list of arrays");
			Py_DECREF(results);
			return NULL;
		}
		if (PyArray_TYPE(item) != NPY_FLOAT32) {
			PyErr_SetString(PyExc_TypeError, "requires float32");
			Py_DECREF(results);
			return NULL;
		}
		if (PyArray_NDIM(item) != 2) {
			PyErr_SetString(PyExc_TypeError, "requires dims=2");
			Py_DECREF(results);
			return NULL;
		}
		npy_intp numel = PyArray_DIM(item,0) * PyArray_DIM(item,1);
		if (PyArray_STRIDE(item,1) != 4 ||
			PyArray_STRIDE(item,0) != 4 * PyArray_DIM(item,1)) {
			PyErr_SetString(PyExc_TypeError, "requires contiguous");
			Py_DECREF(results);
			return NULL;
		}
		float * data = (float*)PyArray_DATA(item);
		float mean = 0.0f;
		for (npy_intp j = 0; j < numel; j++) {
			mean += data[j];
		}
		mean /= numel;
		PyList_SetItem(results, i, PyFloat_FromDouble(mean));
	}
	return results;
}

static PyObject * listofmeanspy(PyObject * self, PyObject * args) {
	PyObject * x;
	if (!PyArg_ParseTuple(args, "O", &x)) {
		PyErr_SetString(PyExc_TypeError, "failed to parse arguments");
		return NULL;
	}
	if (!PyList_Check(x)) {
		PyErr_SetString(PyExc_TypeError, "requires list");
		return NULL;
	}
	Py_ssize_t n = PyList_Size(x);
	PyObject * results = PyList_New(n);
	for (Py_ssize_t i = 0; i < n; i++) {
		PyObject * item = PyList_GetItem(x, i); 
		if (!PyArray_Check(item)) {
			PyErr_SetString(PyExc_TypeError, "requires list of arrays");
			Py_DECREF(results);
			return NULL;
		}
		if (PyArray_TYPE(item) != NPY_FLOAT32) {
			PyErr_SetString(PyExc_TypeError, "requires float32");
			Py_DECREF(results);
			return NULL;
		}
		if (PyArray_NDIM(item) != 2) {
			PyErr_SetString(PyExc_TypeError, "requires dims=2");
			Py_DECREF(results);
			return NULL;
		}
		npy_intp numel = PyArray_DIM(item,0) * PyArray_DIM(item,1);
		if (PyArray_STRIDE(item,1) != 4 ||
			PyArray_STRIDE(item,0) != 4 * PyArray_DIM(item,1)) {
			PyErr_SetString(PyExc_TypeError, "requires contiguous");
			Py_DECREF(results);
			return NULL;
		}
		PyObject * mean = PyArray_Mean((PyArrayObject*)item, NPY_MAXDIMS, NPY_NOTYPE, NULL);  
		PyList_SetItem(results, i, mean);
	}
	return results;
}

static PyObject * xplusx(PyObject * self, PyObject * args) {
	PyObject * x;
	if (!PyArg_ParseTuple(args, "O", &x)) {
		PyErr_SetString(PyExc_TypeError, "failed to parse arguments");
		return NULL;
	}
	if (!PyList_Check(x)) {
		PyErr_SetString(PyExc_TypeError, "requires list");
		return NULL;
	}
	Py_ssize_t n = PyList_Size(x);
	PyObject * results = PyList_New(n);
	for (Py_ssize_t i = 0; i < n; i++) {
		PyObject * item = PyList_GetItem(x, i); 
		if (!PyArray_Check(item)) {
			PyErr_SetString(PyExc_TypeError, "requires list of arrays");
			Py_DECREF(results);
			return NULL;
		}
		if (PyArray_TYPE(item) != NPY_FLOAT32) {
			PyErr_SetString(PyExc_TypeError, "requires float32");
			Py_DECREF(results);
			return NULL;
		}
		if (PyArray_NDIM(item) != 2) {
			PyErr_SetString(PyExc_TypeError, "requires dims=2");
			Py_DECREF(results);
			return NULL;
		}
		npy_intp numel = PyArray_DIM(item,0) * PyArray_DIM(item,1);
		if (PyArray_STRIDE(item,1) != 4 ||
			PyArray_STRIDE(item,0) != 4 * PyArray_DIM(item,1)) {
			PyErr_SetString(PyExc_TypeError, "requires contiguous");
			Py_DECREF(results);
			return NULL;
		}
		//PyObject * sum = item->ob_type->tp_as_number->nb_add(item,item);

		npy_intp dims = 3;
		PyObject * sum = (PyObject*)
			PyArray_EMPTY(1, &dims, NPY_FLOAT32, 0);
		PyList_SetItem(results, i, sum);
	}
	return results;
}



static PyObject * computemean(PyObject * self, PyObject * args) {
	PyObject * x;
	if (!PyArg_ParseTuple(args, "O", &x)) {
		PyErr_SetString(PyExc_TypeError, "failed to parse arguments");
		return NULL;
	}
	if (!PyArray_Check(x)) {
		PyErr_SetString(PyExc_TypeError, "require ndarray");
		return NULL;
	}
	if (PyArray_TYPE(x) != NPY_FLOAT32) {
		PyErr_SetString(PyExc_TypeError, "requires float32");
		return NULL;
	}
	if (PyArray_NDIM(x) != 2) {
		PyErr_SetString(PyExc_TypeError, "requires dims=2");
		return NULL;
	}
	npy_intp numel = PyArray_DIM(x,0) * PyArray_DIM(x,1);
	if (PyArray_STRIDE(x,1) != 4 ||
		PyArray_STRIDE(x,0) != 4 * PyArray_DIM(x,1)) {
		PyErr_SetString(PyExc_TypeError, "requires contiguous");
		return NULL;
	}
	float * data = (float*)PyArray_DATA(x);
	float mean = 0.0f;
	for (npy_intp j = 0; j < numel; j++) {
		mean += data[j];
	}
	mean /= numel;
	return PyFloat_FromDouble(mean);
}
static PyMethodDef method_table[] = {
	{"printlist", printlist, METH_VARARGS, "prints contents of list"},
	{"printtuple", printtuple, METH_VARARGS, "prints contents of tuple"},
	{"makelist", makelist, METH_VARARGS, "returns list [1,2,3]"},
	{"maketuple", maketuple, METH_VARARGS, "returns tuple (1,2,3)"},
	{"makendarray", makendarray, METH_VARARGS, "returns ndarray([1,2,3])"},
	{"listofmeans", listofmeans, METH_VARARGS, "returns a list of means computed on a list of arrays"},
	{"listofmeanspy", listofmeanspy, METH_VARARGS, "returns a list of means computed on a list of arrays using PyArray_Mean"},
	{"xplusx", xplusx, METH_VARARGS, "adds each item in list to itself"},
	{"computemean", computemean, METH_VARARGS, "computes mean of ndarray"}
};

PyMODINIT_FUNC initfoo(void) {
	Py_InitModule("foo", method_table);
	import_array();
}
