#include "Python.h"
#include "arrayobject.h"
#define EIGEN_NO_DEBUG  // comment this for runtime assertions
#include <algorithm>
#include <functional>
#include <iostream>
#include <limits>
#include <vector>
#include "Eigen/Dense"

using namespace Eigen;
using namespace std;

typedef Matrix<float, Dynamic, Dynamic, RowMajor> MatrixType;
typedef Stride<Dynamic, Dynamic> StrideType;
typedef Map<MatrixType, Unaligned, StrideType> MapType;
typedef Map<const MatrixType, Unaligned, StrideType> MapTypeConst;

struct ListItem {
  const float* data;
  float scalar;
  npy_intp m;
  npy_intp n;
  npy_intp row_stride;
  npy_intp col_stride;
};

bool extract_scalar(ListItem& item, PyObject* obj) {
  // If obj is one of the following scalar types:
  // 1. Python scalar: boolean, integer, long, or float
  // 2. Numpy array scalar: float32
  // then the value represented by obj is casted to a float and
  // written to reference variable "scalar" and function returns true,
  // otherwise exception is set, and function returns false.
  item.data = &item.scalar;
  item.m = item.n = 1;
  item.row_stride = item.col_stride = 0;

  if (PyArray_IsScalar(obj, Float)) {
    PyArray_ScalarAsCtype(obj, (void*)&item.scalar);
    return true;
    /*
    } else {
      PyErr_Format(PyExc_TypeError,
        "Unsupported Numpy scalar type: %s",
        PyArray_DESCR((PyArrayObject*)obj)->typeobj->tp_name);
      return false;
    }
    */
  } else if (PyArray_IsPythonScalar(obj)) {
    // why is numpy.float32 array scalar going down this path?
    if (PyFloat_CheckExact(obj)) {
      item.scalar = (float)PyFloat_AsDouble(obj);
      return true;
#if PY_MAJOR_VERSION < 3
    } else if (PyInt_CheckExact(obj)) {
      item.scalar = (float)PyInt_AsLong(obj);
      return true;
#endif
    } else if (PyLong_CheckExact(obj)) {
      item.scalar = (float)PyLong_AsLong(obj);
      return true;
    } else if (PyBool_Check(obj)) {
      if (obj == Py_False) {
        item.scalar = 0.0f;
      } else {
        item.scalar = 1.0f;
      }
      return true;
    } else {
      PyErr_Format(PyExc_TypeError, "Unsupported Python scalar type: %s",
                   obj->ob_type->tp_name);
      return false;
    }
  } else {
    PyErr_Format(PyExc_TypeError,
                 "Object is neither a Python scalar nor "
                 "a Numpy array scalar of type float32");
    return false;
  }
}

bool extract_array(ListItem& item, PyObject* obj, Py_ssize_t item_idx,
                   const char* operand_name = "") {
  // Checks the following conditions:
  // 1. obj is a PyArrayObject
  // 2. obj's dtype is float32
  // 3. obj's ndim is either 1 or 2
  // If all the above conditions hold, then array properties are
  // extracted and function returns true.
  if (!PyArray_Check(obj)) {
    PyErr_Format(PyExc_TypeError,
                 "Encountered non-array type: "
                 "item %d of %slist operand",
                 (int)item_idx, operand_name);
    return false;
  }
  PyArrayObject* arr = (PyArrayObject*)obj;
  if (PyArray_TYPE(arr) != NPY_FLOAT32) {
    PyErr_Format(PyExc_TypeError,
                 "Array dtype must be float32: "
                 "item %d of %slist operand",
                 (int)item_idx, operand_name);
    return false;
  }
  int ndim = PyArray_NDIM(arr);
  if (ndim != 1 && ndim != 2) {
    PyErr_Format(PyExc_ValueError,
                 "Array ndim neither 1 nor 2: "
                 "item %d of %slist operand",
                 (int)item_idx, operand_name);
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
  if (item.m == 1) item.row_stride = 0;
  if (item.n == 1) item.col_stride = 0;
  item.data = (float*)PyArray_DATA(arr);
  return true;
}

bool extract_scalar_or_array(ListItem& item, PyObject* obj, npy_intp item_index,
                             const char* operand_name) {
  if (PyArray_Check(obj)) {
    if (!extract_array(item, obj, item_index, operand_name)) return false;
  } else if (PyArray_IsAnyScalar(obj)) {
    if (!extract_scalar(item, obj)) return false;
  } else {
    PyErr_Format(PyExc_TypeError, "Unsupported item type: %s",
                 obj->ob_type->tp_name);
    return false;
  }
  return true;
}

bool arrays_compatible(npy_intp& m_out, npy_intp& n_out,
                       npy_intp m_in1, npy_intp n_in1,
                       npy_intp m_in2, npy_intp n_in2, Py_ssize_t item_idx) {
  if (m_in1 != m_in2 && m_in1 != 1 && m_in2 != 1 ||
      n_in1 != n_in2 && n_in1 != 1 && n_in2 != 1) {
    PyErr_Format(PyExc_ValueError,
                 "Incompatiable array shapes (%d,%d) and (%d,%d) "
                 "encountered on %d-th list item",
                 (int)m_in1, (int)n_in1, (int)m_in2, (int)n_in2, (int)item_idx);
    return false;
  }
  m_out = m_in1 == 1 ? m_in2 : m_in1;
  n_out = n_in1 == 1 ? n_in2 : n_in1;
  return true;
}

bool extract_operands_to_binop(
    Py_ssize_t& out_len,
    PyObject*& X, Py_ssize_t& x_len, ListItem& x_struct,
    PyObject*& Y, Py_ssize_t& y_len, ListItem& y_struct,
    PyObject* args) {

  if (!PyArg_ParseTuple(args, "OO", &X, &Y)) {
    PyErr_SetString(PyExc_TypeError, "Failed to parse arguments");
    return false;
  }
  if (!PyList_Check(X) || !PyList_Check(Y)) {
    PyErr_SetString(PyExc_TypeError, "Requires list inputs");
    return false;
  }
  x_len = PyList_Size(X);
  y_len = PyList_Size(Y);
  if (x_len == 0 || y_len == 0) {
    PyErr_SetString(PyExc_ValueError, "Requres non-empty lists");
    return false;
  }
  if (x_len != y_len && x_len != 1 && y_len != 1) {
    PyErr_Format(PyExc_ValueError, "Incompatiable input list lengths %d,%d",
                 (int)x_len, (int)y_len);
    return false;
  }
  // note: at this point, x_len > 0, y_len > 0 and
  // at least one of the following are true:
  // 1. x_len == y_len
  // 2. x_len == 1
  // 3. y_len == 1

  if (x_len == 1) {
    PyObject* x_item = PyList_GetItem(X, 0);
    if (!extract_scalar_or_array(x_struct, x_item, 0, "left ")) return false;
  }
  if (y_len == 1) {
    PyObject* y_item = PyList_GetItem(Y, 0);
    if (!extract_scalar_or_array(y_struct, y_item, 0, "right ")) return false;
  }
  out_len = x_len > y_len ? x_len : y_len;
  return true;
}

template <class T>
static PyObject* binary_op_single(PyObject* self, PyObject* args) {
  T op;
  PyObject *X, *Y;
  if (!PyArg_ParseTuple(args, "OO", &X, &Y)) {
    PyErr_SetString(PyExc_TypeError, "Failed to parse arguments.");
    return NULL;
  }
  ListItem x_struct, y_struct;
  if (!extract_scalar_or_array(x_struct, X, 0, "left ")) return NULL;
  if (!extract_scalar_or_array(y_struct, Y, 0, "right ")) return NULL;
  npy_intp out_dims[2];
  if (!arrays_compatible(out_dims[0], out_dims[1],
                         x_struct.m, x_struct.n, y_struct.m, y_struct.n, 0))
    return NULL;
  PyArrayObject* out_item =
      (PyArrayObject*)PyArray_EMPTY(2, out_dims, NPY_FLOAT32, 0);
  float* out_data = (float*)PyArray_DATA(out_item);
  const float* x_ptr = x_struct.data;
  const float* y_ptr = y_struct.data;
  for (npy_intp r = 0; r < out_dims[0]; r++) {
    for (npy_intp c = 0; c < out_dims[1]; c++) {
      *(out_data++) = op(*(x_ptr + c * x_struct.col_stride),
                         *(y_ptr + c * y_struct.col_stride));
    }
    x_ptr += x_struct.row_stride;
    y_ptr += y_struct.row_stride;
  }
  return (PyObject*)out_item;
}

template <class T>
static PyObject* binary_op(PyObject* self, PyObject* args) {
  T op;
  PyObject *X, *Y;
  ListItem x_struct, y_struct;
  Py_ssize_t x_len, y_len, out_len;
  if (!extract_operands_to_binop(out_len,
                                 X, x_len, x_struct,
                                 Y, y_len, y_struct, args))
    return NULL;

  PyObject* out_list = PyList_New(out_len);
  for (Py_ssize_t i = 0; i < out_len; i++) {
    // allowing for x_item and y_item to be scalars adds ~50ns per
    // iteration. to restrict to arrays, use just extract_array()
    if (x_len != 1) {
      PyObject* x_item = PyList_GetItem(X, i);
      if (!extract_scalar_or_array(x_struct, x_item, i, "left ")) return NULL;
    }
    if (y_len != 1) {
      PyObject* y_item = PyList_GetItem(Y, i);
      if (!extract_scalar_or_array(y_struct, y_item, i, "right ")) return NULL;
    }
    npy_intp out_dims[2];
    if (!arrays_compatible(out_dims[0], out_dims[1], x_struct.m, x_struct.n,
                           y_struct.m, y_struct.n, i))
      return NULL;
    PyArrayObject* out_item =
        (PyArrayObject*)PyArray_EMPTY(2, out_dims, NPY_FLOAT32, 0);
    float* out_data = (float*)PyArray_DATA(out_item);
    const float* x_ptr = x_struct.data;
    const float* y_ptr = y_struct.data;
    for (npy_intp r = 0; r < out_dims[0]; r++) {
      for (npy_intp c = 0; c < out_dims[1]; c++) {
        *(out_data++) = op(*(x_ptr + c * x_struct.col_stride),
                           *(y_ptr + c * y_struct.col_stride));
      }
      x_ptr += x_struct.row_stride;
      y_ptr += y_struct.row_stride;
    }
    PyList_SetItem(out_list, i, (PyObject*)out_item);
  }
  return out_list;
}

typedef PyObject* (*py_fn_ptr)(PyObject*, PyObject*);
py_fn_ptr _add = &binary_op<plus<float> >;
py_fn_ptr _sub = &binary_op<minus<float> >;
py_fn_ptr _mul = &binary_op<multiplies<float> >;
py_fn_ptr _div = &binary_op<divides<float> >;

py_fn_ptr _sadd = &binary_op_single<plus<float> >;
py_fn_ptr _ssub = &binary_op_single<minus<float> >;
py_fn_ptr _smul = &binary_op_single<multiplies<float> >;
py_fn_ptr _sdiv = &binary_op_single<divides<float> >;

template <typename T>
struct TypeMap {};

template <>
struct TypeMap<bool> {
  static const int type_num = NPY_BOOL;
};

template <>
struct TypeMap<float> {
  static const int type_num = NPY_FLOAT32;
};

template <>
struct TypeMap<npy_int64> {
  static const int type_num = NPY_INT64;
};

template <class Reducer>
static PyObject* reduction_op_single(PyObject* self, PyObject* args,
                                     PyObject* kwdict) {
  typedef typename Reducer::ResultType ResultType;
  PyObject* X;
  PyObject* axis_obj = Py_None;
  char* keywords[] = {(char*)"X", (char*)"axis", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwdict, "O|O", keywords, &X,
                                   &axis_obj)) {
    PyErr_SetString(PyExc_TypeError, "Failed to parse arguments");
    return NULL;
  }
  int axis;
  if (axis_obj != Py_None) {
    PyObject* temp = PyNumber_Long(axis_obj);
    if (temp) {
      axis = (int)PyLong_AsLong(temp);
      Py_DECREF(temp);
    } else {
      PyErr_Format(PyExc_TypeError,
                   "Type %s object passed "
                   "as axis option",
                   axis_obj->ob_type->tp_name);
      return NULL;
    }
  } else {
    axis = -1;  // denotes reduction over entire matrix
  }
  if (axis < -1 || axis > 1) {
    PyErr_Format(PyExc_ValueError, "Invalid axis value: %d", axis);
    return NULL;
  }
  if (axis == -1) {
    ListItem x_struct;
    if (!extract_array(x_struct, X, 0)) {
      return NULL;
    }

    const float* x_ptr = x_struct.data;
    Reducer reducer;
    reducer.initialize();
    for (npy_intp r = 0; r < x_struct.m; r++) {
      for (npy_intp c = 0; c < x_struct.n; c++)
        reducer.accumulate(*(x_ptr + c * x_struct.col_stride));
      x_ptr += x_struct.row_stride;
    }

    npy_intp out_dim = 1;
    PyArrayObject* out_item = (PyArrayObject*)PyArray_EMPTY(
        1, &out_dim, TypeMap<ResultType>::type_num, 0);
    ResultType* out_data = (ResultType*)PyArray_DATA(out_item);
    *out_data = reducer.finalize();
    PyObject* out_scalar = PyArray_ToScalar((void*)out_data, out_item);
    Py_DECREF(out_item);  // mark out_item for garbage collection
    return out_scalar;
  } else if (axis == 0) {
    ListItem x_struct;
    if (!extract_array(x_struct, X, 0)) {
      return NULL;
    }

    npy_intp out_dims[2] = {1, x_struct.n};
    PyObject* out_item =
        PyArray_EMPTY(2, out_dims, TypeMap<ResultType>::type_num, 0);
    ResultType* out_data = (ResultType*)PyArray_DATA((PyArrayObject*)out_item);
    const float* x_ptr = x_struct.data;
    Reducer reducer;
    for (npy_intp c = 0; c < x_struct.n; c++) {
      reducer.initialize();
      for (npy_intp r = 0; r < x_struct.m; r++)
        reducer.accumulate(*(x_ptr + r * x_struct.row_stride));
      x_ptr += x_struct.col_stride;
      out_data[c] = reducer.finalize();
    }
    return out_item;
  } else {  // axis == 1
    ListItem x_struct;
    if (!extract_array(x_struct, X, 0)) {
      return NULL;
    }

    npy_intp out_dims[2] = {x_struct.m, 1};
    PyObject* out_item =
        PyArray_EMPTY(2, out_dims, TypeMap<ResultType>::type_num, 0);
    ResultType* out_data = (ResultType*)PyArray_DATA((PyArrayObject*)out_item);
    const float* x_ptr = x_struct.data;
    Reducer reducer;
    for (npy_intp r = 0; r < x_struct.m; r++) {
      reducer.initialize();
      for (npy_intp c = 0; c < x_struct.n; c++)
        reducer.accumulate(*(x_ptr + c * x_struct.col_stride));
      x_ptr += x_struct.row_stride;
      out_data[r] = reducer.finalize();
    }
    return out_item;
  }
}

template <class Reducer>
static PyObject* reduction_op(PyObject* self, PyObject* args,
                              PyObject* kwdict) {
  typedef typename Reducer::ResultType ResultType;
  PyObject* X;
  PyObject* axis_obj = Py_None;
  char* keywords[] = {(char*)"X", (char*)"axis", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwdict, "O|O", keywords, &X,
                                   &axis_obj)) {
    PyErr_SetString(PyExc_TypeError, "Failed to parse arguments");
    return NULL;
  }
  if (!PyList_Check(X)) {
    PyErr_SetString(PyExc_TypeError, "Requires list input");
    return NULL;
  }
  int axis;
  if (axis_obj != Py_None) {
    PyObject* temp = PyNumber_Long(axis_obj);
    if (temp) {
      axis = (int)PyLong_AsLong(temp);
      Py_DECREF(temp);
    } else {
      PyErr_Format(PyExc_TypeError,
                   "Type %s object passed "
                   "as axis option",
                   axis_obj->ob_type->tp_name);
      return NULL;
    }
  } else {
    axis = -1;  // denotes reduction over entire matrix
  }
  if (axis < -1 || axis > 1) {
    PyErr_Format(PyExc_ValueError, "Invalid axis value: %d", axis);
    return NULL;
  }

  Py_ssize_t x_len = PyList_Size(X);
  PyObject* out_list = PyList_New(x_len);
  if (axis == -1) {
    for (Py_ssize_t i = 0; i < x_len; i++) {
      PyObject* x_item = PyList_GetItem(X, i);
      ListItem x_struct;
      if (!extract_array(x_struct, x_item, i)) {
        return NULL;
      }

      const float* x_ptr = x_struct.data;
      Reducer reducer;
      reducer.initialize();
      for (npy_intp r = 0; r < x_struct.m; r++) {
        for (npy_intp c = 0; c < x_struct.n; c++)
          reducer.accumulate(*(x_ptr + c * x_struct.col_stride));
        x_ptr += x_struct.row_stride;
      }

      npy_intp out_dim = 1;
      PyArrayObject* out_item = (PyArrayObject*)PyArray_EMPTY(
          1, &out_dim, TypeMap<ResultType>::type_num, 0);
      ResultType* out_data = (ResultType*)PyArray_DATA(out_item);
      *out_data = reducer.finalize();
      PyObject* out_scalar = PyArray_ToScalar((void*)out_data, out_item);
      Py_DECREF(out_item);  // mark out_item for garbage collection
      PyList_SetItem(out_list, i, out_scalar);
    }
  } else if (axis == 0) {
    for (Py_ssize_t i = 0; i < x_len; i++) {
      PyObject* x_item = PyList_GetItem(X, i);
      ListItem x_struct;
      if (!extract_array(x_struct, x_item, i)) {
        return NULL;
      }

      npy_intp out_dims[2] = {1, x_struct.n};
      PyObject* out_item =
          PyArray_EMPTY(2, out_dims, TypeMap<ResultType>::type_num, 0);
      ResultType* out_data =
          (ResultType*)PyArray_DATA((PyArrayObject*)out_item);
      const float* x_ptr = x_struct.data;
      Reducer reducer;
      for (npy_intp c = 0; c < x_struct.n; c++) {
        reducer.initialize();
        for (npy_intp r = 0; r < x_struct.m; r++)
          reducer.accumulate(*(x_ptr + r * x_struct.row_stride));
        x_ptr += x_struct.col_stride;
        out_data[c] = reducer.finalize();
      }
      PyList_SetItem(out_list, i, out_item);
    }
  } else {  // axis == 1
    for (Py_ssize_t i = 0; i < x_len; i++) {
      PyObject* x_item = PyList_GetItem(X, i);
      ListItem x_struct;
      if (!extract_array(x_struct, x_item, i)) {
        return NULL;
      }

      npy_intp out_dims[2] = {x_struct.m, 1};
      PyObject* out_item =
          PyArray_EMPTY(2, out_dims, TypeMap<ResultType>::type_num, 0);
      ResultType* out_data =
          (ResultType*)PyArray_DATA((PyArrayObject*)out_item);
      const float* x_ptr = x_struct.data;
      Reducer reducer;
      for (npy_intp r = 0; r < x_struct.m; r++) {
        reducer.initialize();
        for (npy_intp c = 0; c < x_struct.n; c++)
          reducer.accumulate(*(x_ptr + c * x_struct.col_stride));
        x_ptr += x_struct.row_stride;
        out_data[r] = reducer.finalize();
      }
      PyList_SetItem(out_list, i, out_item);
    }
  }
  return out_list;
}

template <typename T>
struct reducer_sum {
  typedef T ResultType;
  T res;
  void initialize() { res = 0; }
  void accumulate(T v) { res += v; }
  ResultType finalize() { return res; }
};

template <typename T>
struct reducer_mean {
  typedef float ResultType;
  float sum;
  npy_int64 count;
  void initialize() {
    sum = 0.0f;
    count = 0;
  }
  void accumulate(T v) {
    sum += v;
    count++;
  }
  ResultType finalize() { return sum / count; }
};

template <typename T>
struct reducer_prod {
  typedef T ResultType;
  T res;
  void initialize() { res = 1; }
  void accumulate(T v) { res *= v; }
  ResultType finalize() { return res; }
};

template <typename T>
struct reducer_all {
  typedef bool ResultType;
  bool res;
  void initialize() { res = true; }
  void accumulate(T v) { res = res && (bool)v; }
  ResultType finalize() { return res; }
};

template <typename T>
struct reducer_any {
  typedef bool ResultType;
  bool res;
  void initialize() { res = false; }
  void accumulate(T v) { res = res || (bool)v; }
  ResultType finalize() { return res; }
};

template <typename T>
struct reducer_min {
  typedef T ResultType;
  T res;
  void initialize() { res = numeric_limits<T>::max(); }
  void accumulate(T v) { res = v < res ? v : res; }
  ResultType finalize() { return res; }
};

template <typename T>
struct reducer_max {
  typedef T ResultType;
  T res;
  void initialize() { res = -numeric_limits<T>::max(); }
  void accumulate(T v) { res = v > res ? v : res; }
  ResultType finalize() { return res; }
};

template <typename T>
struct reducer_argmin {
  typedef npy_int64 ResultType;
  T min_val;
  npy_int64 curr_index;
  npy_int64 min_val_index;
  void initialize() {
    min_val = numeric_limits<T>::max();
    curr_index = min_val_index = 0;
  }
  void accumulate(T v) {
    if (v < min_val) {
      min_val = v;
      min_val_index = curr_index;
    }
    curr_index++;
  }
  ResultType finalize() { return min_val_index; }
};

template <typename T>
struct reducer_argmax {
  typedef npy_int64 ResultType;
  T max_val;
  npy_int64 curr_index;
  npy_int64 max_val_index;
  void initialize() {
    max_val = -numeric_limits<T>::max();
    curr_index = max_val_index = 0;
  }
  void accumulate(T v) {
    if (v > max_val) {
      max_val = v;
      max_val_index = curr_index;
    }
    curr_index++;
  }
  ResultType finalize() { return max_val_index; }
};

typedef PyObject* (*py_kwarg_fn_ptr)(PyObject*, PyObject*, PyObject*);
py_kwarg_fn_ptr _sum = &reduction_op<reducer_sum<float> >;
py_kwarg_fn_ptr _mean = &reduction_op<reducer_mean<float> >;
py_kwarg_fn_ptr _prod = &reduction_op<reducer_prod<float> >;
py_kwarg_fn_ptr _all = &reduction_op<reducer_all<float> >;
py_kwarg_fn_ptr _any = &reduction_op<reducer_any<float> >;
py_kwarg_fn_ptr _min = &reduction_op<reducer_min<float> >;
py_kwarg_fn_ptr _max = &reduction_op<reducer_max<float> >;
py_kwarg_fn_ptr _argmin = &reduction_op<reducer_argmin<float> >;
py_kwarg_fn_ptr _argmax = &reduction_op<reducer_argmax<float> >;

py_kwarg_fn_ptr _ssum = &reduction_op_single<reducer_sum<float> >;
py_kwarg_fn_ptr _smean = &reduction_op_single<reducer_mean<float> >;
py_kwarg_fn_ptr _sprod = &reduction_op_single<reducer_prod<float> >;
py_kwarg_fn_ptr _sall = &reduction_op_single<reducer_all<float> >;
py_kwarg_fn_ptr _sany = &reduction_op_single<reducer_any<float> >;
py_kwarg_fn_ptr _smin = &reduction_op_single<reducer_min<float> >;
py_kwarg_fn_ptr _smax = &reduction_op_single<reducer_max<float> >;
py_kwarg_fn_ptr _sargmin = &reduction_op_single<reducer_argmin<float> >;
py_kwarg_fn_ptr _sargmax = &reduction_op_single<reducer_argmax<float> >;

static PyObject* _sdot(PyObject* self, PyObject* args) {
  PyObject *X, *Y;
  if (!PyArg_ParseTuple(args, "OO", &X, &Y)) {
    PyErr_SetString(PyExc_TypeError, "Failed to parse arguments");
    return NULL;
  }
  ListItem x_struct, y_struct;
  if (!extract_scalar_or_array(x_struct, X, 0, "left ")) return NULL;
  if (!extract_scalar_or_array(y_struct, Y, 0, "right ")) return NULL;
  bool x_is_scalar = x_struct.m == 1 && x_struct.n == 1;
  bool y_is_scalar = y_struct.m == 1 && y_struct.n == 1;
  if (x_struct.n != y_struct.m && !x_is_scalar && !y_is_scalar) {
    PyErr_Format(
        PyExc_ValueError, "Incompatible matrix sizes (%d, %d), (%d, %d)",
        (int)x_struct.m, (int)x_struct.n, (int)y_struct.m, (int)y_struct.n);
    return NULL;
  }
  npy_intp out_dims[2];
  if (x_is_scalar) {
    out_dims[0] = y_struct.m;
    out_dims[1] = y_struct.n;
  } else if (y_is_scalar) {
    out_dims[0] = x_struct.m;
    out_dims[1] = x_struct.n;
  } else {
    out_dims[0] = x_struct.m;
    out_dims[1] = y_struct.n;
  }
  PyArrayObject* out_item =
      (PyArrayObject*)PyArray_EMPTY(2, out_dims, NPY_FLOAT32, 0);
  float* out_data = (float*)PyArray_DATA(out_item);
  if (x_is_scalar) {
    const float x_val = *x_struct.data;
    const float* y_ptr = y_struct.data;
    for (npy_intp r = 0; r < y_struct.m; r++) {
      for (npy_intp c = 0; c < y_struct.n; c++)
        *out_data++ = x_val * *(y_ptr + c * y_struct.col_stride);
      y_ptr += y_struct.row_stride;
    }
  } else if (y_is_scalar) {
    const float* x_ptr = x_struct.data;
    const float y_val = *y_struct.data;
    for (npy_intp r = 0; r < x_struct.m; r++) {
      for (npy_intp c = 0; c < x_struct.n; c++)
        *out_data++ = *(x_ptr + c * x_struct.col_stride) * y_val;
      x_ptr += x_struct.row_stride;
    }
  } else {
    const float* x_ptr = x_struct.data;
    for (npy_intp r = 0; r < out_dims[0]; r++) {
      const float* y_ptr = y_struct.data;
      for (npy_intp c = 0; c < out_dims[1]; c++) {
        *out_data = 0.0f;
        for (npy_intp k = 0; k < x_struct.n; k++)
          *out_data += *(x_ptr + k * x_struct.col_stride) *
                       *(y_ptr + k * y_struct.row_stride);
        out_data++;
        y_ptr += y_struct.col_stride;
      }
      x_ptr += x_struct.row_stride;
    }
  }
  return (PyObject*)out_item;
}

static PyObject* _dot(PyObject* self, PyObject* args) {
  PyObject *X, *Y;
  ListItem x_struct, y_struct;
  Py_ssize_t x_len, y_len, out_len;
  if (!extract_operands_to_binop(out_len, X, x_len, x_struct, Y, y_len,
                                 y_struct, args))
    return NULL;

  PyObject* out_list = PyList_New(out_len);
  for (Py_ssize_t i = 0; i < out_len; i++) {
    if (x_len != 1) {
      PyObject* x_item = PyList_GetItem(X, i);
      if (!extract_scalar_or_array(x_struct, x_item, i, "left ")) return NULL;
    }
    if (y_len != 1) {
      PyObject* y_item = PyList_GetItem(Y, i);
      if (!extract_scalar_or_array(y_struct, y_item, i, "right ")) return NULL;
    }
    bool x_is_scalar = x_struct.m == 1 && x_struct.n == 1;
    bool y_is_scalar = y_struct.m == 1 && y_struct.n == 1;
    if (x_struct.n != y_struct.m && !x_is_scalar && !y_is_scalar) {
      PyErr_Format(PyExc_ValueError,
                   "Incompatible matrix sizes (%d, %d), (%d, %d). (item %d)",
                   (int)x_struct.m, (int)x_struct.n,
                   (int)y_struct.m, (int)y_struct.n, (int)i);
      return NULL;
    }
    npy_intp out_dims[2];
    if (x_is_scalar) {
      out_dims[0] = y_struct.m;
      out_dims[1] = y_struct.n;
    } else if (y_is_scalar) {
      out_dims[0] = x_struct.m;
      out_dims[1] = x_struct.n;
    } else {
      out_dims[0] = x_struct.m;
      out_dims[1] = y_struct.n;
    }
    PyArrayObject* out_item =
        (PyArrayObject*)PyArray_EMPTY(2, out_dims, NPY_FLOAT32, 0);
    float* out_data = (float*)PyArray_DATA(out_item);
    // note: not sure why using Eigen's matrix multiply is
    // alot slower than a naive triple for-loop matrix multiply:
    // multiplying two 3x3 matrices 1M times takes:
    // ~1.6 s for Eigen and just ~0.4 s for hand-rolled code
    // todo: profile Eigen matrix multiply in separate cpp file
#if 0
    MapType mat_out(out_data, out_dims[0], out_dims[1],
      StrideType(out_dims[1],1));
    MapTypeConst mat_x(x_struct.data, x_struct.m, x_struct.n,
      StrideType(x_struct.row_stride, x_struct.col_stride));
    MapTypeConst mat_y(y_struct.data, y_struct.m, y_struct.n,
      StrideType(y_struct.row_stride, y_struct.col_stride));
    if (x_is_scalar)
      mat_out.noalias() = mat_x(0,0) * mat_y;
    else if (y_is_scalar)
      mat_out.noalias() = mat_x * mat_y(0,0);
    else
      mat_out.noalias() = mat_x * mat_y;
#else
    if (x_is_scalar) {
      const float x_val = *x_struct.data;
      const float* y_ptr = y_struct.data;
      for (npy_intp r = 0; r < y_struct.m; r++) {
        for (npy_intp c = 0; c < y_struct.n; c++)
          *out_data++ = x_val * *(y_ptr + c * y_struct.col_stride);
        y_ptr += y_struct.row_stride;
      }
    } else if (y_is_scalar) {
      const float* x_ptr = x_struct.data;
      const float y_val = *y_struct.data;
      for (npy_intp r = 0; r < x_struct.m; r++) {
        for (npy_intp c = 0; c < x_struct.n; c++)
          *out_data++ = *(x_ptr + c * x_struct.col_stride) * y_val;
        x_ptr += x_struct.row_stride;
      }
    } else {
      const float* x_ptr = x_struct.data;
      for (npy_intp r = 0; r < out_dims[0]; r++) {
        const float* y_ptr = y_struct.data;
        for (npy_intp c = 0; c < out_dims[1]; c++) {
          *out_data = 0.0f;
          for (npy_intp k = 0; k < x_struct.n; k++)
            *out_data += *(x_ptr + k * x_struct.col_stride) *
                         *(y_ptr + k * y_struct.row_stride);
          out_data++;
          y_ptr += x_struct.col_stride;
        }
        x_ptr += x_struct.row_stride;
      }
    }
#endif
    PyList_SetItem(out_list, i, (PyObject*)out_item);
  }
  return out_list;
}

static PyObject* _stranspose(PyObject* self, PyObject* args) {
  PyObject* X;
  if (!PyArg_ParseTuple(args, "O", &X)) {
    PyErr_SetString(PyExc_TypeError, "Failed to parse arguments");
    return NULL;
  }
  ListItem x_struct;
  if (!extract_array(x_struct, X, 0)) return NULL;
  MapTypeConst mat_x(x_struct.data, x_struct.m, x_struct.n,
                     StrideType(x_struct.row_stride, x_struct.col_stride));
  npy_intp out_dims[2] = {x_struct.n, x_struct.m};
  PyArrayObject* out_item =
      (PyArrayObject*)PyArray_EMPTY(2, out_dims, NPY_FLOAT32, 0);
  float* out_data = (float*)PyArray_DATA(out_item);
  MapType mat_out(out_data, out_dims[0], out_dims[1],
                  StrideType(out_dims[1], 1));
  mat_out = mat_x.transpose();
  return (PyObject*)out_item;
}

static PyObject* _transpose(PyObject* self, PyObject* args) {
  PyObject* X;
  if (!PyArg_ParseTuple(args, "O", &X)) {
    PyErr_SetString(PyExc_TypeError, "Failed to parse arguments");
    return NULL;
  }
  if (!PyList_Check(X)) {
    PyErr_SetString(PyExc_TypeError, "Requires list as first argument");
    return NULL;
  }
  Py_ssize_t x_len = PyList_Size(X);
  if (x_len == 0) {
    PyErr_SetString(PyExc_ValueError, "Requres non-empty lists");
    return NULL;
  }
  Py_ssize_t out_len = x_len;
  PyObject* out_list = PyList_New(out_len);
  for (Py_ssize_t i = 0; i < out_len; i++) {
    ListItem x_struct;
    if (!extract_array(x_struct, PyList_GetItem(X, i), i, "indexee "))
      return NULL;
    MapTypeConst mat_x(x_struct.data, x_struct.m, x_struct.n,
                       StrideType(x_struct.row_stride, x_struct.col_stride));
    npy_intp out_dims[2] = {x_struct.n, x_struct.m};
    PyArrayObject* out_item =
        (PyArrayObject*)PyArray_EMPTY(2, out_dims, NPY_FLOAT32, 0);
    float* out_data = (float*)PyArray_DATA(out_item);
    MapType mat_out(out_data, out_dims[0], out_dims[1],
                    StrideType(out_dims[1], 1));
    mat_out = mat_x.transpose();
    PyList_SetItem(out_list, i, (PyObject*)out_item);
  }
  return out_list;
}

static PyObject* _abs(PyObject* self, PyObject* args) {
  cout << "not implemented" << endl;
  Py_RETURN_NONE;
}

bool extract_integer(npy_intp& number, PyObject* item, npy_intp item_index) {
  PyObject* number_obj = PyNumber_Long(item);
  if (number_obj == NULL) {
    PyErr_Format(PyExc_TypeError,
                 "Encountered non-\"int()-able\" %s object on %d-th item",
                 item->ob_type->tp_name, (int)item_index);
    return false;
  }
  number = (npy_intp)PyLong_AsLong(number_obj);
  Py_DECREF(number_obj);
  return true;
}

bool extract_indices(vector<npy_intp>& indices, PyObject* obj,
                     Py_ssize_t item_idx) {
  // obj must be one of the following, otherwise return false
  // 1. Python list of objects convertable into an integer
  //    via PyNumber_Long (i.e. int(x))
  // 2. 1-d numpy array of integer indices ((u)int32 or (u)int64)
  // 3. 1-d numpy array of booleans
  indices.clear();
  if (PyList_Check(obj)) {
    Py_ssize_t len = PyList_Size(obj);
    indices.reserve(len);
    for (Py_ssize_t i = 0; i < len; i++) {
      npy_intp number;
      if (!extract_integer(number, PyList_GetItem(obj, i), item_idx))
        return NULL;
      indices.push_back(number);
    }
    return true;
  } else if (PyArray_Check(obj)) {
    PyArrayObject* arr = (PyArrayObject*)obj;
    if (PyArray_NDIM(arr) != 1) {
      PyErr_Format(PyExc_ValueError,
                   "On %d-th item, numpy array has ndim = %d. "
                   "Array of indices can only have ndim = 1.",
                   (int)item_idx, PyArray_NDIM(arr));
      return false;
    }
    npy_intp arr_len = PyArray_DIM(arr, 0);
    npy_intp arr_stride = PyArray_STRIDE(arr, 0) / PyArray_ITEMSIZE(arr);
    indices.reserve(arr_len);
    if (PyArray_TYPE(arr) == NPY_BOOL) {
      bool* arr_data = (bool*)PyArray_DATA(arr);
      for (npy_intp i = 0; i < arr_len; i++)
        if (*(arr_data + i * arr_stride)) indices.push_back((npy_intp)i);
      return true;
    } else {
      if (PyArray_TYPE(arr) == NPY_INT64) {
        npy_int64* ptr = (npy_int64*)PyArray_DATA(arr);
        for (npy_intp i = 0; i < arr_len; i++)
          indices.push_back(*(ptr + i * arr_stride));
      } else if (PyArray_TYPE(arr) == NPY_INT32) {
        npy_int32* ptr = (npy_int32*)PyArray_DATA(arr);
        for (npy_intp i = 0; i < arr_len; i++)
          indices.push_back(*(ptr + i * arr_stride));
      } else if (PyArray_TYPE(arr) == NPY_UINT64) {
        npy_uint64* ptr = (npy_uint64*)PyArray_DATA(arr);
        for (npy_intp i = 0; i < arr_len; i++)
          indices.push_back(*(ptr + i * arr_stride));
      } else if (PyArray_TYPE(arr) == NPY_UINT32) {
        npy_uint32* ptr = (npy_uint32*)PyArray_DATA(arr);
        for (npy_intp i = 0; i < arr_len; i++)
          indices.push_back(*(ptr + i * arr_stride));
      } else {
        PyErr_Format(PyExc_TypeError,
                     "On %d-th item, "
                     "numpy array of %s's unsupported for indexing.",
                     (int)item_idx,
                     Py_TYPE(PyArray_DESCR(arr)->typeobj)->tp_name);
        return false;
      }
      return true;
    }
  } else {
    npy_intp number;
    if (!extract_integer(number, obj, item_idx)) {
      PyErr_Format(PyExc_TypeError,
                   "On %d-th item, %s object unusable for indexing.",
                   (int)item_idx, obj->ob_type->tp_name);
      return false;
    }
    indices.push_back(number);
    return true;
  }
}

bool check_indices(const vector<npy_intp>& indices, const npy_intp max_index,
                   const npy_intp item_index) {
  for (size_t i = 0; i < indices.size(); i++) {
    if (indices[i] < -max_index || indices[i] >= max_index) {
      PyErr_Format(PyExc_IndexError,
                   "Out of bounds [0,%d) index value %d (item %d, index %d)",
                   (int)max_index, (int)indices[i], (int)item_index, (int)i);
      return false;
    }
  }
  return true;
}

// In Python 3.2+, type of first argument of PySlice_GetIndicesEx changed from
// PySliceObject to PyObject
#if PY_VERSION_HEX >= 0x030200f0
typedef PyObject PySliceObjectT;
#else
typedef PySliceObject PySliceObjectT;
#endif

void slice_rows(ListItem& item_struct, PySliceObject* slice) {
  // apply slice to rows of item_struct
  npy_intp start, stop, step, m_new;
  PySlice_GetIndicesEx((PySliceObjectT*)slice, item_struct.m, &start, &stop,
                       &step, &m_new);
  item_struct.data += start * item_struct.row_stride;
  item_struct.row_stride *= step;
  item_struct.m = m_new;
}

void slice_cols(ListItem& item_struct, PySliceObject* slice) {
  // apply slice to columns of item_struct
  npy_intp start, stop, step, n_new;
  PySlice_GetIndicesEx((PySliceObjectT*)slice, item_struct.n, &start, &stop,
                       &step, &n_new);
  item_struct.data += start * item_struct.col_stride;
  item_struct.col_stride *= step;
  item_struct.n = n_new;
}

void slice_rows_and_cols(ListItem& item_struct, PySliceObject* row_slice,
                         PySliceObject* col_slice) {
  // apply row slice followed by column slice
  slice_rows(item_struct, row_slice);
  slice_cols(item_struct, col_slice);
}

static PyObject* _idx(PyObject* self, PyObject* args) {
  // Unlike indexing, separate slice per list item is unsupported.
  // Hence, only single slice object need be specified
  PyObject* X;
  PyObject* I;
  if (!PyArg_ParseTuple(args, "OO", &X, &I)) {
    PyErr_SetString(PyExc_TypeError, "Failed to parse arguments");
    return NULL;
  }
  // check first argument X
  if (!PyList_Check(X)) {
    PyErr_SetString(PyExc_TypeError, "Requires list as first argument");
    return NULL;
  }
  Py_ssize_t x_len = PyList_Size(X);

  // check second argument I
  if (!PyTuple_Check(I) || PyTuple_GET_SIZE(I) != 2) {
    PyErr_SetString(PyExc_TypeError, "Requires 2-tuple as second argument");
    return NULL;
  }
  PyObject* Ir = PyTuple_GET_ITEM(I, 0);
  PyObject* Ic = PyTuple_GET_ITEM(I, 1);
  PyObject* Y = NULL;
  Py_ssize_t y_len;
  int index_type = 0;
  if (!PySlice_Check(Ir)) {
    if (!PyList_Check(Ir)) {
      PyErr_Format(PyExc_TypeError,
                   "In _idx(X,(Y,Z)), Y must be either list or slice. "
                   "Intead found %s object.",
                   Ir->ob_type->tp_name);
      return NULL;
    }
    Y = Ir;
    y_len = PyList_Size(Y);
    index_type |= 2;
  }
  if (!PySlice_Check(Ic)) {
    if (!PyList_Check(Ic)) {
      PyErr_Format(PyExc_TypeError,
                   "In _idx(X,(Y,Z)), Z must be either list or slice. "
                   "Intead found %s object.",
                   Ic->ob_type->tp_name);
      return NULL;
    }
    Y = Ic;
    y_len = PyList_Size(Y);
    index_type |= 1;
  }
  if (index_type == 3) {
    PyErr_SetString(PyExc_TypeError,
                    "Simultaneous row and column indexing unsupported.");
    return NULL;
  }
  // note: at this point, meaning of index_type is as follows:
  // 0 - slice rows with Ir, slice columns with Ic
  // 1 - slice rows with Ir, index columns with Y
  // 2 - index rows with Y,  slice columns with Ic

  // determine length of output list
  Py_ssize_t out_len;
  if (x_len == 0 || Y != NULL && y_len == 0) {
    PyErr_SetString(PyExc_ValueError, "Requres non-empty lists");
    return NULL;
  }
  if (index_type == 0) {
    out_len = x_len;
  } else {  // index_type == 1 or 2
    if (x_len != y_len && x_len != 1 && y_len != 1) {
      if (index_type == 1)
        PyErr_Format(PyExc_ValueError,
                     "In _idx(X,(Y,Z)), "
                     "len(X) = %d and len(Y) = %d incompatible.",
                     (int)x_len, (int)y_len);
      else  // index_type == 2
        PyErr_Format(PyExc_ValueError,
                     "In _idx(X,(Y,Z)), "
                     "len(X) = %d and len(Z) = %d incompatible.",
                     (int)x_len, (int)y_len);
      return NULL;
    }
    out_len = x_len > y_len ? x_len : y_len;
  }

  // read X[0] once and use for all y in Y if x_len == 1
  ListItem x_struct;
  if (x_len == 1) {
    PyObject* x_item = PyList_GetItem(X, 0);
    if (!extract_array(x_struct, x_item, 0, "indexee ")) {
      return NULL;
    }
    if (index_type == 0)
      slice_rows_and_cols(x_struct, (PySliceObject*)Ir, (PySliceObject*)Ic);
    else if (index_type == 1)
      slice_rows(x_struct, (PySliceObject*)Ir);
    else  // index_type == 2
      slice_cols(x_struct, (PySliceObject*)Ic);
  }

  // read Y[0] once and use for all x in X if y_len == 1
  vector<npy_intp> indices;
  if (Y != NULL && y_len == 1) {
    PyObject* y_item = PyList_GetItem(Y, 0);
    if (!extract_indices(indices, y_item, 0)) return NULL;
  }

  PyObject* out_list = PyList_New(out_len);
  if (index_type == 0) {
    for (Py_ssize_t i = 0; i < out_len; i++) {
      if (x_len != 1) {
        PyObject* x_item = PyList_GetItem(X, i);
        if (!extract_array(x_struct, PyList_GetItem(X, i), i, "indexee "))
          return NULL;
        slice_rows_and_cols(x_struct, (PySliceObject*)Ir, (PySliceObject*)Ic);
      }

      npy_intp out_dims[2] = {x_struct.m, x_struct.n};
      PyArrayObject* out_item =
          (PyArrayObject*)PyArray_EMPTY(2, out_dims, NPY_FLOAT32, 0);
      const float* in_data = x_struct.data;
      float* out_data = (float*)PyArray_DATA(out_item);
      for (npy_intp r = 0; r < x_struct.m; r++) {
        for (npy_intp c = 0; c < x_struct.n; c++)
          *(out_data++) = *(in_data + c * x_struct.col_stride);
        in_data += x_struct.row_stride;
      }
      PyList_SetItem(out_list, i, (PyObject*)out_item);
    }
  } else if (index_type == 1) {
    // slice rows, index columns
    for (Py_ssize_t i = 0; i < out_len; i++) {
      if (x_len != 1) {
        if (!extract_array(x_struct, PyList_GetItem(X, i), i, "indexee "))
          return NULL;
        slice_rows(x_struct, (PySliceObject*)Ir);
      }

      if (y_len != 1)
        if (!extract_indices(indices, PyList_GetItem(Y, i), i)) return NULL;

      if (!check_indices(indices, x_struct.n, i)) return NULL;

      npy_intp out_dims[2] = {x_struct.m, (npy_intp)indices.size()};
      PyArrayObject* out_item =
          (PyArrayObject*)PyArray_EMPTY(2, out_dims, NPY_FLOAT32, 0);
      const float* in_data = x_struct.data;
      float* out_data = (float*)PyArray_DATA(out_item);
      for (npy_intp r = 0; r < x_struct.m; r++) {
        for (size_t c = 0; c < indices.size(); c++) {
          npy_intp col_index = indices[c];
          if (col_index < 0) col_index += x_struct.n;
          *(out_data++) = *(in_data + col_index * x_struct.col_stride);
        }
        in_data += x_struct.row_stride;
      }
      PyList_SetItem(out_list, i, (PyObject*)out_item);
    }
  } else if (index_type == 2) {
    // index rows, slice columns
    for (Py_ssize_t i = 0; i < out_len; i++) {
      if (x_len != 1) {
        if (!extract_array(x_struct, PyList_GetItem(X, i), i, "indexee "))
          return NULL;
        slice_cols(x_struct, (PySliceObject*)Ic);
      }

      if (y_len != 1)
        if (!extract_indices(indices, PyList_GetItem(Y, i), i)) return NULL;

      if (!check_indices(indices, x_struct.m, i)) return NULL;

      npy_intp out_dims[2] = {(npy_intp)indices.size(), x_struct.n};
      PyArrayObject* out_item =
          (PyArrayObject*)PyArray_EMPTY(2, out_dims, NPY_FLOAT32, 0);
      float* out_data = (float*)PyArray_DATA(out_item);
      for (size_t r = 0; r < indices.size(); r++) {
        npy_intp row_index = indices[r];
        if (row_index < 0) row_index += x_struct.m;
        const float* in_data = x_struct.data + row_index * x_struct.row_stride;
        for (npy_intp c = 0; c < x_struct.n; c++)
          *(out_data++) = *(in_data + c * x_struct.col_stride);
      }
      PyList_SetItem(out_list, i, (PyObject*)out_item);
    }
  }
  return out_list;
}

static PyObject* _seigh(PyObject* self, PyObject* args) {
  PyObject* X;
  if (!PyArg_ParseTuple(args, "O", &X)) {
    PyErr_SetString(PyExc_TypeError, "Failed to parse arguments");
    return NULL;
  }
  ListItem x_struct;
  if (!extract_array(x_struct, X, 0)) return NULL;
  if (x_struct.m != x_struct.n) {
    PyErr_Format(PyExc_ValueError, "Array not square. %d x %d", (int)x_struct.m,
                 (int)x_struct.n);
    return NULL;
  }
  MapTypeConst mat_x(x_struct.data, x_struct.m, x_struct.n,
                     StrideType(x_struct.row_stride, x_struct.col_stride));
  SelfAdjointEigenSolver<MatrixType> solver(mat_x);
  PyObject* out_tuple = PyTuple_New(2);
  npy_intp out_dims[2] = {x_struct.m, x_struct.n};

  // store eigenvalues in first tuple slot
  PyArrayObject* eigvals =
      (PyArrayObject*)PyArray_EMPTY(1, out_dims, NPY_FLOAT32, 0);
  float* eigval_data = (float*)PyArray_DATA(eigvals);
  for (npy_intp j = 0; j < out_dims[0]; j++)
    *(eigval_data++) = solver.eigenvalues()[j];
  PyTuple_SET_ITEM(out_tuple, 0, (PyObject*)eigvals);

  // store eigenvectors in second tuple slot
  PyArrayObject* eigvecs =
      (PyArrayObject*)PyArray_EMPTY(2, out_dims, NPY_FLOAT32, 0);
  float* eigvec_data = (float*)PyArray_DATA(eigvecs);
  copy(&solver.eigenvectors()(0, 0),
       &solver.eigenvectors()(out_dims[0] - 1, out_dims[1] - 1) + 1,
       eigvec_data);
  PyTuple_SET_ITEM(out_tuple, 1, (PyObject*)eigvecs);

  return (PyObject*)out_tuple;
}

static PyObject* _eigh(PyObject* self, PyObject* args) {
  PyObject* X;
  if (!PyArg_ParseTuple(args, "O", &X)) {
    PyErr_SetString(PyExc_TypeError, "Failed to parse arguments");
    return NULL;
  }
  if (!PyList_Check(X)) {
    PyErr_SetString(PyExc_TypeError, "Requires list as first argument");
    return NULL;
  }
  Py_ssize_t x_len = PyList_Size(X);
  if (x_len == 0) {
    PyErr_SetString(PyExc_ValueError, "Requres non-empty lists");
    return NULL;
  }
  Py_ssize_t out_len = x_len;
  PyObject* out_list = PyList_New(out_len);
  for (Py_ssize_t i = 0; i < out_len; i++) {
    ListItem x_struct;
    if (!extract_array(x_struct, PyList_GetItem(X, i), i, "indexee "))
      return NULL;
    if (x_struct.m != x_struct.n) {
      PyErr_Format(PyExc_ValueError, "Array not square. %d x %d. Item %d.",
                   (int)x_struct.m, (int)x_struct.n, (int)i);
      return NULL;
    }
    MapTypeConst mat_x(x_struct.data, x_struct.m, x_struct.n,
                       StrideType(x_struct.row_stride, x_struct.col_stride));
    SelfAdjointEigenSolver<MatrixType> solver(mat_x);
    PyObject* out_tuple = PyTuple_New(2);
    npy_intp out_dims[2] = {x_struct.m, x_struct.n};

    // store eigenvalues in first tuple slot
    PyArrayObject* eigvals =
        (PyArrayObject*)PyArray_EMPTY(1, out_dims, NPY_FLOAT32, 0);
    float* eigval_data = (float*)PyArray_DATA(eigvals);
    for (npy_intp j = 0; j < out_dims[0]; j++)
      *(eigval_data++) = solver.eigenvalues()[j];
    PyTuple_SET_ITEM(out_tuple, 0, (PyObject*)eigvals);

    // store eigenvectors in second tuple slot
    PyArrayObject* eigvecs =
        (PyArrayObject*)PyArray_EMPTY(2, out_dims, NPY_FLOAT32, 0);
    float* eigvec_data = (float*)PyArray_DATA(eigvecs);
    copy(&solver.eigenvectors()(0, 0),
         &solver.eigenvectors()(out_dims[0] - 1, out_dims[1] - 1) + 1,
         eigvec_data);
    PyTuple_SET_ITEM(out_tuple, 1, (PyObject*)eigvecs);

    PyList_SetItem(out_list, i, out_tuple);
  }
  return out_list;
}

static PyMethodDef method_table[] = {
    // functions instantiated from the binary_op template
    {"_add", _add, METH_VARARGS, "[x1+y1,...,xn+yn]<-[x1,...,xn]+[y1,...yn]"},
    {"_sub", _sub, METH_VARARGS, "[x1-y1,...,xn-yn]<-[x1,...,xn]-[y1,...yn]"},
    {"_mul", _mul, METH_VARARGS, "[x1*y1,...,xn*yn]<-[x1,...,xn]*[y1,...yn]"},
    {"_div", _div, METH_VARARGS, "[x1/y1,...,xn/yn]<-[x1,...,xn]/[y1,...yn]"},
    // functions instantiated from the reduction_op template
    {"_sum", (PyCFunction)_sum, METH_VARARGS | METH_KEYWORDS,
     "[sum(x1),...,sum(xn)]<-sum([x1,...,xn])"},
    {"_prod", (PyCFunction)_prod, METH_VARARGS | METH_KEYWORDS, "product" },
    {"_min", (PyCFunction)_min, METH_VARARGS | METH_KEYWORDS, "min" },
    {"_max", (PyCFunction)_max, METH_VARARGS | METH_KEYWORDS, "max" },
    {"_all", (PyCFunction)_all, METH_VARARGS | METH_KEYWORDS, "all" },
    {"_any", (PyCFunction)_any, METH_VARARGS | METH_KEYWORDS, "any" },
    {"_mean", (PyCFunction)_mean, METH_VARARGS | METH_KEYWORDS,
     "[mean(x1),...,mean(xn)]<-mean([x1,...,xn])"},
    {"_argmin", (PyCFunction)_argmin, METH_VARARGS | METH_KEYWORDS,
     "minimizer index" },
    {"_argmax", (PyCFunction)_argmax, METH_VARARGS | METH_KEYWORDS,
     "maximizer index" },
    // not yet implemented
    {"_abs", _abs, METH_VARARGS, "absolute value"},
    {"_dot", _dot, METH_VARARGS, "matrix multiply"},
    {"_transpose", _transpose, METH_VARARGS, "[x1.T,...,xn.T]<-[x1,...,xn].T"},
    {"_idx", _idx, METH_VARARGS, "array indexing"},
    {"_eigh", _eigh, METH_VARARGS, "eigenvalues, eigenvectors"},
    // functions instantiated from binary_op_single
    {"_sadd", _sadd, METH_VARARGS, "[x1+y1,...,xn+yn]<-[x1,...,xn]+[y1,...yn]"},
    {"_ssub", _ssub, METH_VARARGS, "[x1-y1,...,xn-yn]<-[x1,...,xn]-[y1,...yn]"},
    {"_smul", _smul, METH_VARARGS, "[x1*y1,...,xn*yn]<-[x1,...,xn]*[y1,...yn]"},
    {"_sdiv", _sdiv, METH_VARARGS, "[x1/y1,...,xn/yn]<-[x1,...,xn]/[y1,...yn]"},

    // functions instantiated from reduction_op_single
    {"_ssum", (PyCFunction)_ssum, METH_VARARGS | METH_KEYWORDS, "single sum" },
    {"_sprod", (PyCFunction)_sprod, METH_VARARGS | METH_KEYWORDS,
     "single prod" },
    {"_smin", (PyCFunction)_smin, METH_VARARGS | METH_KEYWORDS, "single min" },
    {"_smax", (PyCFunction)_smax, METH_VARARGS | METH_KEYWORDS, "single max" },
    {"_sall", (PyCFunction)_sall, METH_VARARGS | METH_KEYWORDS, "single all" },
    {"_sany", (PyCFunction)_sany, METH_VARARGS | METH_KEYWORDS, "single any" },
    {"_smean", (PyCFunction)_smean, METH_VARARGS | METH_KEYWORDS,
     "single mean" },
    {"_sargmin", (PyCFunction)_sargmin, METH_VARARGS | METH_KEYWORDS,
     "single argmin" },
    {"_sargmax", (PyCFunction)_sargmax, METH_VARARGS | METH_KEYWORDS,
     "single argmax" },
    {"_sdot", _sdot, METH_VARARGS, "matrix multiply"},
    {"_stranspose", _stranspose, METH_VARARGS,
     "[x1.T,...,xn.T]<-[x1,...,xn].T"},
    {"_seigh", _seigh, METH_VARARGS, "single (eigenvalues, eigenvectors)"},
    {NULL, NULL, 0, NULL}};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef module_def = {PyModuleDef_HEAD_INIT,
                                        "vfuncs",
                                        NULL,
                                        -1,
                                        method_table,
                                        NULL,
                                        NULL,
                                        NULL,
                                        NULL};

PyMODINIT_FUNC PyInit_vfuncs(void) {
  PyObject* module = PyModule_Create(&module_def);
#else
PyMODINIT_FUNC initvfuncs(void) {
  Py_InitModule("vfuncs", method_table);
#endif

  import_array();

#if PY_MAJOR_VERSION >= 3
  return module;
#endif
}
