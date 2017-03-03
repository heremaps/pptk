#ifndef __PYTHON_UTIL_H__
#define __PYTHON_UTIL_H__

#include <vector>
#include "Python.h"
#include "arrayobject.h"

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

template <typename T>
bool extract_indices(std::vector<T>& indices, PyObject* obj,
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
      indices.push_back((T)number);
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
        if (*(arr_data + i * arr_stride)) indices.push_back((T)i);
      return true;
    } else {
      if (PyArray_TYPE(arr) == NPY_INT64) {
        npy_int64* ptr = (npy_int64*)PyArray_DATA(arr);
        for (npy_intp i = 0; i < arr_len; i++)
          indices.push_back((T)(*(ptr + i * arr_stride)));
      } else if (PyArray_TYPE(arr) == NPY_INT32) {
        npy_int32* ptr = (npy_int32*)PyArray_DATA(arr);
        for (npy_intp i = 0; i < arr_len; i++)
          indices.push_back((T)(*(ptr + i * arr_stride)));
      } else if (PyArray_TYPE(arr) == NPY_UINT64) {
        npy_uint64* ptr = (npy_uint64*)PyArray_DATA(arr);
        for (npy_intp i = 0; i < arr_len; i++)
          indices.push_back((T)(*(ptr + i * arr_stride)));
      } else if (PyArray_TYPE(arr) == NPY_UINT32) {
        npy_uint32* ptr = (npy_uint32*)PyArray_DATA(arr);
        for (npy_intp i = 0; i < arr_len; i++)
          indices.push_back((T)(*(ptr + i * arr_stride)));
      } else {
        PyErr_Format(PyExc_TypeError,
                     "On %d-th item, "
                     "numpy array of %s's unsupported for indexing.",
                     (int)item_idx,
                     PyArray_DESCR(arr)->typeobj->ob_type->tp_name);
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
    indices.push_back((T)number);
    return true;
  }
}

template <typename T>
bool check_indices(const std::vector<T>& indices, const npy_intp max_index,
                   const npy_intp item_index) {
  for (size_t i = 0; i < indices.size(); i++) {
    if ((npy_intp)indices[i] < -max_index ||
        (npy_intp)indices[i] >= max_index) {
      PyErr_Format(PyExc_IndexError,
                   "Out of bounds [0,%d) index value %d (item %d, index %d)",
                   (int)max_index, (int)indices[i], (int)item_index, (int)i);
      return false;
    }
  }
  return true;
}

template <typename T>
void fix_negative_indices(std::vector<T>& indices, const npy_intp max_index) {
  for (size_t i = 0; i < indices.size(); i++) {
    if (indices[i] < 0) indices[i] += (T)max_index;
  }
}

#endif