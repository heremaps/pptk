#ifndef __PYTHON_UTIL_H__
#define __PYTHON_UTIL_H__

#include <cstdint>
#include <vector>
#include "Python.h"
#include "arrayobject.h"

struct Array2D {
  const unsigned char* data;
  std::vector<std::uint8_t> scalar;
  npy_intp m;           // number of rows
  npy_intp n;           // number of columns
  npy_intp row_stride;  // in bytes
  npy_intp col_stride;  // in bytes
  npy_intp item_size;   // in bytes
  int type_num;
  // see https://docs.scipy.org/doc/numpy/reference/c-api.dtype.html
};

bool IsIntegral(int type_num) {
  return type_num == NPY_INT64 || type_num == NPY_INT32 ||
         type_num == NPY_INT16 || type_num == NPY_INT8 ||

         type_num == NPY_UINT64 || type_num == NPY_UINT32 ||
         type_num == NPY_UINT16 || type_num == NPY_UINT8 ||

         type_num == NPY_BYTE || type_num == NPY_SHORT || type_num == NPY_INT ||
         type_num == NPY_LONG || type_num == NPY_LONGLONG ||

         type_num == NPY_UBYTE || type_num == NPY_USHORT ||
         type_num == NPY_UINT || type_num == NPY_ULONG ||
         type_num == NPY_ULONGLONG;
}

bool IsContiguous(const Array2D& x) {
  return (x.n <= 1 || x.item_size == x.col_stride) &&
         (x.m <= 1 || x.n * x.item_size == x.row_stride);
}

/**
  * Assumes x.scalar and x.type_num has been filled.
*/
void InitScalarStruct(Array2D& x) {
  x.data = (const unsigned char*)&x.scalar[0];
  x.m = x.n = 1;
  x.row_stride = x.col_stride = 0;
  x.item_size = x.scalar.size();
}

/**
  * TODO: multiple macros may have the same int value
*/
std::string TypeNameFromTypeNum(int type_num) {
  if (type_num == NPY_FLOAT || type_num == NPY_FLOAT32)
    return "NPY_FLOAT/NPY_FLOAT32";
  else if (type_num == NPY_DOUBLE || type_num == NPY_FLOAT64)
    return "NPY_DOUBLE/NPY_FLOAT64";
  else if (type_num == NPY_HALF || type_num == NPY_FLOAT16)
    return "NPY_HALF/NPY_FLOAT16";
  else if (type_num == NPY_INT || type_num == NPY_INT32)
    return "NPY_INT/NPY_INT32";
  else if (type_num == NPY_LONG)
    return "NPY_LONG";
  else if (type_num == NPY_LONGLONG || type_num == NPY_INT64)
    return "NPY_LONGLONG/NPY_INT64";
  else if (type_num == NPY_BYTE || type_num == NPY_INT8)
    return "NPY_BYTE/NPY_INT8";
  else if (type_num == NPY_SHORT || type_num == NPY_INT16)
    return "NPY_SHORT/NPY_INT16";
  else if (type_num == NPY_UINT || type_num == NPY_UINT32)
    return "NPY_UINT/NPY_UINT32";
  else if (type_num == NPY_ULONG)
    return "NPY_ULONG";
  else if (type_num == NPY_ULONGLONG || type_num == NPY_UINT64)
    return "NPY_ULONGLONG/NPY_UINT64";
  else if (type_num == NPY_UBYTE || type_num == NPY_UINT8)
    return "NPY_UBYTE/NPY_UINT8";
  else if (type_num == NPY_USHORT || type_num == NPY_UINT16)
    return "NPY_USHORT/NPY_UINT16";
  else if (type_num == NPY_BOOL)
    return "NPY_BOOL";
  else if (type_num == NPY_CFLOAT || type_num == NPY_COMPLEX64)
    return "NPY_CFLOAT/NPY_COMPLEX64";
  else if (type_num == NPY_CDOUBLE || type_num == NPY_COMPLEX128)
    return "NPY_CDOUBLE/NPY_COMPLEX128";
  else if (type_num == NPY_DATETIME)
    return "NPY_DATETIME";
  else if (type_num == NPY_TIMEDELTA)
    return "NPY_TIMEDELTA";
  else if (type_num == NPY_STRING)
    return "NPY_STRING";
  else if (type_num == NPY_UNICODE)
    return "NPY_UNICODE";
  else if (type_num == NPY_OBJECT)
    return "NPY_OBJECT";
  else if (type_num == NPY_VOID)
    return "NPY_VOID";
  else
    return "???";
}

/**
  * This template is used to enforce uniform behavior across all
  * CheckAndExtract* functions for extracting scalar values.  Function pointer
  * Check ensures obj has a certain type.  Function pointer Extract proceeds
  * with scalar extraction assuming obj has the "right" type.
*/
template <bool (*Check)(PyObject*),
          void (*Extract)(std::vector<std::uint8_t>&, int&, PyObject*)>
bool CheckAndExtractScalar_(std::vector<std::uint8_t>& scalar, int& type_num,
                            PyObject* obj) {
  if (Check(obj)) {
    Extract(scalar, type_num, obj);
    if (PyErr_Occurred())
      return false;
    else
      return true;
  }
  return false;
}

/**
  * Wraps macro PyArray_Check into an actual function
*/
bool CheckPyArray(PyObject* obj) { return PyArray_Check(obj); }

/**
  * Wraps macro PyArray_IsScalar into an actual function
*/
bool CheckArrayScalar(PyObject* obj) { return PyArray_IsScalar(obj, Generic); }

/**
  * Wraps macro PyFloat_CheckExact into an actual function
*/
bool CheckPyFloat(PyObject* obj) { return PyFloat_CheckExact(obj); }

#if PY_MAJOR_VERSION < 3
/**
  * Wraps macro PyInt_CheckExact into an actual function
*/
bool CheckPyInt(PyObject* obj) { return PyInt_CheckExact(obj); }
#endif

/**
  * Wraps macro PyBool_Check into an actual function
*/
bool CheckPyBool(PyObject* obj) { return PyBool_Check(obj); }

/**
  * Wraps macro PyLong_CheckExact into an actual function
*/
bool CheckPyLong(PyObject* obj) { return PyLong_CheckExact(obj); }

/**
  * Assumes obj is an array scalar (i.e. PyArray_IsScalar(obj, Generic) == true)
  * E.g. np.r_[1][0]
*/
void ExtractScalarFromArrayScalar(std::vector<std::uint8_t>& scalar,
                                  int& type_num, PyObject* obj) {
  PyArray_Descr* descr = PyArray_DescrFromScalar(obj);
  if (descr->type_num == NPY_STRING || descr->type_num == NPY_UNICODE ||
      descr->type_num == NPY_VOID || descr->type_num == NPY_CFLOAT ||
      descr->type_num == NPY_CDOUBLE || descr->type_num == NPY_CLONGDOUBLE ||
      descr->type_num == NPY_DATETIME || descr->type_num == NPY_TIMEDELTA ||
      descr->type_num == NPY_OBJECT) {
    PyErr_Format(PyExc_ValueError,
                 "ExtractScalarFromArrayScalar(): "
                 "array-scalar type_num = %d (%s) not supported",
                 descr->type_num, TypeNameFromTypeNum(descr->type_num).c_str());
  } else {
    type_num = descr->type_num;
    scalar.resize(descr->elsize);
    PyArray_ScalarAsCtype(obj, &scalar[0]);
  }
  Py_DECREF(descr);
}

/**
  * Assumes obj is a 0-d array (i.e. PyArray_IsZeroDim(obj) == true)
  * E.g. np.ones([])
*/
void ExtractScalarFromNumpy0DArray(std::vector<std::uint8_t>& scalar,
                                   int& type_num, PyObject* obj) {
  PyObject* temp = PyArray_NewCopy((PyArrayObject*)obj, NPY_ANYORDER);
  temp = PyArray_Return((PyArrayObject*)temp);  // steals reference to temp
                                                // temp is now an array scalar
  ExtractScalarFromArrayScalar(scalar, type_num, temp);
  Py_DECREF(temp);
}

/**
  * Assumes obj is a Python float (i.e. PyFloat_CheckExact(obj) == true)
*/
void ExtractScalarFromPyFloat(std::vector<std::uint8_t>& scalar, int& type_num,
                              PyObject* obj) {
  type_num = NPY_DOUBLE;
  scalar.resize(sizeof(double));
  *(double*)&scalar[0] = PyFloat_AS_DOUBLE(obj);
}

#if PY_MAJOR_VERSION < 3
/**
  * Assumes obj is a Python int (i.e. PyInt_CheckExact(obj) == true)
*/
void ExtractScalarFromPyInt(std::vector<std::uint8_t>& scalar, int& type_num,
                            PyObject* obj) {
  type_num = NPY_LONG;
  scalar.resize(sizeof(long));
  *(long*)&scalar[0] = PyInt_AS_LONG(obj);
}
#endif

/**
  * Assumes obj is a Python long (i.e. PyLong_CheckExact(obj) == true)
  * Raises OverflowError if long represented by obj is larger than Py_LLONG_MAX
*/
void ExtractScalarFromPyLong(std::vector<std::uint8_t>& scalar, int& type_num,
                             PyObject* obj) {
  int overflow;
  PY_LONG_LONG temp = PyLong_AsLongLongAndOverflow(obj, &overflow);
  if (PyErr_Occurred()) {  // possibly TypeError or MemoryError
    return;
  } else if (overflow != 0) {
    PyErr_SetString(PyExc_OverflowError,
                    "ExtractScalarFromPyLong(): "
                    "Python long integer too large.");
  } else {
    type_num = NPY_LONGLONG;
    scalar.resize(sizeof(PY_LONG_LONG));
    *(PY_LONG_LONG*)&scalar[0] = temp;
  }
}

/**
  * Assumes obj is a Python bool (i.e. PyBool_CheckExact(obj) == true)
*/
void ExtractScalarFromPyBool(std::vector<std::uint8_t>& scalar, int& type_num,
                             PyObject* obj) {
  type_num = NPY_BOOL;
  scalar.resize(sizeof(bool));
  if (obj == Py_True)
    *(bool*)&scalar[0] = true;
  else
    *(bool*)&scalar[0] = false;
}

/**
  * Assumes obj is a PyArrayObject and populates fields in x.
  * Raises ValueError if obj has ndim neither 0, 1 nor 2
*/
void ExtractArray2DFromPyArray(Array2D& x, PyObject* obj) {
  PyArrayObject* arr = (PyArrayObject*)obj;
  int ndim = PyArray_NDIM(arr);
  if (ndim < 0 || ndim > 2) {
    PyErr_Format(PyExc_ValueError,
                 "ExtractArray2DFromPyArray(): "
                 "only handles ndim of 0, 1 or 2; encountered ndim=%d",
                 ndim);
    return;
  }

  if (ndim == 0) {
    ExtractScalarFromNumpy0DArray(x.scalar, x.type_num, obj);
    InitScalarStruct(x);
    return;
  }

  if (ndim == 1) {
    x.m = 1;
    x.n = PyArray_DIM(arr, 0);
    x.row_stride = 0;
    x.col_stride = PyArray_STRIDE(arr, 0);
  } else {
    x.m = PyArray_DIM(arr, 0);
    x.n = PyArray_DIM(arr, 1);
    x.row_stride = PyArray_STRIDE(arr, 0);
    x.col_stride = PyArray_STRIDE(arr, 1);
  }
  if (x.m == 1) x.row_stride = 0;
  if (x.n == 1) x.col_stride = 0;
  x.data = (const unsigned char*)PyArray_DATA(arr);
  x.type_num = PyArray_TYPE(arr);
  x.item_size = PyArray_ITEMSIZE(arr);
}

bool CheckAndExtractScalar(std::vector<uint8_t>& scalar, int& type_num,
                           PyObject* obj) {
  if (CheckAndExtractScalar_<CheckArrayScalar, ExtractScalarFromArrayScalar>(
          scalar, type_num, obj)) {
    return true;
  } else if (PyErr_Occurred())
    return false;

  if (CheckAndExtractScalar_<CheckPyFloat, ExtractScalarFromPyFloat>(
          scalar, type_num, obj)) {
    return true;
  } else if (PyErr_Occurred())
    return false;

#if PY_MAJOR_VERSION < 3
  if (CheckAndExtractScalar_<CheckPyInt, ExtractScalarFromPyInt>(
          scalar, type_num, obj)) {
    return true;
  } else if (PyErr_Occurred())
    return false;
#endif

  if (CheckAndExtractScalar_<CheckPyBool, ExtractScalarFromPyBool>(
          scalar, type_num, obj)) {
    return true;
  } else if (PyErr_Occurred())
    return false;

  if (CheckAndExtractScalar_<CheckPyLong, ExtractScalarFromPyLong>(
          scalar, type_num, obj)) {
    return true;
  } else if (PyErr_Occurred())
    return false;

  return false;
}

/**
  * Interpret obj as an Array2D struct.
  * Supports obj's having one of the following types:
  *   - Numpy arrays (raises ValueError if dim is neither 0, 1 or 2)
  *   - Numpy array scalars
  *   - Python float, int, bool or long
  * Returns false if obj is none of the above, or if obj is one of the above
  * but an exception was raised (e.g. OverflowError is raised if obj is a Python
  * long representing an integer greater than Py_LLONG_MAX.
*/
bool CheckAndExtractArray2D(Array2D& x, PyObject* obj) {
  x.scalar.clear();

  if (PyArray_Check(obj)) {
    ExtractArray2DFromPyArray(x, obj);
    if (PyErr_Occurred())
      return false;
    else
      return true;
  }

  if (CheckAndExtractScalar(x.scalar, x.type_num, obj)) {
    InitScalarStruct(x);
    return true;
  } else if (PyErr_Occurred())
    return false;

  return false;
}

/**
  * Cast Python object as long
  * Equivalent to Python expression int(obj).
  * Returns true on success, and false on failure.
*/
bool CastAsLong(long& number, PyObject* item) {
  PyObject* number_obj = PyNumber_Long(item);
  if (number_obj == NULL) return false;

  number = PyLong_AsLong(number_obj);  // may raise OverflowError
  Py_DECREF(number_obj);
  if (PyErr_Occurred())
    return false;
  else
    return true;
}

/**
  * Cast Python object as long long
  * Equivalent to Python expression int(obj).
  * Returns true on success, and false on failure.
*/
bool CastAsLongLong(long long& number, PyObject* item) {
  PyObject* number_obj = PyNumber_Long(item);
  if (number_obj == NULL) return false;

  number = PyLong_AsLongLong(number_obj);  // may raise OverflowError
  Py_DECREF(number_obj);
  if (PyErr_Occurred())
    return false;
  else
    return true;
}

/**
  * Cast Python object as double.
  * Equivalent to Python expression float(obj).
  * Returns true on success, and false on failure.
*/
bool CastAsDouble(double& number, PyObject* item) {
  PyObject* number_obj = PyNumber_Float(item);
  if (number_obj == NULL) return false;

  number = (float)PyFloat_AS_DOUBLE(number_obj);
  Py_DECREF(number_obj);
  return true;
}

/**
  * Cast Python object as index.
  * Equivalent to Python expression obj.__index__().
  * Returns true on success, and false on failure.
*/
template <typename T>  // assumes T is integral type
bool CastAsIndex(T& index, PyObject* item) {
  PyObject* index_obj = PyNumber_Index(item);
  if (index_obj == NULL) return false;

  long long temp;
  bool success = CastAsLongLong(temp, index_obj);
  index = (T)temp;
  Py_DECREF(index_obj);
  return success;
}

/**
  * Assumes obj is a Python list (i.e. PyList_Check(obj) == true).
  * Raises exception if obj contains an item that cannot be cast as an index.
*/
template <typename T, typename A>
void ExtractIndicesFromPyList(std::vector<T, A>& indices, PyObject* obj) {
  Py_ssize_t len = PyList_Size(obj);
  std::vector<T, A> temp;
  temp.reserve(len);
  for (Py_ssize_t i = 0; i < len; i++) {
    T index;
    PyObject* item = PyList_GetItem(obj, i);
    if (CastAsIndex(index, item)) {
      temp.push_back(index);
    } else if (!PyErr_Occurred()) {
      PyErr_Format(PyExc_TypeError,
                   "ExtractIndicesFromPyList(): "
                   "Encountered non-index %s object at the %d-th item",
                   item->ob_type->tp_name, (int)i);
      return;
    }
  }
  indices.swap(temp);
}

/**
  * Reads numbers of type Ty from data into vector<Tx> v.
*/
template <template <typename Tx, typename A> class V, typename Tx, typename A,
          typename Ty>
void VectorFromArray2D_(V<Tx, A>& v, const Ty* data, npy_intp rows,
                        npy_intp row_stride, npy_intp cols,
                        npy_intp col_stride) {
  v.clear();
  v.reserve(rows * cols);
  const unsigned char* ptr = (const unsigned char*)data;
  for (npy_intp i = 0; i < rows; i++) {
    for (npy_intp j = 0; j < cols; j++) {
      v.push_back((Tx)(*(const Ty*)(ptr + j * col_stride)));
    }
    ptr += row_stride;
  }
}

/**
  * Reads 2-d array x in row major order into vector<T> v.
  * Numbers in x are typecasted to type T before being written into v.
*/
template <template <typename T, typename A> class V, typename T, typename A>
bool VectorFromArray2D(V<T, A>& v, const Array2D& x) {
  if (x.type_num == NPY_FLOAT || x.type_num == NPY_FLOAT32)
    VectorFromArray2D_(v, (const npy_float*)x.data, x.m, x.row_stride, x.n,
                       x.col_stride);
  else if (x.type_num == NPY_DOUBLE || x.type_num == NPY_FLOAT64)
    VectorFromArray2D_(v, (const npy_double*)x.data, x.m, x.row_stride, x.n,
                       x.col_stride);
  else if (x.type_num == NPY_INT || x.type_num == NPY_INT32)
    VectorFromArray2D_(v, (const npy_int*)x.data, x.m, x.row_stride, x.n,
                       x.col_stride);
  else if (x.type_num == NPY_LONG)
    VectorFromArray2D_(v, (const npy_long*)x.data, x.m, x.row_stride, x.n,
                       x.col_stride);
  else if (x.type_num == NPY_LONGLONG || x.type_num == NPY_INT64)
    VectorFromArray2D_(v, (const npy_longlong*)x.data, x.m, x.row_stride, x.n,
                       x.col_stride);
  else if (x.type_num == NPY_BYTE || x.type_num == NPY_INT8)
    VectorFromArray2D_(v, (const npy_byte*)x.data, x.m, x.row_stride, x.n,
                       x.col_stride);
  else if (x.type_num == NPY_SHORT || x.type_num == NPY_INT16)
    VectorFromArray2D_(v, (const npy_short*)x.data, x.m, x.row_stride, x.n,
                       x.col_stride);
  else if (x.type_num == NPY_UINT || x.type_num == NPY_UINT32)
    VectorFromArray2D_(v, (const npy_uint*)x.data, x.m, x.row_stride, x.n,
                       x.col_stride);
  else if (x.type_num == NPY_ULONG)
    VectorFromArray2D_(v, (const npy_ulong*)x.data, x.m, x.row_stride, x.n,
                       x.col_stride);
  else if (x.type_num == NPY_ULONGLONG || x.type_num == NPY_UINT64)
    VectorFromArray2D_(v, (const npy_ulonglong*)x.data, x.m, x.row_stride, x.n,
                       x.col_stride);
  else if (x.type_num == NPY_UBYTE || x.type_num == NPY_UINT8)
    VectorFromArray2D_(v, (const npy_ubyte*)x.data, x.m, x.row_stride, x.n,
                       x.col_stride);
  else if (x.type_num == NPY_USHORT || x.type_num == NPY_UINT16)
    VectorFromArray2D_(v, (const npy_ushort*)x.data, x.m, x.row_stride, x.n,
                       x.col_stride);
  else if (x.type_num == NPY_BOOL)
    VectorFromArray2D_(v, (const npy_bool*)x.data, x.m, x.row_stride, x.n,
                       x.col_stride);
  else {
    PyErr_Format(PyExc_ValueError,
                 "VectorFromArray2D(): type_num = %d (%s) not supported",
                 x.type_num, TypeNameFromTypeNum(x.type_num).c_str());
    return false;
  }
  return true;
}

/**
  * Assumes obj is a Numpy array (i.e. PyArray_Check(obj) == true).
  * Raises ValueError if obj's dim is neither 0, 1 nor 2
  * Indices are extracted from obj in row-major order.
*/
template <typename T, typename A>
void ExtractIndicesFromPyArray(std::vector<T, A>& indices,
                               PyObject* obj, int n) {
  int dim = PyArray_NDIM(obj);
  if (dim > 1)
    PyErr_Format(PyExc_ValueError,
                 "ExtractIndicesFromPyArray(): "
                 "PyArray has dim = %d (expected dim = 0 or 1).",
                 dim);
  Array2D x;
  ExtractArray2DFromPyArray(x, obj);  // we know after this statement m == 1
  if (PyErr_Occurred()) return;
  if (x.type_num == NPY_BOOL) {
    if (x.n != n) {
      PyErr_Format(PyExc_ValueError,
                   "ExtractIndicesFromPyArray(): "
                   "binary indexing mask has incorrect size %d (expected %d).",
                   (int)x.n, n);
    } else {
      for (npy_intp i = 0; i < x.n; i++) {
        if (*(npy_bool*)(x.data + i * x.col_stride))
          indices.push_back((T)i);  // record indices where x is true
      }
    }
  } else if (IsIntegral(x.type_num)) {
    VectorFromArray2D(indices, x);
  } else {
    PyErr_Format(PyExc_ValueError,
                 "ExtractIndicesFromPyArray(): "
                 "encountered unsupported type_num = %d (%s) "
                 "(expected boolean or integral type).",
                 x.type_num, TypeNameFromTypeNum(x.type_num).c_str());
  }
}

template <typename T, typename A>
std::size_t FindInvalidIndex(const std::vector<T, A>& indices,
                             const T max_index) {
  std::size_t i;
  for (i = 0; i < indices.size(); i++) {
    if (indices[i] < -max_index || indices[i] >= max_index) {
      break;
    }
  }
  return i;
}

template <typename T, typename A>
void FixNegativeIndices(std::vector<T, A>& indices, const npy_intp max_index) {
  for (size_t i = 0; i < indices.size(); i++) {
    if (indices[i] < 0) indices[i] += (T)max_index;
  }
}

template <typename T, typename A>
bool CheckAndExtractIndices(std::vector<T, A>& indices, PyObject* obj, int n) {
  Array2D x;
  if (PyList_Check(obj)) {
    ExtractIndicesFromPyList(indices, obj);
    if (PyErr_Occurred()) return false;
  } else if (PyArray_Check(obj)) {
    ExtractIndicesFromPyArray(indices, obj, n);
    if (PyErr_Occurred()) return false;
  } else if (CheckAndExtractScalar(x.scalar, x.type_num, obj)) {
    if (!IsIntegral(x.type_num)) {
      PyErr_Format(PyExc_ValueError,
                   "CheckAndExtractIndices(): "
                   "encountered non-integral scalar type_num = %d (%s).",
                   x.type_num, TypeNameFromTypeNum(x.type_num).c_str());
      return false;
    }
    InitScalarStruct(x);
    VectorFromArray2D(indices, x);
    if (PyErr_Occurred()) return false;
  } else
    return false;

  std::size_t i = FindInvalidIndex(indices, n);
  if (i != indices.size()) {
    PyErr_Format(PyExc_RuntimeError,
                 "CheckAndExtractIndices(): "
                 "%lu-th query index is outside of [%d,%d]",
                 i, -n, n - 1);
    return false;
  } else {  // all indices in [-n, n-1]
    FixNegativeIndices(indices, n);
  }
  return true;
}

#endif
