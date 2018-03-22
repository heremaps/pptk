#include <omp.h>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <iostream>
#include <limits>
#include <vector>
#include "Python.h"
#include "arrayobject.h"
#include "kdtree.h"
#include "progress_bar.h"
#include "python_util.h"

using namespace Eigen;
using namespace std;

template <typename T>
void estimate_normals(vector<T>* eigenvectors, vector<T>* eigenvalues,
                      const vector<T>& points, const std::size_t k,
                      T d_max = numeric_limits<T>::infinity(),
                      int num_eigen = 1, bool verbose = true) {
  size_t num_points = points.size() / 3;
  Map<const Matrix<T, Dynamic, 3, RowMajor> > P(&points[0], num_points, 3);

  // organize points into k-d tree
  pointkd::KdTree<T, 3> tree(points);

  // pre-allocate space for results (assume num_eigen either 1 or 3)
  if (eigenvectors) eigenvectors->resize(num_points * num_eigen * 3);
  if (eigenvalues) eigenvalues->resize(num_points * num_eigen);

  int num_procs = omp_get_num_procs();
  omp_set_num_threads(num_procs);
  cout << "Estimating normals with " << num_procs << " threads." << endl;

  ProgressBar<int> bar((int)num_points);

#pragma omp parallel for schedule(static, 1000)
  for (int i = 0; i < (int)num_points; i++) {
    if (verbose && omp_get_thread_num() == 0 && i % 1000 == 0) {
      bar.update(i);
      cout << "\r" << bar.get_string();
    }

    // not using KNearestNeighborsSelf, because we want to include the
    // current point in the normal estimation calculation
    pointkd::Indices indices;
    vector<T> q(P.data() + i * 3, P.data() + (i + 1) * 3);
    tree.KNearestNeighbors(indices, &q[0], k, d_max);

    Matrix<T, Dynamic, 3, RowMajor> X(indices.size(), 3);
    for (size_t j = 0; j < indices.size(); j++) X.row(j) = P.row(indices[j]);
    X.rowwise() -= X.colwise().mean();
    Matrix<T, 3, 3> C = X.transpose() * X;
    SelfAdjointEigenSolver<Matrix<T, 3, 3> > es(C);

    // record results of PCA
    if (eigenvectors) {
      if (num_eigen == 1) {
        Map<Matrix<T, 3, 1> > temp(&(*eigenvectors)[i * 3]);
        temp = es.eigenvectors().col(0);
      } else {  // num_eigen == 3
        Map<Matrix<T, 3, 3> > temp(&(*eigenvectors)[i * 9]);
        temp = es.eigenvectors();
      }
    }
    if (eigenvalues) {
      if (num_eigen == 1) {
        (*eigenvalues)[i] = es.eigenvalues()(0);
      } else {  // num_eigen == 1
        Map<Matrix<T, 3, 1> > temp(&(*eigenvalues)[i * 3]);
        temp = es.eigenvalues();
      }
    }
  }

  if (verbose) {
    bar.update((int)num_points);
    cout << "\r" << bar.get_string() << endl;
  }
}

template <typename T>
struct NumpyTypeNumber {};

template <>
struct NumpyTypeNumber<float> {
  static const int value = NPY_FLOAT32;
};

template <>
struct NumpyTypeNumber<double> {
  static const int value = NPY_FLOAT64;
};

template <typename T>
void estimate_normals(PyObject*& out1, PyObject*& out2, const Array2D& arr,
                      int k, float r, bool output_eigenvalues,
                      bool output_all_eigenvectors, bool verbose) {
  vector<T> points;
  VectorFromArray2D(points, arr);

  int num_eigen = 1;
  int out1_ndim = 2;
  int out2_ndim = 1;
  npy_intp out1_dims[3] = {arr.m, 3, -1};
  npy_intp out2_dims[2] = {arr.m, -1};
  if (output_all_eigenvectors) {
    num_eigen = 3;
    out1_ndim = 3;
    out2_ndim = 2;
    out1_dims[2] = 3;
    out2_dims[1] = 3;
  }

  int typenum = NumpyTypeNumber<T>::value;

  out1 = NULL;
  out2 = NULL;
  vector<T> evecs;
  vector<T> evals;
  estimate_normals<T>(&evecs, output_eigenvalues == 1 ? &evals : NULL, points,
                      k, r, num_eigen, (bool)verbose);
  out1 = PyArray_EMPTY(out1_ndim, out1_dims, typenum, false);
  copy(evecs.begin(), evecs.end(), (T*)PyArray_DATA((PyArrayObject*)out1));
  if (output_eigenvalues) {
    out2 = PyArray_EMPTY(out2_ndim, out2_dims, typenum, false);
    copy(evals.begin(), evals.end(), (T*)PyArray_DATA((PyArrayObject*)out2));
  }
}

static char estimate_normals_usage[] =
    "Estimates normals at all points using principal component analysis "
    "(PCA).\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "points : 3-column numpy array of type float32\n"
    "    Input point cloud.\n"
    "k : int\n"
    "    Number of neighbors to use in PCA.\n"
    "r : float\n"
    "    Use neighbors within r of query point.\n"
    "output_eigenvalues : bool, optional (default: False)\n"
    "output_all_eigenvectors : bool, optional (default: False)\n"
    "verbose : bool (default: True)\n"
    "\n"
    "Returns\n"
    "-------\n"
    "\n";

static PyObject* estimate_normals_wrapper(PyObject* self, PyObject* args,
                                          PyObject* kwargs) {
  PyObject* p = NULL;
  int k;
  float r;
  int output_eigenvalues = 0;
  int output_all_eigenvectors = 0;
  int verbose = 1;
  static char* keywords[] = {
      "points", "k", "r", "output_eigenvalues", "output_all_eigenvectors",
      "verbose", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oif|iii", keywords, &p, &k,
                                   &r, &output_eigenvalues,
                                   &output_all_eigenvectors, &verbose)) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to parse inputs");
    return NULL;
  }

  Array2D arr;
  if (!CheckAndExtractArray2D(arr, p)) {
    if (!PyErr_Occurred())
      PyErr_SetString(PyExc_TypeError, "points must be 2-d array");
    return NULL;
  }

  PyObject* out1 = NULL;
  PyObject* out2 = NULL;
  if (arr.type_num == NPY_FLOAT32) {
    estimate_normals<float>(out1, out2, arr, k, r, (bool)output_eigenvalues,
                            (bool)output_all_eigenvectors, (bool)verbose);
  } else if (arr.type_num == NPY_FLOAT64) {
    estimate_normals<double>(out1, out2, arr, k, r, (bool)output_eigenvalues,
                             (bool)output_all_eigenvectors, (bool)verbose);
  } else {
    PyErr_SetString(PyExc_TypeError, "points must be float32 or float64");
    return NULL;
  }

  if (out2 == NULL) {
    return out1;
  } else {
    PyObject* out = PyTuple_New(2);
    PyTuple_SetItem(out, 0, out1);
    PyTuple_SetItem(out, 1, out2);
    return out;
  }
}

static PyMethodDef methods[] = {
    {"estimate_normals", (PyCFunction)estimate_normals_wrapper,
     METH_VARARGS | METH_KEYWORDS, estimate_normals_usage},
    {NULL, NULL, 0, NULL}};

PyMODINIT_FUNC initestimate_normals(void) {
  (void)Py_InitModule("estimate_normals", methods);
  import_array();
}
