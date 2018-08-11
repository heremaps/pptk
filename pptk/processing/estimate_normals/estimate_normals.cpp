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
#include "timer.h"

using namespace Eigen;
using namespace std;

template <typename T>
void estimate_normals(vector<T>* eigenvectors, vector<T>* eigenvalues,
                      vector<int>* neighborhood_sizes, const vector<T>& points,
                      const std::size_t k, const T d_max,
                      const std::vector<int>* subsample_indices,
                      int num_eigen = 1, bool verbose = true,
                      int num_procs = -1) {
  size_t num_points = points.size() / 3;
  Map<const Matrix<T, Dynamic, 3, RowMajor> > P(&points[0], num_points, 3);

  if (num_procs < 0) num_procs = omp_get_num_procs();
  if (verbose)
    cout << "(estimate_normals) using " << num_procs << " threads" << endl;

  // organize points into k-d tree
  pointkd::BuildParams build_params;
  build_params.num_proc = num_procs;
  double build_time = getTime();
  pointkd::KdTree<T, 3> tree(points, build_params);
  build_time = getTime() - build_time;
  if (verbose) {
    cout << "(estimate_normals) ";
    cout << "k-d tree build time (s): " << build_time << endl;
  }

  int num_normals = num_points;
  if (subsample_indices != NULL)
    num_normals = subsample_indices->size();

  // pre-allocate space for results (assume num_eigen either 1 or 3)
  if (eigenvectors) eigenvectors->resize(num_normals * num_eigen * 3);
  if (eigenvalues) eigenvalues->resize(num_normals * num_eigen);
  if (neighborhood_sizes) neighborhood_sizes->resize(num_normals);

  tbb::task_scheduler_init(1);  // use just 1 thread for k-d tree queries
  omp_set_num_threads(num_procs);

  if (verbose) {
    cout << "(estimate_normals) neighborhood parameters: " << endl;
    cout << "  k = " << k << endl;
    cout << "  r = " << d_max << endl;
  }

  ProgressBar<int> bar((int)num_normals);

  double pca_time = getTime();

#pragma omp parallel for schedule(static, 1000)
  for (int i = 0; i < (int)num_normals; i++) {
    if (verbose && omp_get_thread_num() == 0 && i % 1000 == 0) {
      bar.update(i);
      cout << "\r" << bar.get_string();
    }

    int i_ = i;
    if (subsample_indices != NULL)
      i_ = (*subsample_indices)[i];

    // not using KNearestNeighborsSelf, because we want to include the
    // current point in the normal estimation calculation
    pointkd::Indices indices;
    vector<T> q(P.data() + i_ * 3, P.data() + (i_ + 1) * 3);
    if (k == -1)
      tree.RNearNeighbors(indices, &q[0], d_max);
    else
      tree.KNearestNeighbors(indices, &q[0], k, d_max);

    Matrix<T, Dynamic, 3, RowMajor> X(indices.size(), 3);
    for (size_t j = 0; j < indices.size(); j++) X.row(j) = P.row(indices[j]);
    X.rowwise() -= X.colwise().mean();
    Matrix<T, 3, 3> C = X.transpose() * X;
    C /= (T)indices.size();
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
    if (neighborhood_sizes)
      (*neighborhood_sizes)[i] = indices.size();
  }

  pca_time = getTime() - pca_time;

  if (verbose) {
    bar.update((int)num_normals);
    cout << "\r" << bar.get_string() << endl;
    cout << "(estimate_normals) PCA time (s): " << pca_time << std::endl;
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
void estimate_normals(PyObject*& out1, PyObject*& out2, PyObject*& out3,
                      const Array2D& arr, int k, float r,
                      std::vector<int>* ptr_subsample_indices,
                      bool output_eigenvalues, bool output_all_eigenvectors,
                      bool output_neighborhood_sizes, bool verbose,
                      int num_procs) {
  vector<T> points;
  VectorFromArray2D(points, arr);

  int num_normals = arr.m;
  if (ptr_subsample_indices != NULL)
    num_normals = ptr_subsample_indices->size();

  int num_eigen = 1;
  int out1_ndim = 2;
  int out2_ndim = 1;
  npy_intp out1_dims[3] = {num_normals, 3, -1};
  npy_intp out2_dims[2] = {num_normals, -1};
  if (output_all_eigenvectors) {
    num_eigen = 3;
    out1_ndim = 3;
    out2_ndim = 2;
    out1_dims[2] = 3;
    out2_dims[1] = 3;
  }

  int out3_ndim = 1;
  int out3_typenum = NPY_INT;
  npy_intp out3_dims[2] = {num_normals, -1};

  int typenum = NumpyTypeNumber<T>::value;

  out1 = NULL;
  out2 = NULL;
  out3 = NULL;
  vector<T> evecs;
  vector<T> evals;
  vector<int> nbhd_sizes;
  estimate_normals<T>(&evecs, output_eigenvalues == 1 ? &evals : NULL,
                      output_neighborhood_sizes == 1 ? &nbhd_sizes : NULL,
                      points, k, r, ptr_subsample_indices, num_eigen, verbose,
                      num_procs);
  out1 = PyArray_EMPTY(out1_ndim, out1_dims, typenum, false);
  copy(evecs.begin(), evecs.end(), (T*)PyArray_DATA((PyArrayObject*)out1));
  if (output_eigenvalues) {
    out2 = PyArray_EMPTY(out2_ndim, out2_dims, typenum, false);
    copy(evals.begin(), evals.end(), (T*)PyArray_DATA((PyArrayObject*)out2));
  }
  if (output_neighborhood_sizes) {
    out3 = PyArray_EMPTY(out3_ndim, out3_dims, out3_typenum, false);
    copy(nbhd_sizes.begin(), nbhd_sizes.end(),
         (int*)PyArray_DATA((PyArrayObject*)out3));
  }
}

static char estimate_normals_usage[] =
    "Estimates normals at all points using principal component analysis\n"
    "(PCA). Specifically, computes eigenvectors of the covariance matrix C\n"
    "based on the local neighborhoods of each point.\n"
    "\n"
    ".. math::\n"
    "   C = \\frac{1}{N}\\sum_{i=1}^{N}{(p_i-\\mu)(p_i-\\mu)^T}\n"
    "\n"
    "where :math:`p_1 ... p_N` are the points in a given neighborhood and\n"
    ":math:`\\mu` is the centroid of the points.\n"
    "\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "points : 3-column numpy array of type float32 or float64\n"
    "    Input point cloud.\n"
    "k : int\n"
    "    Number of neighbors to use.\n"
    "    For pure r-near neighborhoods, set this to -1.\n"
    "r : float\n"
    "    Use neighbors within r of query point.\n"
    "    For pure k-nearest neighborhoods, set this to np.inf.\n"
    "subsample : 1-d array of bool or int, optional\n"
    "    Optionally estimate normals at subset of points specified by a\n"
    "    boolean mask having length equal to the number of points, or by\n"
    "    integer indices into the array of points. Default: None.\n"
    "output_eigenvalues : bool, optional\n"
    "    Default: False.\n"
    "output_all_eigenvectors : bool, optional\n"
    "    Default: False.\n"
    "output_neighborhood_sizes : bool, optional\n"
    "    Default: False.\n"
    "verbose : bool, optional\n"
    "    Default: True.\n"
    "num_procs : int, optional\n"
    "    Default: use all processors.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "results : ndarray or tuple of ndarray's\n"
    "    PCA results. The tuple return type is triggered if any of\n"
    "    output_eigenvalues or output_neighborhood_sizes is True.\n"
    "\n"
    "\n"
    "The following table summarizes all potential return types. Here m is the\n"
    "number of PCAs performed. If subsample is not None, then m is the size\n"
    "of subsample. Otherwise, m is just the number of input points.\n"
    "\n"
    "+---------------------------+------------------------------------------+\n"
    "|                           |output_all_eigenvectors                   |\n"
    "|                           +----------------+-------------------------+\n"
    "|                           |F               |T                        |\n"
    "+======================+====+================+=========================+\n"
    "|**output_eigenvalues**|F, F|m x 3           |m x 3 x 3                |\n"
    "|                      +----+----------------+-------------------------+\n"
    "|**output_nbhd_sizes** |F, T|(m x 3, None, m)|(m x 3 x 3, None, m)     |\n"
    "|                      +----+----------------+-------------------------+\n"
    "|                      |T, F|(m x 3, m, None)|(m x 3 x 3, m x 3, None) |\n"
    "|                      +----+----------------+-------------------------+\n"
    "|                      |T, T|(m x 3, m, m)   |(m x 3 x 3, m x 3, m)    |\n"
    "+----------------------+----+----------------+-------------------------+\n"
    "\n"
    "Note:\n"
    "    * The j-th eigenvector of the i-th PCA result is given by\n"
    "      indices [i, j, :].\n"
    "    * eigenvectors are sorted in order of increasing eigenvalue\n"
    "\n";

static PyObject* estimate_normals_wrapper(PyObject* self, PyObject* args,
                                          PyObject* kwargs) {
  PyObject* p = NULL;
  PyObject* subsample = Py_None;
  int k;
  float r;
  int output_eigenvalues = 0;
  int output_all_eigenvectors = 0;
  int output_neighborhood_sizes = 0;
  int verbose = 1;
  int num_procs = -1;
  static char* keywords[] = {
      "points", "k", "r", "subsample", "output_eigenvalues",
      "output_all_eigenvectors", "output_neighborhood_sizes", "verbose",
      "num_procs", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oif|Oiiiii", keywords, &p, &k,
                                   &r, &subsample, &output_eigenvalues,
                                   &output_all_eigenvectors,
                                   &output_neighborhood_sizes, &verbose,
                                   &num_procs)) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to parse inputs");
    return NULL;
  }

  // check k and r
  if (r <= 0.0f) {
    PyErr_SetString(PyExc_ValueError, "r must be positive");
    return NULL;
  } else if (k == 0) {
    PyErr_SetString(PyExc_ValueError, "k cannot be zero");
    return NULL;
  } else if (k < 0 && r == std::numeric_limits<float>::infinity()) {
    PyErr_SetString(PyExc_ValueError, "invalid combo: r == inf and k < 0");
    return NULL;
  }

  Array2D arr;
  if (!CheckAndExtractArray2D(arr, p)) {
    if (!PyErr_Occurred())
      PyErr_SetString(PyExc_TypeError,
                      "points must be interpretable as 0, 1 or 2-d array");
    return NULL;
  }

  std::vector<int> subsample_indices;
  std::vector<int>* ptr_subsample_indices = NULL;
  if (subsample != Py_None) {
    if (!CheckAndExtractIndices(subsample_indices, subsample, arr.m)) {
      if (!PyErr_Occurred())
        PyErr_SetString(PyExc_TypeError,
                        "subsample must be interpretable as specifying a "
                        "subset of points");
      return NULL;
    }
    ptr_subsample_indices = &subsample_indices;
  }

  PyObject* out1 = NULL;
  PyObject* out2 = NULL;
  PyObject* out3 = NULL;
  if (arr.type_num == NPY_FLOAT32) {
    estimate_normals<float>(out1, out2, out3, arr, k, r, ptr_subsample_indices,
                            (bool)output_eigenvalues,
                            (bool)output_all_eigenvectors,
                            (bool)output_neighborhood_sizes, (bool)verbose,
                            num_procs);
  } else if (arr.type_num == NPY_FLOAT64) {
    estimate_normals<double>(out1, out2, out3, arr, k, r, ptr_subsample_indices,
                             (bool)output_eigenvalues,
                             (bool)output_all_eigenvectors,
                             (bool)output_neighborhood_sizes, (bool)verbose,
                             num_procs);
  } else {
    PyErr_SetString(PyExc_TypeError, "points must be float32 or float64");
    return NULL;
  }

  if (out2 == NULL && out3 == NULL) {
    return out1;
  } else {
    PyObject* out = PyTuple_New(3);
    PyTuple_SetItem(out, 0, out1);
    PyTuple_SetItem(out, 1, out2 != NULL ? out2 : Py_None);
    PyTuple_SetItem(out, 2, out3 != NULL ? out3 : Py_None);
    return out;
  }
}

static PyMethodDef methods[] = {
    {"estimate_normals", (PyCFunction)estimate_normals_wrapper,
     METH_VARARGS | METH_KEYWORDS, estimate_normals_usage},
    {NULL, NULL, 0, NULL}};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef module_def = {PyModuleDef_HEAD_INIT,
                                        "estimate_normals",
                                        NULL,
                                        -1,
                                        methods,
                                        NULL,
                                        NULL,
                                        NULL,
                                        NULL};

PyMODINIT_FUNC PyInit_estimate_normals(void) {
  PyObject* module = PyModule_Create(&module_def);
#else
PyMODINIT_FUNC initestimate_normals(void) {
  (void)Py_InitModule("estimate_normals", methods);
#endif

  import_array();

#if PY_MAJOR_VERSION >= 3
  return module;
#endif
}
