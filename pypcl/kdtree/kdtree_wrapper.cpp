#include <iostream>
#include <limits>
#include "python_util.h"
#include "kdtree.h"
using namespace std;

struct KdTreeStruct {
  KdTreeStruct(void* tree_ptr, int type_num, int dim)
      : tree_ptr(tree_ptr), type_num(type_num), dim(dim) {}
  void* tree_ptr;
  int type_num;
  int dim;
};

template <typename Action, int dim>
void PerformAction_(Action& action, int type_num) {
  if (type_num == NPY_FLOAT32)
    action.template Perform<float, dim>();
  else if (type_num == NPY_FLOAT64)
    action.template Perform<double, dim>();
  else if (type_num == NPY_INT8)
    action.template Perform<std::int8_t, dim>();
  else if (type_num == NPY_INT16)
    action.template Perform<std::int16_t, dim>();
  else if (type_num == NPY_INT32)
    action.template Perform<std::int32_t, dim>();
  else if (type_num == NPY_INT64)
    action.template Perform<std::int64_t, dim>();
  else if (type_num == NPY_UINT8)
    action.template Perform<std::uint8_t, dim>();
  else if (type_num == NPY_UINT16)
    action.template Perform<std::uint16_t, dim>();
  else if (type_num == NPY_UINT32)
    action.template Perform<std::uint32_t, dim>();
  else if (type_num == NPY_UINT64)
    action.template Perform<std::uint64_t, dim>();
  else {
    PyErr_Format(PyExc_RuntimeError,
                 "PerformAction_(): Invalid k-d tree type_num = %d.", type_num);
  }
}

template <typename Action>
void PerformAction(Action& action, int type_num, int dim) {
  if (dim == 2) {
    PerformAction_<Action, 2>(action, type_num);
  } else if (dim == 3) {
    PerformAction_<Action, 3>(action, type_num);
  } else if (dim == 4) {
    PerformAction_<Action, 4>(action, type_num);
  } else {
    PyErr_Format(PyExc_RuntimeError,
	             "PerformAction(): Invalid k-d tree dim = %d.", dim);
  }
}

class DeleteTreeAction {
 public:
  DeleteTreeAction(KdTreeStruct* ptr) : ptr_(ptr) {}
  template <typename T, int dim>
  void Perform() {
    delete (pointkd::KdTree<T, dim>*)ptr_->tree_ptr;
    delete ptr_;
  }
 private:
  KdTreeStruct* ptr_;
};

void DeleteKdTree(PyObject* obj) {
  KdTreeStruct* ptr = (KdTreeStruct*)PyCapsule_GetPointer(obj, NULL);
  DeleteTreeAction action(ptr);
  PerformAction<DeleteTreeAction>(action, ptr->type_num, ptr->dim);
}

class BuildTreeAction {
 public:
  BuildTreeAction(const Array2D& x, const pointkd::BuildParams& params)
      : x_(x), params_(params), results_(NULL) {}

  template <typename T, int dim>
  void Perform() {
    typedef pointkd::KdTree<T, dim> KdTreeT;
    KdTreeT* tree;
    // TODO(longer term): make kd-tree accept pointer to strided array
    if (IsContiguous(x_)) {
      tree = new KdTreeT((const T*)x_.data, x_.m, params_);
    } else {
      std::vector<T> points;
      VectorFromArray2D(points, x_);
      tree = new KdTreeT(points, params_);
    }
    KdTreeStruct* kdtree_struct = new KdTreeStruct((void*)tree,
                                                   x_.type_num, dim);
    results_ = PyCapsule_New((void*)kdtree_struct, NULL, &DeleteKdTree);
  }

  PyObject* results() const { return results_; }

 private:
  const Array2D& x_;
  const pointkd::BuildParams& params_;
  PyObject* results_;
};

PyObject* MakeList(const std::vector<pointkd::Indices>& v) {
  PyObject* X = PyList_New((Py_ssize_t)v.size());
  for (std::size_t i = 0; i < v.size(); i++) {
    npy_intp n = (npy_intp)v[i].size();
    PyObject* x = PyArray_EMPTY(1, &n, NPY_INT32, false);
    copy(v[i].begin(), v[i].end(), (int*)PyArray_DATA((PyArrayObject*)x));
    PyList_SetItem(X, i, x);
  }
  return X;
}

template <typename T, int dim>
PyObject* QueryWithIndices(const pointkd::KdTree<T, dim>* kdtree,
                           const pointkd::Indices& indices,
                           long k, double dmax) {
  // assume k >= -1 and dMax >= 0.0
  typedef typename pointkd::KdTree<T, dim>::DistT DistT;
  std::vector<pointkd::Indices> results;
  if (k > 0) {  // k-nearest
    kdtree->KNearestNeighborsSelf(results, indices, (int)k, (DistT)dmax);
    return MakeList(results);
  } else if (dmax != std::numeric_limits<double>::infinity()) {  // r-near
    kdtree->RNearNeighborsSelf(results, indices, (DistT)dmax);
    return MakeList(results);
  } else {
    PyErr_Format(PyExc_ValueError,
                 "QueryWithIndices(): "
				 "k = %ld and dmax = %lf is an invalid combination.", k, dmax);
    return NULL;
  }
}

template <typename Tx, typename Ty, int dim>
PyObject* QueryWithPoints(const pointkd::KdTree<Tx, dim>* kdtree,
                          const std::vector<Ty>& query_points,
                          long k, double dmax) {
  // assume k >= -1 and dMax >= 0.0
  typedef typename pointkd::KdTree<Tx, dim>::DistT DistT;
  std::vector<pointkd::Indices> results;
  if (k > 0) {  // k-nearest
    kdtree->KNearestNeighbors(results, query_points, k, (DistT)dmax);
    return MakeList(results);
  } else if (dmax != std::numeric_limits<double>::infinity()) {  // r-near
    kdtree->RNearNeighbors(results, query_points, (DistT)dmax);
    return MakeList(results);
  } else {
    PyErr_Format(PyExc_ValueError,
                 "QueryWithPoints(): "
				 "k = %ld and dmax = %lf is an invalid combination.", k, dmax);
    return NULL;
  }
}

class QueryTreeAction {
 public:
  QueryTreeAction(const KdTreeStruct* ptr, PyObject* obj_queries, long k, double dmax)
      : ptr_(ptr),
        obj_queries_(obj_queries),
        k_(k),
        dmax_(dmax),
        results_(NULL) {}
  template <typename T, int dim>
  void Perform() {
    const pointkd::KdTree<T, dim>* tree =
        (const pointkd::KdTree<T, dim>*)ptr_->tree_ptr;
    pointkd::Indices indices;
    std::vector<pointkd::Indices> r;
    if (obj_queries_ == NULL || obj_queries_ == Py_None) {
      // query with indices 0 ... num_points-1
      for (int i = 0; i < tree->num_points(); i++)
        indices.push_back(i);
      results_ = QueryWithIndices(tree, indices, k_, dmax_);
    } else if (PySlice_Check(obj_queries_)) {
      // query with indices represented by a slice (begin:end:step)
      PyErr_SetString(PyExc_NotImplementedError,
	                  "QueryTreeAction::Perform(): "
                      "slice-based query not yet implemented");
      results_ = NULL;
    } else if (PyArray_Check(obj_queries_) && PyArray_NDIM(obj_queries_) == 2) {
      // query with points
      Array2D x;
      ExtractArray2DFromPyArray(x, obj_queries_);
	  if (x.n != dim) {
		  PyErr_Format(PyExc_ValueError,
		               "QueryTreeAction::Perform(): "
		               "query point dim = %d (expecting dim = %d).",
					   (int)x.n, dim);
		  results_ = NULL;
      } else if (x.type_num == NPY_FLOAT32) {
        std::vector<float> q;
        VectorFromArray2D(q, x);
        results_ = QueryWithPoints(tree, q, k_, dmax_);
      } else if (x.type_num == NPY_FLOAT64) {
        std::vector<double> q;
        VectorFromArray2D(q, x);
        results_ = QueryWithPoints(tree, q, k_, dmax_);
      } else if (x.type_num == NPY_INT8) {
        std::vector<std::int8_t> q;
        VectorFromArray2D(q, x);
        results_ = QueryWithPoints(tree, q, k_, dmax_);
      } else if (x.type_num == NPY_INT16) {
        std::vector<std::int16_t> q;
        VectorFromArray2D(q, x);
        results_ = QueryWithPoints(tree, q, k_, dmax_);
      } else if (x.type_num == NPY_INT32) {
        std::vector<std::int32_t> q;
        VectorFromArray2D(q, x);
        results_ = QueryWithPoints(tree, q, k_, dmax_);
      } else if (x.type_num == NPY_INT64) {
        std::vector<std::int64_t> q;
        VectorFromArray2D(q, x);
        results_ = QueryWithPoints(tree, q, k_, dmax_);
      } else if (x.type_num == NPY_UINT8) { 
        std::vector<std::uint8_t> q;
        VectorFromArray2D(q, x);
        results_ = QueryWithPoints(tree, q, k_, dmax_);
      } else if (x.type_num == NPY_UINT16) {
        std::vector<std::uint16_t> q;
        VectorFromArray2D(q, x);
        results_ = QueryWithPoints(tree, q, k_, dmax_);
      } else if (x.type_num == NPY_UINT32) {
        std::vector<std::uint32_t> q;
        VectorFromArray2D(q, x);
        results_ = QueryWithPoints(tree, q, k_, dmax_);
      } else if (x.type_num == NPY_UINT64) {
        std::vector<std::uint64_t> q;
        VectorFromArray2D(q, x);
        results_ = QueryWithPoints(tree, q, k_, dmax_);
      }
    } else if (CheckAndExtractIndices(indices,
	                                  obj_queries_, tree->num_points())) {
      results_ = QueryWithIndices(tree, indices, k_, dmax_);
    } else {
      if (!PyErr_Occurred())
        PyErr_Format(PyExc_TypeError,
		             "QueryTreeAction::Perform(): "
                     "could not use object of type %s as query input.",
                     obj_queries_->ob_type->tp_name);
      results_ = NULL;
    }
  }

  PyObject* results() const { return results_; }

 private:
  const KdTreeStruct* ptr_;
  PyObject* obj_queries_;
  long k_;
  double dmax_;
  PyObject* results_;
};

static PyObject* Build(PyObject* self, PyObject* args, PyObject* kwargs) {
  static char* keywords[] = {(char*)"data", (char*)"numprocs",
                             (char*)"maxleafsize", (char*)"emptysplit", NULL};
  PyObject* data = NULL;
  pointkd::BuildParams build_params;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|iid", keywords, &data,
                                   &build_params.num_proc, &build_params.max_leaf_size, &build_params.empty_split_threshold)) {
    PyErr_SetString(PyExc_TypeError, "Build(): failed parsing arguments");
    return NULL;
  }
  // TODO: make sure data is never NULL at this point 
  Array2D x;
  if (CheckAndExtractArray2D(x, data)) {
    BuildTreeAction action(x, build_params);
    PerformAction<BuildTreeAction>(action, x.type_num, x.n);
    return action.results();
  } else {
    if (!PyErr_Occurred())
      PyErr_Format(PyExc_TypeError, "Build(): points array type %s unsupported",
                   data->ob_type->tp_name);
    return NULL;
  }
}

static PyObject* Query(PyObject* self, PyObject* args, PyObject* kwargs) {
  const float flt_infinity = numeric_limits<float>::infinity();
  const double dbl_infinity = numeric_limits<double>::infinity();
  PyObject* obj_kdtree;
  PyObject* obj_queries;
  PyObject* obj_k = NULL;
  PyObject* obj_dmax = NULL;
  int num_procs = -1;
  static char* keywords[] = {(char*)"kdtree", (char*)"queries",  (char*)"k",
                             (char*)"dmax",   (char*)"numprocs", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|OOi", keywords,
                                   &obj_kdtree, &obj_queries, &obj_k, &obj_dmax,
                                   &num_procs)) {
    PyErr_SetString(PyExc_TypeError, "Query(): failed parsing arguments");
    return NULL;
  }

  // check option obj_k and set k
  long k;
  if (obj_k == NULL) {  // k not specified
    k = 1;
  } else if (obj_k == Py_None) {
    k = -1;
  } else if (!CastAsLong(k, obj_k)) {
    if (!PyErr_Occurred())
      PyErr_Format(PyExc_RuntimeError,
                   "Query(): could not interpret k of type %s as C long.",
                   obj_k->ob_type->tp_name);
    return NULL;
  } else if (k <= 0) {
    PyErr_Format(PyExc_ValueError, "Query(): encountered negative k=%ld", k);
    return NULL; 
  }

  // check option obj_dmax and set dmax
  double dmax;
  if (obj_dmax == NULL) {  // dmax not specified
    dmax = dbl_infinity;
  } else if (obj_dmax == Py_None) {
    dmax = dbl_infinity;
  } else if (!CastAsDouble(dmax, obj_dmax)) {
    if (!PyErr_Occurred())
      PyErr_Format(PyExc_RuntimeError,
                   "Query(): could not interpret dmax of type %s as C double.",
                   obj_dmax->ob_type->tp_name);
    return NULL;
  } else if (dmax < 0.0) {
    PyErr_Format(PyExc_ValueError,
                 "Query(): encountered negative dmax %lf", dmax);
    return NULL;
  }

  // get pointer to k-d tree
  const KdTreeStruct* kdtree_struct;
  if (PyCapsule_CheckExact(obj_kdtree)) {
    kdtree_struct = (const KdTreeStruct*)PyCapsule_GetPointer(obj_kdtree, NULL);
  } else {
    PyErr_SetString(PyExc_TypeError,
                    "Query(): "
                    "first arg must be capsule of a KdTreeStruct pointer");
    return NULL;
  }

  // note: at this point, k >= -1 and dMax >= 0.0
  QueryTreeAction action(kdtree_struct, obj_queries, k, dmax);
  PerformAction<QueryTreeAction>(action,
                                 kdtree_struct->type_num, kdtree_struct->dim);
  if (PyErr_Occurred()) {
    return NULL;
  } else {
    return action.results();
  }
}

static PyMethodDef methods_table[] = {
    {"_build", (PyCFunction)Build, METH_KEYWORDS, "build k-d tree"},
    {"_query", (PyCFunction)Query, METH_KEYWORDS,
     "performs either k-nearest or r-near"},
    {NULL, NULL, 0, NULL}};

PyMODINIT_FUNC initkdtree(void) {
  Py_InitModule("kdtree", methods_table);
  import_array();
}
