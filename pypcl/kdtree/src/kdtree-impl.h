/** TODO: license boiler plate here
  *
  * By Victor Lu (victor.1.lu@here.com)
*/

#ifndef __KDTREE_IMPL_H__
#define __KDTREE_IMPL_H__

#include <cmath>
#include <queue>
#include <vector>
#include "kdtree.h"

#ifdef USE_TBB
#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"
#include "tbb/scalable_allocator.h"
#include "tbb/task.h"
#include "tbb/task_scheduler_init.h"
inline void* Allocate(size_t size) { return scalable_malloc(size); }
inline void Free(void* ptr) { return scalable_free(ptr); }
#else
inline void* Allocate(size_t size) { return malloc(size); }
inline void Free(void* ptr) { return free(ptr); }
#endif  // USE_TBB

#ifdef max
#undef max
#endif

#ifdef min
#undef min
#endif

namespace pointkd {
namespace impl {
struct EmptyGap {
  EmptyGap() : dim(-1), side(-1), size(0.0) {}
  EmptyGap(int dim, int side, double size) : dim(dim), side(side), size(size) {}
  int dim;
  int side;
  double size;
};

template <bool B, class T = void>
struct enable_if {};

template <class T>
struct enable_if<true, T> {
  typedef T type;
};

template <typename T>
typename enable_if<!std::is_floating_point<T>::value>::type ValidPointIndices(
    std::vector<int>& indices, const T* points, int num_points, int dim) {
  for (int i = 0; i < num_points; i++) {
    indices.push_back(i);
  }
}

template <typename T>
typename enable_if<std::is_floating_point<T>::value>::type ValidPointIndices(
    std::vector<int>& indices, const T* points, int num_points, int dim) {
  for (int i = 0; i < num_points; i++) {
    bool valid_point = true;
    for (int j = 0; j < dim; j++) {
      T v = points[i * dim + j];
      if (std::isinf(v) || std::isnan(v)) {
        valid_point = false;
        break;
      }
    }
    if (valid_point) indices.push_back(i);
  }
}

template <typename T>
void DestructorHelper(Node<T>* node) {
  if (node) {
    DestructorHelper(node->left);
    DestructorHelper(node->right);
    Free(node);
  }
}

template <typename T>
Node<T>* CopyConstructorHelper(const Node<T>* node) {
  if (node) {
    Node<T>* node_copy = (Node<T>*)Allocate(sizeof(Node<T>));
    *node_copy = *node;
    node_copy->left = CopyConstructorHelper(node->left);
    node_copy->right = CopyConstructorHelper(node->right);
    return node_copy;
  } else {
    return NULL;
  }
}

template <typename T, int dim>
EmptyGap LargestEmptyGap(const Box<T, dim>& A, const Box<T, dim>& B) {
  // assume A is entirely contained in B
  EmptyGap max_gap(-1, -1, 0.0);
  for (int i = 0; i < dim; i++) {
    double gap = std::max(0.0, (double)B.max(i) - (double)A.max(i));
    if (gap >= max_gap.size)  // must use >= rather than >, to ensure
                              // max_gap is overwritten at least once
      max_gap = EmptyGap(i, 1, gap);
    gap = std::max(0.0, (double)A.min(i) - (double)B.min(i));
    if (gap >= max_gap.size) max_gap = EmptyGap(i, 0, gap);
  }
  return max_gap;
}

template <typename T, int dim>
Node<T>* TrimEmptyGaps(Node<T>*& current_node, int begin_index, int end_index,
                       Box<T, dim>& node_box, Box<T, dim>& bounding_box,
                       double trim_threshold) {
  // perform as many empty splits as possible
  Node<T>* node = NULL;
  current_node = NULL;
  while (true) {
    EmptyGap gap = LargestEmptyGap(bounding_box, node_box);
    if (gap.size == 0.0)  // node_box identical to bounding_box
                          // nothing to trim
      break;
    double width = node_box.width(gap.dim);
    // gap.size != 0.0 implies width > 0.0,
    // therefore no need to worry about divide by zero
    if (gap.size / width <= trim_threshold) break;

    Node<T>* new_node = (Node<T>*)Allocate(sizeof(Node<T>));
    new_node->left = new_node->right = NULL;
    new_node->split_dim = gap.dim;
    if (gap.side == 0) {  // left empty
      new_node->split_value = bounding_box.min(gap.dim);
      new_node->split_index = begin_index;
      node_box.min(gap.dim) = new_node->split_value;
    } else {  // right empty
      new_node->split_value = bounding_box.max(gap.dim);
      new_node->split_index = end_index;
      node_box.max(gap.dim) = new_node->split_value;
    }

    if (current_node) {
      if ((int)current_node->split_index == begin_index) {  // left empty
        current_node->left = NULL;
        current_node->right = new_node;
      } else {  // current_node->split_index == end_index (right empty)
        current_node->left = new_node;
        current_node->right = NULL;
      }
    }
    current_node = new_node;

    if (!node) node = new_node;
  }
  return node;
}

template <typename T>
typename enable_if<std::is_unsigned<T>::value, T>::type midpoint(T a, T b) {
  if (a < b) std::swap(a, b);
  return (a - b) / 2 + b;  // a >= b
}

template <typename T>
typename enable_if<!std::is_unsigned<T>::value, T>::type midpoint(T a, T b) {
  // to avoid arithmetic overflow
  if ((a >= 0 && b >= 0) || (a < 0 && b < 0))
    return (b - a) / (T)2 + a;
  else
    return (a + b) / (T)2;
}

template <typename T, int dim>
void ComputeSplit(int& split_dim, T& split_val, Box<T, dim>& bounding_box) {
  // split mid way along dimension where bounding_box is widest
  T max_width = (T)0;
  for (int i = 0; i < dim; i++) {
    T width = bounding_box.max(i) - bounding_box.min(i);
    if (width >= max_width) {
      max_width = width;
      split_dim = i;
      if (bounding_box.min(i) == bounding_box.max(i))
        split_val = bounding_box.min(i);
      else
        split_val = midpoint(bounding_box.min(i), bounding_box.max(i));
    }
  }
}

template <typename T, int dim>
int PartitionIndices(int* indices, int count, int split_dim, T split_val,
                     const T* points) {
  int left = 0;
  int right = count - 1;
  for (;;) {
    while (left <= right &&
           points[indices[left] * dim + split_dim] < split_val) {
      left++;
    }
    while (left <= right &&
           points[indices[right] * dim + split_dim] >= split_val) {
      right--;
    }
    if (left >= right)  // does left > right also give same behavior?
      break;
    std::swap(indices[left], indices[right]);
    left++;
    right--;
  }
  int lim1 = left;
  right = count - 1;
  for (;;) {
    while (left <= right &&
           points[indices[left] * dim + split_dim] <= split_val) {
      left++;
    }
    while (left <= right &&
           points[indices[right] * dim + split_dim] > split_val) {
      right--;
    }
    if (left >= right) break;
    std::swap(indices[left], indices[right]);
    left++;
    right--;
  }
  int lim2 = left;

  // at this point, [0,count) is partitioned as follows:
  //   [0,lim1) - strictly less than split_val
  //   [lim1,lim2) - equal to split_val
  //   [lim2,count) - strictly greater than split_val
  int num_left;
  if (lim1 > count / 2)
    num_left = lim1;
  else if (lim2 < count / 2)
    num_left = lim2;
  else
    num_left = count / 2;

  // when does this happen?
  if (lim1 == count && lim2 == 0) num_left = count / 2;

  return num_left;
}

template <typename T, int dim>
Node<T>* MakeNode(Node<T>*& current_node, int begin_index, int end_index,
                  std::vector<int>& indices, Box<T, dim>& node_box,
                  const T* points, int num_points,
                  const pointkd::BuildParams& build_params) {
  typedef typename KdTree<T, dim>::DistT DistT;
  int node_size = end_index - begin_index;
  Node<T>* node = NULL;
  current_node = NULL;  // needed in case is_root and TrimEmptyGaps not run
  Box<T, dim> bounding_box(node_box);
  bool is_root = (node_size == num_points);
  if (!is_root) {  // non-root node extents may not be tight
    // make bounding_box tight
    Box<T, dim> temp_box;
    for (int i = begin_index; i < end_index; i++)
      temp_box.AddPoint(&points[indices[i] * dim]);
    bounding_box = temp_box;

    node =
        TrimEmptyGaps<T, dim>(current_node, begin_index, end_index, node_box,
                              bounding_box, build_params.empty_split_threshold);
  }

  // split node if node is too large
  if (!bounding_box.IsPoint() &&  // a degenerate bounding_box implies all
                                  // points are coinciding, in which case
                                  // we avoid further splitting
      node_size > build_params.max_leaf_size) {
    Node<T>* new_node = (Node<T>*)Allocate(sizeof(Node<T>));
    new_node->left = new_node->right = NULL;
    int split_dim;
    ComputeSplit<T, dim>(split_dim, new_node->split_value, bounding_box);
    new_node->split_dim = (unsigned int)split_dim;
    int num_left = PartitionIndices<T, dim>(&indices[begin_index], node_size,
                                            new_node->split_dim,
                                            new_node->split_value, points);
    new_node->split_index = begin_index + num_left;

    if (current_node) {
      // current node is an empty split node
      if ((int)current_node->split_index == begin_index)  // left empty
        current_node->right = new_node;
      else  // right empty
        current_node->left = new_node;
    }
    current_node = new_node;
    if (!node) node = current_node;
  }
  return node;
}

template <typename T, int dim>
Node<T>* RecursiveBuildHelper(int begin_index, int end_index,
                              std::vector<int>& indices, Box<T, dim>& node_box,
                              const T* points, int num_points,
                              const pointkd::BuildParams& build_params) {
  typedef typename KdTree<T, dim>::DistT DistT;
  Node<T>* current_node;
  Node<T>* node =
      MakeNode<T, dim>(current_node, begin_index, end_index, indices, node_box,
                       points, num_points, build_params);
  if (!current_node) return node;
  T split_value = current_node->split_value;
  int split_dim = current_node->split_dim;
  int split_index = current_node->split_index;

  if (begin_index < split_index) {
    Box<T, dim> left_box(node_box);
    left_box.max(split_dim) = split_value;
    current_node->left = RecursiveBuildHelper<T, dim>(begin_index, split_index,
                                                      indices, left_box, points,
                                                      num_points, build_params);
  }
  if (split_index < end_index) {
    Box<T, dim> right_box(node_box);
    right_box.min(split_dim) = split_value;
    current_node->right =
        RecursiveBuildHelper<T, dim>(split_index, end_index, indices, right_box,
                                     points, num_points, build_params);
  }
  return node;
}

template <typename A, typename B, typename DistT, int dim>
DistT PointDistance2(const A* a, const B* b) {
  DistT dist2 = (DistT)0.0;
  for (int i = 0; i < dim; i++) {
    DistT temp = (DistT)b[i] - (DistT)a[i];
    dist2 += temp * temp;
  }
  return dist2;
}

template <typename DistT, int dim>
DistT VectorLength2(const DistT* v) {
  DistT length2 = (DistT)0.0;
  for (int i = 0; i < dim; i++) length2 += v[i] * v[i];
  return length2;
}

template <typename T, typename Q, int dim>
void KNearestNeighborsHelper(
    typename KdTree<T, dim>::PriorityQueue& nearest_neighbors,
    Box<T, dim>& node_box, int begin_index, int end_index, Node<T>* node,
    const Q* q, const int k,
    const typename KdTree<T, dim>::DistT r2,  // distance squared
    const std::vector<T>& points) {
  // assumes k > 0
  // assumes squared distance to current node is less than r2
  typedef typename KdTree<T, dim>::DistT DistT;
  typedef typename KdTree<T, dim>::Pair Pair;
  int num_points = end_index - begin_index;
  if (!node ||  // node is a leaf
      num_points <= k - (int)nearest_neighbors.size() &&
          MaxDist2<Q, T, dim, DistT>(q, node_box) < r2) {
    for (int i = begin_index; i < end_index; i++) {
      DistT dist2 = PointDistance2<Q, T, DistT, dim>(q, &points[i * dim]);
      if (dist2 >= r2) continue;
      if (nearest_neighbors.size() < k)
        nearest_neighbors.push(Pair(i, dist2));
      else if (dist2 < nearest_neighbors.top().dist2) {
        // nearest_neighbors.size() == k
        nearest_neighbors.pop();
        nearest_neighbors.push(Pair(i, dist2));
      }
    }
  } else {  // node is not a leaf
    struct NodeInfo {
      Node<T>* ptr;
      int begin;
      int end;
      int side;
    };

    NodeInfo near, far;
    T split_value = node->split_value;
    unsigned int split_dim = node->split_dim;
    if (q[split_dim] < node->split_value) {
      near.ptr = node->left;
      near.begin = begin_index;
      near.end = node->split_index;
      near.side = 1;
      far.ptr = node->right;
      far.begin = node->split_index;
      far.end = end_index;
      far.side = 0;
    } else {
      near.ptr = node->right;
      near.begin = node->split_index;
      near.end = end_index;
      near.side = 0;
      far.ptr = node->left;
      far.begin = begin_index;
      far.end = node->split_index;
      far.side = 1;
    }

    if (near.begin < near.end) {  // near child is non-empty
      T temp = node_box.val(split_dim, near.side);
      node_box.val(split_dim, near.side) = split_value;
      KNearestNeighborsHelper<T, Q, dim>(nearest_neighbors, node_box,
                                         near.begin, near.end, near.ptr, q, k,
                                         r2, points);
      node_box.val(split_dim, near.side) = temp;
    }

    if (far.begin == far.end)  // far child is empty
      return;

    T temp = node_box.val(split_dim, far.side);
    node_box.val(split_dim, far.side) = split_value;
    DistT far_dist2 = MinDist2<Q, T, dim, DistT>(q, node_box);
    if (far_dist2 < r2) {
      if (nearest_neighbors.size() < k ||
          far_dist2 < nearest_neighbors.top().dist2)
        KNearestNeighborsHelper<T, Q, dim>(nearest_neighbors, node_box,
                                           far.begin, far.end, far.ptr, q, k,
                                           r2, points);
    }
    node_box.val(split_dim, far.side) = temp;
  }
}

template <typename T, typename Q, int dim>
void KNearestNeighborsHelper(
    typename KdTree<T, dim>::PriorityQueue& nearest_neighbors,
    Box<T, dim>& node_box, int begin_index, int end_index, int node_index,
    const Q* q, const int k,
    const typename KdTree<T, dim>::DistT r2,  // distance squared
    const std::vector<SmallNode<T> >& nodes, const std::vector<T>& points) {
  // assumes k > 0
  // assumes squared distance to current node is less than r2
  typedef typename KdTree<T, dim>::DistT DistT;
  typedef typename KdTree<T, dim>::Pair Pair;
  if (node_index == -1 ||
      end_index - begin_index <= k - (int)nearest_neighbors.size() &&
          MaxDist2<Q, T, dim, DistT>(q, node_box) < r2) {
    // TODO: consider accelerating special case of degenerate node extent
    //       i.e. all points in leaf are coinciding
    for (int i = begin_index; i < end_index; i++) {
      DistT dist2 = PointDistance2<Q, T, DistT, dim>(q, &points[i * dim]);
      if (dist2 >= r2) continue;
      if (nearest_neighbors.size() < k)
        nearest_neighbors.push(Pair(i, dist2));
      else if (dist2 < nearest_neighbors.top().dist2) {
        // nearest_neighbors.size() == k
        // TODO: consider merging pop and push in a customized priority
        //       queue implementation
        nearest_neighbors.pop();
        nearest_neighbors.push(Pair(i, dist2));
      }
    }
  } else {  // node_index != -1
    const SmallNode<T>& node = nodes[node_index];
    int split_dim = node.GetSplitDim();
    int split_index = node.GetSplitIndex();
    T split_value = node.split_value;

    struct NodeInfo {
      int begin;
      int end;
      int index;
      int side;
    };

    NodeInfo near, far;
    if (q[split_dim] < split_value) {
      near.index = node.LeftChildIndex(node_index);
      near.begin = begin_index;
      near.end = split_index;
      near.side = 1;
      far.index = node.RightChildIndex(node_index);
      far.begin = split_index;
      far.end = end_index;
      far.side = 0;
    } else {
      near.index = node.RightChildIndex(node_index);
      near.begin = split_index;
      near.end = end_index;
      near.side = 0;
      far.index = node.LeftChildIndex(node_index);
      far.begin = begin_index;
      far.end = split_index;
      far.side = 1;
    }
    if (near.begin < near.end) {  // near child is non-empty
      T temp = node_box.val(split_dim, near.side);
      node_box.val(split_dim, near.side) = split_value;
      KNearestNeighborsHelper<T, Q, dim>(nearest_neighbors, node_box,
                                         near.begin, near.end, near.index, q, k,
                                         r2, nodes, points);
      node_box.val(split_dim, near.side) = temp;
    }

    if (far.begin == far.end)  // far child is empty
      return;

    T temp = node_box.val(split_dim, far.side);
    node_box.val(split_dim, far.side) = split_value;
    DistT far_dist2 = MinDist2<Q, T, dim, DistT>(q, node_box);
    if (far_dist2 < r2) {
      if (nearest_neighbors.size() < k ||
          far_dist2 < nearest_neighbors.top().dist2)
        KNearestNeighborsHelper<T, Q, dim>(nearest_neighbors, node_box,
                                           far.begin, far.end, far.index, q, k,
                                           r2, nodes, points);
    }
    node_box.val(split_dim, far.side) = temp;
  }
}

template <typename T, typename Q, int dim>
void RNearNeighborsHelper(Indices& results, Box<T, dim>& node_box,
                          int begin_index, int end_index, const Node<T>* node,
                          const Q* q, const typename KdTree<T, dim>::DistT r2,
                          const std::vector<T>& points) {
  typedef typename KdTree<T, dim>::DistT DistT;
  if (MinDist2<Q, T, dim, DistT>(q, node_box) >= r2)
    return;
  else if (MaxDist2<Q, T, dim, DistT>(q, node_box) < r2) {
    // node is entirely inside r-ball centered at q
    for (int i = begin_index; i < end_index; i++) results.push_back(i);
  } else if (!node) {  // leaf node
    for (int i = begin_index; i < end_index; i++) {
      DistT dist2 = PointDistance2<Q, T, DistT, dim>(q, &points[i * dim]);
      if (dist2 < r2) results.push_back(i);
    }
  } else {
    T split_value = node->split_value;
    int split_dim = node->split_dim;
    int split_index = node->split_index;

    // examine left child node
    if (split_index > begin_index) {  // left child is non-empty
      T temp = node_box.max(split_dim);
      node_box.max(split_dim) = split_value;
      RNearNeighborsHelper<T, Q, dim>(results, node_box, begin_index,
                                      split_index, node->left, q, r2, points);
      node_box.max(split_dim) = temp;
    }

    // examine right child node
    if (split_index < end_index) {  // right child is non-empty
      T temp = node_box.min(split_dim);
      node_box.min(split_dim) = split_value;
      RNearNeighborsHelper<T, Q, dim>(results, node_box, split_index, end_index,
                                      node->right, q, r2, points);
      node_box.min(split_dim) = temp;
    }
  }
}

template <typename T, typename Q, int dim>
void RNearNeighborsHelper(Indices& results, int begin_index, int end_index,
                          int node_index, Box<T, dim>& node_box, const Q* q,
                          const typename KdTree<T, dim>::DistT r2,
                          const std::vector<SmallNode<T> >& nodes,
                          const std::vector<T>& points) {
  typedef typename KdTree<T, dim>::DistT DistT;
  if (MinDist2<Q, T, dim, DistT>(q, node_box) >= r2)
    return;
  else if (MaxDist2<Q, T, dim, DistT>(q, node_box) < r2) {
    // node is entirely inside r-ball centered at q
    for (int i = begin_index; i < end_index; i++) results.push_back(i);
  } else if (node_index == -1) {  // used to indicate leaf node
    for (int i = begin_index; i < end_index; i++) {
      DistT dist2 = PointDistance2<Q, T, DistT, dim>(q, &points[i * dim]);
      if (dist2 < r2) results.push_back(i);
    }
  } else {  // node_index != -1
    const SmallNode<T>& node = nodes[node_index];
    int split_dim = node.GetSplitDim();
    int split_index = node.GetSplitIndex();
    int left_index = node.LeftChildIndex(node_index);
    int right_index = node.RightChildIndex(node_index);
    // examine left child node
    if (split_index > begin_index) {  // left child is non-empty
      T temp = node_box.max(split_dim);
      node_box.max(split_dim) = node.split_value;
      RNearNeighborsHelper<T, Q, dim>(results, begin_index, split_index,
                                      left_index, node_box, q, r2, nodes,
                                      points);
      node_box.max(split_dim) = temp;
    }

    // examine right child node
    if (split_index < end_index) {  // right child is non-empty
      T temp = node_box.min(split_dim);
      node_box.min(split_dim) = node.split_value;
      RNearNeighborsHelper<T, Q, dim>(results, split_index, end_index,
                                      right_index, node_box, q, r2, nodes,
                                      points);
      node_box.min(split_dim) = temp;
    }
  }
}

template <typename T>
void SerializeHelper(std::vector<SmallNode<T> >& buf, int node_index,
                     const Node<T>* node) {
  // assumes current node is not a leaf, therefore node != NULL
  int child_offset = 0;
  bool has_left = node->left != NULL;
  bool has_right = node->right != NULL;
  if (has_left && has_right) {
    child_offset = (int)buf.size() - node_index;
    buf.push_back(SmallNode<T>());
    buf.push_back(SmallNode<T>());
  } else if (has_left || has_right) {  // exactly one non-NULL child
    child_offset = (int)buf.size() - node_index;
    buf.push_back(SmallNode<T>());
  }

  buf[node_index] =
      SmallNode<T>(node->split_value, node->split_dim, node->split_index,
                   child_offset, has_left, has_right);
  int child_index = node_index + child_offset;
  if (has_left) SerializeHelper(buf, child_index++, node->left);
  if (has_right) SerializeHelper(buf, child_index, node->right);
}

#ifdef USE_TBB
template <typename T, int dim>
class BuildTask : public tbb::task {
 public:
  typedef typename Accumulator<T>::Type DistT;
  typedef Node<T> NodeT;
  BuildTask(NodeT*& node, int begin_index, int end_index,
            std::vector<int>& indices, const Box<T, dim>& node_box,
            const T* points, int num_points,
            const pointkd::BuildParams& build_params)
      : node_(node),
        node_box_(node_box),  // makes copy of node_box
        begin_index_(begin_index),
        end_index_(end_index),
        indices_(indices),
        points_(points),
        num_points_(num_points),
        build_params_(build_params) {}

  NodeT* get_node() const { return node_; }

  tbb::task* execute() {
    // assume node_ will not be a leaf
    // and that node_ has already been empty-trimmed
    int node_size = end_index_ - begin_index_;
    if (node_size < build_params_.serial_cutoff) {
      // perform serial k-d tree construction
      node_ = RecursiveBuildHelper<T, dim>(begin_index_, end_index_, indices_,
                                           node_box_, points_, num_points_,
                                           build_params_);
      return NULL;
    } else {
      NodeT* current_node = NULL;
      node_ = MakeNode<T, dim>(current_node, begin_index_, end_index_, indices_,
                               node_box_, points_, num_points_, build_params_);
      // note: if current_node != NULL, then node_box_ is its extent.

      if (!current_node) return NULL;

      T split_value = current_node->split_value;
      int split_index = current_node->split_index;
      int split_dim = current_node->split_dim;

      // create left task
      BuildTask<T, dim>* left_task = NULL;
      if (split_index > begin_index_) {
        left_task = new (tbb::task::allocate_child()) BuildTask<T, dim>(
            current_node->left, begin_index_, split_index, indices_, node_box_,
            points_, num_points_, build_params_);
        left_task->node_box_.max(split_dim) = split_value;
      }

      // create right task
      BuildTask<T, dim>* right_task = NULL;
      if (end_index_ > split_index) {
        right_task = new (tbb::task::allocate_child()) BuildTask<T, dim>(
            current_node->right, split_index, end_index_, indices_, node_box_,
            points_, num_points_, build_params_);
        right_task->node_box_.min(split_dim) = split_value;
      }

      // spawn tasks
      if (left_task && right_task) {
        set_ref_count(3);
        spawn(*right_task);
        spawn_and_wait_for_all(*left_task);
      } else if (right_task) {  // left empty
        set_ref_count(2);
        spawn_and_wait_for_all(*right_task);
      } else {  // right empty
        set_ref_count(2);
        spawn_and_wait_for_all(*left_task);
      }

      return NULL;
    }
  }

 private:
  NodeT*& node_;
  Box<T, dim> node_box_;
  int begin_index_;
  int end_index_;
  std::vector<int>& indices_;
  const T* points_;
  int num_points_;
  const pointkd::BuildParams& build_params_;
};  // class BuildTask

template <typename Q, typename T, int dim>
struct KNearestNeighbors_ {
  std::vector<Indices>* results;
  const KdTree<T, dim>* tree;
  const Q* query_points;
  int k;
  typename KdTree<T, dim>::DistT r;
  void operator()(const tbb::blocked_range<int>& range) const {
    for (int i = range.begin(); i < range.end(); i++) {
      tree->KNearestNeighbors((*results)[i], &query_points[i * dim], k, r);
    }
  }
};

template <typename T, int dim>
struct KNearestNeighborsSelf_ {
  std::vector<Indices>* results;
  const KdTree<T, dim>* tree;
  const int* query_indices;
  int k;
  typename KdTree<T, dim>::DistT r;
  void operator()(const tbb::blocked_range<int>& range) const {
    for (int i = range.begin(); i < range.end(); i++) {
      tree->KNearestNeighborsSelf((*results)[i], query_indices[i], k, r);
    }
  }
};

template <typename Q, typename T, int dim>
struct RNearNeighbors_ {
  std::vector<Indices>* results;
  const KdTree<T, dim>* tree;
  const Q* query_points;
  typename KdTree<T, dim>::DistT r;
  void operator()(const tbb::blocked_range<int>& range) const {
    for (int i = range.begin(); i < range.end(); i++) {
      tree->RNearNeighbors((*results)[i], &query_points[i * dim], r);
    }
  }
};

template <typename T, int dim>
struct RNearNeighborsSelf_ {
  std::vector<Indices>* results;
  const KdTree<T, dim>* tree;
  const int* query_indices;
  typename KdTree<T, dim>::DistT r;
  void operator()(const tbb::blocked_range<int>& range) const {
    for (int i = range.begin(); i < range.end(); i++) {
      tree->RNearNeighborsSelf((*results)[i], query_indices[i], r);
    }
  }
};
#endif  // USE_TBB

template <typename T, int dim>
void BuildTree(Node<T>*& root, Box<T, dim>& bounding_box,
               std::vector<T>& reordered_points,
               std::vector<int>& forward_indices,
               std::vector<int>& reverse_indices, const T* points,
               int num_points, const pointkd::BuildParams& build_params) {
  // only record indices of points that are non-inf and non-nan
  std::vector<int> indices;
  ValidPointIndices<T>(indices, points, num_points, dim);
  std::size_t num_valid_points = indices.size();
  bounding_box = Box<T, dim>(points, indices);

#ifdef USE_TBB
  if (build_params.num_proc == 1) {
    root = RecursiveBuildHelper<T, dim>(0, (int)num_valid_points, indices,
                                        bounding_box, points, num_points,
                                        build_params);
  } else {
    BuildTask<T, dim>& root_task =
        *new (tbb::task::allocate_root())
            BuildTask<T, dim>(root, 0, (int)num_valid_points, indices,
                              bounding_box, points, num_points, build_params);
    tbb::task::spawn_root_and_wait(root_task);
  }
#else
  root = RecursiveBuildHelper<T, dim>(0, (int)num_valid_points, indices,
                                      bounding_box, points, num_points,
                                      build_params);
#endif  // USE_TBB

  // reorder points
  reordered_points.resize(indices.size() * dim);
  for (std::size_t i = 0; i < num_valid_points; i++) {
    for (std::size_t j = 0; j < (std::size_t)dim; j++)
      reordered_points[i * dim + j] = points[indices[i] * dim + j];
  }

  // the i-th original point is now the forward_indices[i]-th point
  forward_indices.resize(num_points, -1);
  for (std::size_t i = 0; i < num_valid_points; i++) {
    forward_indices[(std::size_t)indices[i]] = (int)i;
  }

  // the i-th point was originally the reverse_indices[i]-th point
  reverse_indices.swap(indices);  // swapping more efficient than copy
}
}  // namespace impl

template <typename T, int dim>
KdTree<T, dim>::KdTree(const std::vector<T>& points,
                       const pointkd::BuildParams build_params) {
  impl::BuildTree(root_, bounding_box_, points_, indices_, reverse_indices_,
                  &points[0], (int)(points.size() / dim), build_params);
}

template <typename T, int dim>
KdTree<T, dim>::KdTree(const T* points, int num_points,
                       const pointkd::BuildParams build_params) {
  impl::BuildTree(root_, bounding_box_, points_, indices_, reverse_indices_,
                  points, num_points, build_params);
}

template <typename T, int dim>
KdTree<T, dim>::KdTree(const KdTree<T, dim>& other)
    : root_(impl::CopyConstructorHelper(other.root_)),
      bounding_box_(other.bounding_box_),
      points_(other.points_),
      indices_(other.indices_),
      reverse_indices_(other.reverse_indices_),
      small_nodes_(other.small_nodes_) {}

template <typename T, int dim>
KdTree<T, dim>::~KdTree() {
  impl::DestructorHelper(root_);
}

template <typename T, int dim>
KdTree<T, dim>& KdTree<T, dim>::operator=(const KdTree<T, dim>& other) {
  if (this != &other) {
    impl::DestructorHelper(root_);
    root_ = impl::CopyConstructorHelper(other.root());
    bounding_box_ = other.bounding_box_;
    points_ = other.points_;
    indices_ = other.indices_;
    reverse_indices_ = other.reverse_indices_;
    small_nodes_ = other.small_nodes_;
  }
  return *this;
}

template <typename T, int dim>
const std::vector<SmallNode<T> >& KdTree<T, dim>::SmallNodes() {
  // serialize tree into an array of small nodes
  if (root_ != NULL && small_nodes_.empty()) {
    small_nodes_.push_back(SmallNode<T>());
    impl::SerializeHelper(small_nodes_, 0, root_);
  }
  return small_nodes_;
}

template <typename T, int dim>
template <typename Q>
void KdTree<T, dim>::KNearestNeighbors(Indices& results, const Q* query_point,
                                       int k, DistT r) const {
  results.clear();
  if (k < 1 || r < (DistT)0.0) return;
  PriorityQueue nearest_neighbors;
  Box<T, dim> node_box(bounding_box_);
  if (MinDist2<Q, T, dim, DistT>(query_point, node_box) >= r * r) return;
  if (small_nodes_.empty())
    impl::KNearestNeighborsHelper<T, Q, dim>(nearest_neighbors, node_box, 0,
                                             (int)points_.size() / dim, root_,
                                             query_point, k, r * r, points_);
  else {
    impl::KNearestNeighborsHelper<T, Q, dim>(
        nearest_neighbors, node_box, 0, (int)points_.size() / dim,
        small_nodes_.empty() ? -1 : 0, query_point, k, r * r, small_nodes_,
        points_);
  }
  // pop out nearest neighbors and insert into results array,
  // at the same time, re-map result indices using reverse_indices_
  std::size_t num_nearest_neighbors = nearest_neighbors.size();
  results.resize(num_nearest_neighbors);
  for (int i = (int)num_nearest_neighbors - 1; i >= 0; i--) {
    results[i] = reverse_indices_[nearest_neighbors.top().index];
    nearest_neighbors.pop();
  }
}

template <typename T, int dim>
template <typename Q>
void KdTree<T, dim>::KNearestNeighbors(std::vector<Indices>& results,
                                       const std::vector<Q>& queries, int k,
                                       DistT r) const {
  KNearestNeighbors(results, &queries[0], queries.size() / dim, k, r);
}

template <typename T, int dim>
template <typename Q>
void KdTree<T, dim>::KNearestNeighbors(std::vector<Indices>& results,
                                       const Q* queries, int num_queries, int k,
                                       DistT r) const {
  results.resize(num_queries);
#ifdef USE_TBB
  impl::KNearestNeighbors_<Q, T, dim> body;
  body.results = &results;
  body.tree = this;
  body.query_points = queries;
  body.k = k;
  body.r = r;
  tbb::parallel_for(tbb::blocked_range<int>(0, num_queries), body);
#else
  for (int i = 0; i < num_queries; i++) {
    KNearestNeighbors(results[i], &queries[i * dim], k, r);
  }
#endif  // USE_TBB
}

template <typename T, int dim>
void KdTree<T, dim>::KNearestNeighborsSelf(Indices& results, int query_index,
                                           int k, DistT r) const {
  results.clear();
  // ensure query_index refers to a valid point (i.e. not inf/nan)
  if (indices_[query_index] < 0) return;
  // get pointer to point referred to by query_indices[i]
  const T* query_point = &points_[indices_[query_index] * dim];
  // find k + 1 nearest neighbors: query point itself + k other neighbors
  KNearestNeighbors(results, query_point, k + 1, r);
  // keep up to k nearest neighbors, while removing the query index if
  // found in the search results
  Indices temp;
  for (std::size_t i = 0; temp.size() < k && i < results.size(); i++) {
    if (results[i] != query_index) temp.push_back(results[i]);
  }
  results.swap(temp);
}

template <typename T, int dim>
void KdTree<T, dim>::KNearestNeighborsSelf(std::vector<Indices>& results,
                                           const Indices& query_indices, int k,
                                           DistT r) const {
  KNearestNeighborsSelf(results, &query_indices[0], query_indices.size(), k, r);
}

template <typename T, int dim>
void KdTree<T, dim>::KNearestNeighborsSelf(std::vector<Indices>& results,
                                           const int* query_indices,
                                           int num_queries, int k,
                                           DistT r) const {
  results.resize(num_queries);
#ifdef USE_TBB
  impl::KNearestNeighborsSelf_<T, dim> body;
  body.results = &results;
  body.tree = this;
  body.query_indices = query_indices;
  body.k = k;
  body.r = r;
  tbb::parallel_for(tbb::blocked_range<int>(0, num_queries), body);
#else
  for (int i = 0; i < num_queries; i++) {
    KNearestNeighborsSelf(results[i], query_indices[i], k, r);
  }
#endif  // USE_TBB
}

template <typename T, int dim>
template <typename Q>
void KdTree<T, dim>::RNearNeighbors(Indices& results, const Q* query_point,
                                    DistT r) const {
  results.clear();
  if (r < (DistT)0.0) return;
  Box<T, dim> node_box(bounding_box_);
  if (small_nodes_.empty())
    impl::RNearNeighborsHelper<T, Q, dim>(results, node_box, 0,
                                          (int)points_.size() / dim, root_,
                                          query_point, r * r, points_);
  else
    impl::RNearNeighborsHelper<T, Q, dim>(
        results, 0, (int)points_.size() / dim, small_nodes_.empty() ? -1 : 0,
        node_box, query_point, r * r, small_nodes_, points_);

  // re-map result indices using reverse_indices_
  for (std::size_t i = 0; i < results.size(); i++) {
    results[i] = reverse_indices_[results[i]];
  }
}

template <typename T, int dim>
template <typename Q>
void KdTree<T, dim>::RNearNeighbors(std::vector<Indices>& results,
                                    const std::vector<Q>& queries,
                                    DistT r) const {
  RNearNeighbors(results, &queries[0], queries.size() / dim, r);
}

template <typename T, int dim>
template <typename Q>
void KdTree<T, dim>::RNearNeighbors(std::vector<Indices>& results,
                                    const Q* queries, int num_queries,
                                    DistT r) const {
  results.resize(num_queries);
#ifdef USE_TBB
  impl::RNearNeighbors_<Q, T, dim> body;
  body.results = &results;
  body.tree = this;
  body.query_points = queries;
  body.r = r;
  tbb::parallel_for(tbb::blocked_range<int>(0, num_queries), body);
#else
  for (int i = 0; i < num_queries; i++) {
    RNearNeighbors(results[i], &queries[i * dim], r);
  }
#endif  // USE_TBB
}

template <typename T, int dim>
void KdTree<T, dim>::RNearNeighborsSelf(Indices& results, int query_index,
                                        DistT r) const {
  results.clear();
  // ensure query_index refers to a valid point (i.e. not inf/nan)
  if (indices_[query_index] < 0) return;
  // get pointer to point referred to by query_indices[i]
  const T* query_point = &points_[indices_[query_index] * dim];
  RNearNeighbors(results, query_point, r);
  // remove query_index from search results
  Indices temp;
  for (std::size_t i = 0; i < results.size(); i++) {
    if (results[i] != query_index) temp.push_back(results[i]);
  }
  results.swap(temp);
}

template <typename T, int dim>
void KdTree<T, dim>::RNearNeighborsSelf(std::vector<Indices>& results,
                                        const Indices& query_indices,
                                        DistT r) const {
  RNearNeighborsSelf(results, &query_indices[0], (int)query_indices.size(), r);
}

template <typename T, int dim>
void KdTree<T, dim>::RNearNeighborsSelf(std::vector<Indices>& results,
                                        const int* query_indices,
                                        int num_queries, DistT r) const {
  results.resize(num_queries);
#ifdef USE_TBB
  impl::RNearNeighborsSelf_<T, dim> body;
  body.results = &results;
  body.tree = this;
  body.query_indices = query_indices;
  body.r = r;
  tbb::parallel_for(tbb::blocked_range<int>(0, num_queries), body);
#else
  for (int i = 0; i < num_queries; i++) {
    RNearNeighborsSelf(results[i], query_indices[i], r);
  }
#endif  // USE_TBB
}
}  // namespace pointkd

#endif  // __KDTREE_IMPL_H__
