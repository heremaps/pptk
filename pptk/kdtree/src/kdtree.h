/** TODO: license boiler plate here
  *
  * By Victor Lu (victor.1.lu@here.com)
*/

#ifndef __KDTREE_H__
#define __KDTREE_H__

#include <cstdint>
#include <queue>
#include <vector>
#include "accumulator.h"
#include "box.h"
#include "node.h"
#include "small_node.h"

#ifdef USE_TBB
#include "tbb/scalable_allocator.h"
#endif  // USE_TBB

namespace pointkd {
// note to user: do not rely on Indices being equal to vector<int>
#ifdef USE_TBB
typedef std::vector<int, tbb::scalable_allocator<int> > Indices;
#else
typedef std::vector<int> Indices;
#endif

struct BuildParams {
  BuildParams()
      : num_proc(-1),
        serial_cutoff(0),
        max_leaf_size(10),
        empty_split_threshold(0.2) {}
  int num_proc;
  int serial_cutoff;
  // leaf nodes have at most this many points
  int max_leaf_size;
  // perform empty split if resulting gap ratio greater than threshold
  // threshold must be non-negative
  double empty_split_threshold;
};

template <typename T, int dim>
class KdTree {
 public:
  struct Pair;
  typedef typename Accumulator<T>::Type DistT;
  typedef Box<T, dim> BoxT;
  typedef Node<T> NodeT;
#ifdef USE_TBB
  typedef std::vector<Pair, tbb::scalable_allocator<Pair> > Pairs;
#else
  typedef std::vector<Pair> Pairs;
#endif  // USE_TBB
  typedef std::priority_queue<Pair, Pairs> PriorityQueue;

  struct Pair {
    Pair() : index(-1), dist2(0) {}
    Pair(int index, DistT dist2) : index(index), dist2(dist2) {}
    bool operator<(const Pair& other) const { return dist2 < other.dist2; }
    int index;
    DistT dist2;
  };

  KdTree() : root_(NULL) {}

  /**
    * Assumes points stored in array-of-struct format.
  */
  KdTree(const std::vector<T>& points,
         const BuildParams build_params = BuildParams());

  KdTree(const T* points, int num_points,
         const BuildParams build_params = BuildParams());

  KdTree(const KdTree<T, dim>& other);

  ~KdTree();

  KdTree& operator=(const KdTree<T, dim>& other);

  /** Serializes k-d tree nodes into an array of small nodes.
    *
    * Returns a reference to the resulting array.
    * Subsequent queries are performed on nodes in this array.
  */
  const std::vector<SmallNode<T> >& SmallNodes();

  /** Finds k-nearest neighbors to query point.
    *
    * Assumes query_point is a Q[dim] array. Contents of results should be
    * interpreted as 0-based integer indices into the array of points that
    * this k-d tree was originally built on.  Overwrites any existing content
    * in results.
    *
    * Optional argument r is used to require neighbors have distance from
    * query_point strictly less than r.
  */
  template <typename Q>
  void KNearestNeighbors(Indices& results, const Q* query_point, int k,
                         DistT r = inf()) const;

  /** Finds k-nearest neighbors of a set of query points.
    *
    * Assumes query points are stored in an array-of-struct format (i.e. the
    * i-th component of the j-th point is given by queries[i + j * dim]).
    * Contents of results should be interpreted as 0-based integer indices into
    * the array of points that this k-d tree was originally built on.  Clears
    * any existing content in results and resizes it to the number of query
    * points in queries.
    *
    * Optional argument r is used to require neighbors have distance from
    * query_point strictly less than r.
    *
    * By default, queries are performed in parallel using a number of threads
    * determined by Intel TBB. To explicitly set the number of threads call
    * tbb::init_task_scheduler(num_threads) before calling this function.
  */
  template <typename Q>
  void KNearestNeighbors(std::vector<Indices>& results,
                         const std::vector<Q>& queries, int k,
                         DistT r = inf()) const;

  template <typename Q>
  void KNearestNeighbors(std::vector<Indices>& results, const Q* queries,
                         int num_queries, int k, DistT r = inf()) const;

  /** Finds k-nearest neighbors to a point from the original point cloud.
    *
    * Assumes query_index is a 0-based integer index into the array of points
    * that the k-d tree was originally built on.  Contents of results should
    * likewise be interpreted as 0-based integer indices into the original array
    * of points.  Overwrites any existing content in results.  Excludes
    * query_index itself from results.
    *
    * Optional argument r is used to require neighbors have distance from
    * query_point strictly less than r.
  */
  void KNearestNeighborsSelf(Indices& results, int query_index, int k,
                             DistT r = inf()) const;

  /** Finds k-nearest neighbors to points from the original point cloud.
    *
    * Assumes query_indices is a vector of 0-based integer indices into the
    * array of points that the k-d tree was originally built on.  Contents of
    * results should be likewise interpreted as 0-based integer indices into the
    * original array of points.  Clears any existing content in results and
    * resizes it to be the size of query_indices.  Excludes indices in
    * query_indices from results.
    *
    * Optional argument r is used to require neighbors have distance from
    * query_point strictly less than r.
    *
    * By default, queries are performed in parallel using a number of threads
    * determined by Intel TBB. To explicitly set the number of threads call
    * tbb::init_task_scheduler(num_threads) before calling this function.
  */
  void KNearestNeighborsSelf(std::vector<Indices>& results,
                             const Indices& query_indices, int k,
                             DistT r = inf()) const;

  void KNearestNeighborsSelf(std::vector<Indices>& results,
                             const int* query_indices, int num_queries, int k,
                             DistT r = inf()) const;

  /** Finds r-near neighbors to query point
    *
    * Finds points whose distance to query_point are strictly less than r.
    * Assumes query_point is a Q[dim] array.  Replaces any existing content
    * in results with 0-based integer indices into the array of points that the
    * k-d tree was originally built on.
  */
  template <typename Q>
  void RNearNeighbors(Indices& results, const Q* query_point, DistT r) const;

  /** Finds r-near neighbors to a set of query points
    *
    * For each query point in queries, finds points whose distance to the query
    * point is strictly less than r.  Assumes queries organized in an
    * array-of-struct format (i.e. the i-th coordinate of the j-th query point
    * is given by queries[i + j * dim]).  Clears any existing content in results
    * and resizes it to the number of query points.  The contents of results
    * should be interpreted as 0-based integer indices into the array of points
    * that this k-d tree was originally built on.
  */
  template <typename Q>
  void RNearNeighbors(std::vector<Indices>& results,
                      const std::vector<Q>& queries, DistT r) const;

  template <typename Q>
  void RNearNeighbors(std::vector<Indices>& results, const Q* queries,
                      int num_queries, DistT r) const;

  /** Finds r-near neighbors to a point in the original point cloud.
    *
    * Finds points whose distance to the query point is strictly less than r.
    * Uses the "query_index-th" point from the original point cloud as the query
    * point.  Replaces any existing content in results with 0-based integer
    * indices into the original point cloud.
  */
  void RNearNeighborsSelf(Indices& results, int query_index, DistT r) const;

  /** Finds r-near neighbors to points in the original point cloud.
    *
    * For each query point, finds points whose distance to the query point is
    * strictly less than r.  Uses the query_indices[i]-th point from the
    * original point cloud as the i-th query point.  Clears any existing
    * content in results and resizes it to the number of query points.  Fills
    * the i-th item in results with the i-th query point's r-near neighbors,
    * which are represented by 0-based integer indices into the original point
    * cloud.
    *
    * By default, queries are performed in parallel using a number of threads
    * determined by Intel TBB. To explicitly set the number of threads call
    * tbb::init_task_scheduler(num_threads) before calling this function.
  */
  void RNearNeighborsSelf(std::vector<Indices>& results,
                          const Indices& query_indices, DistT r) const;

  void RNearNeighborsSelf(std::vector<Indices>& results,
                          const int* query_indices, int num_queries,
                          DistT r) const;

  // TODO: support slice(begin,end,stride) as query objects?

  static DistT inf() { return std::numeric_limits<DistT>::infinity(); }
  const int dimension() const { return dim; }
  const int num_points() const { return points_.size() / dim; }
  const NodeT* root() const { return root_; }
  const Box<T, dim>& bounding_box() const { return bounding_box_; }
  const std::vector<T>& points() const { return points_; }
  const std::vector<int>& indices() const { return indices_; }
  const std::vector<int>& reverse_indices() const { return reverse_indices_; }

 private:
  NodeT* root_;
  Box<T, dim> bounding_box_;

  // reordered copy of original input points, stored in AOS format
  // this copy will not contain any NaN/inf points from the original input
  std::vector<T> points_;
  // i-th point in the original input now at points_[indices_[i] * dim]
  std::vector<int> indices_;
  // i-th point in points_ came from points[reverse_indices_[i] * dim]
  std::vector<int> reverse_indices_;
  // serialized array of small nodes
  std::vector<SmallNode<T> > small_nodes_;
};
}  // namespace pointkd

#endif  // __KDTREE_H__

#include "kdtree-impl.h"
