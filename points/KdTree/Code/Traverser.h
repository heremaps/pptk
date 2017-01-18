#ifndef __TRAVERSER_H__
#define __TRAVERSER_H__

#include "Accumulator.h"
#include "PointKdTree_decl.h"
#include "vltools/smallestk.h"
#include <vector>
#include <limits>

namespace pointkd {

template <typename T> class PointKdTree;

template <typename T>
class Traverser {
public:
	typedef T ElementType;
	typedef typename Accumulator<T>::Type DistanceType;
	typedef typename PointKdTree<T>::Node Node;
	typedef vltools::SmallestK<std::pair<DistanceType,int> > 
			KNearestQueue;

	struct NodeDistPair {
		NodeDistPair () {}
		NodeDistPair (const Node * node, DistanceType dist) :
			node(node), dist(dist) {}
		const Node * node;
		DistanceType dist;
		bool operator< (const NodeDistPair & rhs) const {
			return this->dist < rhs.dist;
		}
		bool operator> (const NodeDistPair & rhs) const {
			return this->dist > rhs.dist;
		}
		bool operator<= (const NodeDistPair & rhs) const {
			return this->dist <= rhs.dist;
		}
		bool operator>= (const NodeDistPair & rhs) const {
			return this->dist >= rhs.dist;
		}
	};

	Traverser(const PointKdTree<T> & tree) : _tree(tree) { }

	virtual void nearest(
		const std::vector<ElementType> & queries,
		const std::size_t k,
		std::vector<int> & nearestNeighbors,
		std::vector<DistanceType> & nearestDistances,
		const DistanceType initialMaxDist = 
			std::numeric_limits<DistanceType>::max()) = 0;

	virtual void range(
		const std::vector<ElementType> & queries,
		const DistanceType r,
		std::vector<int> & rNeighbors,
		std::vector<DistanceType> & neighborDistances) = 0;

protected:
	const PointKdTree<T> & _tree;
};

} // namespace pointkd

#endif
