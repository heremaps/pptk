#ifndef __TRAVERSERBBF_H__
#define __TRAVERSERBBF_H__

#include "Traverser.h"
#include "vltools/shortvecs.h"
#include "vltools/pq.h"
#include <limits>

//#define PROFILE_TRAVERSAL

namespace pointkd {

template <typename T> class PointKdTree;

template <typename T>
class TraverserBBF : public Traverser<T> {
	using Traverser<T>::_tree;
public:
	typedef typename Traverser<T>::ElementType ElementType;
	typedef typename Traverser<T>::DistanceType DistanceType;
	typedef typename Traverser<T>::Node Node;
	typedef typename Traverser<T>::KNearestQueue KNearestQueue;

	// struct holding node pointer and its extent
	struct NodeBoxType {
		NodeBoxType () : node(NULL), box(vltools::make_float3(0,0,0)),
			dist(std::numeric_limits<DistanceType>::max()) {}
		NodeBoxType (
			const Node * node,
			const vltools::float3 & box,
			const DistanceType dist) :
			node(node), box(box), dist(dist) {}
		bool operator< (const NodeBoxType & rhs) const {
			return this->dist < rhs.dist;
		}
		bool operator> (const NodeBoxType & rhs) const {
			return this->dist > rhs.dist;
		}
		const Node * node;
		vltools::float3 box;
		DistanceType dist;
	};

	typedef vltools::PriorityQueue<NodeBoxType,
		std::vector<NodeBoxType>, std::greater<NodeBoxType> > NodeQueue;

	TraverserBBF(const PointKdTree<T> & tree) :
		Traverser<T>(tree),
		_numLeafsVisited(0),
		_numInnerVisited(0),
		_numPointsChecked(0) { }

	virtual void nearest(
		const std::vector<ElementType> & queries,
		const std::size_t k,
		std::vector<int> & nearestNeighbors,
		std::vector<DistanceType> & nearestDistances)
	{
		if (k == 0)
			return;

		std::size_t numQueries = queries.size() / _tree.getDim();
		nearestNeighbors.resize (k * numQueries, -1);
		nearestDistances.resize (k * numQueries,
			std::numeric_limits<DistanceType>::max());

		#ifdef PROFILE_TRAVERSAL
		_numLeafsVisited = 0;
		_numInnerVisited = 0;
		_numPointsChecked = 0;
		#endif
		for (std::size_t i = 0; i < numQueries; i++) {
			KNearestQueue bestNhbrs(k);
			NodeQueue bestLeafs;
			bestLeafs.push(NodeBoxType(_tree.getRoot(), 
				vltools::make_float3(0,0,0), 0.0));
			while (bestNhbrs.size() < k || !bestLeafs.empty() &&
				bestLeafs.top().dist < bestNhbrs.top().first) {
				// get next node to descend from
				NodeBoxType next = bestLeafs.top();
				bestLeafs.pop();

				// descend
				nearestHelper (
					bestNhbrs,
					bestLeafs,
					next.box, next.dist,
					next.node, &queries[i * _tree.getDim()], k);
			}

			int * nhbrs = &nearestNeighbors[i * k];
			DistanceType * dists = &nearestDistances[i * k];
			for (std::size_t j = 0; !bestNhbrs.empty(); j++) {
				nhbrs[j] = bestNhbrs.top().second;
				dists[j] = bestNhbrs.top().first;
				bestNhbrs.pop();
			}
		}
		#ifdef PROFILE_TRAVERSAL
		std::cout << "leafs: " << _numLeafsVisited << std::endl;
		std::cout << "inner: " << _numInnerVisited << std::endl;
		std::cout << "check: " << _numPointsChecked << std::endl;
		#endif
	}

	virtual void range(
		const std::vector<ElementType> & queries,
		const DistanceType r,
		std::vector<int> & rNeighbors,
		std::vector<DistanceType> & neighborDistances) {
		
	}
private:
	void nearestHelper (
		KNearestQueue & bestNeighbors,
		NodeQueue & bestLeafs,
		const vltools::float3 & nodeDistVec,
		const DistanceType nodeDist,
		const Node * node,
		const ElementType * query,
		const std::size_t k)
	{
		if (node->isLeaf()) {
			#ifdef PROFILE_TRAVERSAL
			_numPointsChecked += (node->endIndex-node->beginIndex);
			_numLeafsVisited ++;
			#endif
			addPointsToQueue (bestNeighbors, node, query);
		} else {
			#ifdef PROFILE_TRAVERSAL
			_numInnerVisited ++;
			#endif
			Node * near, * far;
			if (query[node->splitDim] < node->splitVal) {
				near = node->left;
				far = node->right;
			} else {
				near = node->right;
				far = node->left;
			}

			// push far node onto bestLeafs
			if (far != NULL) {
				DistanceType a = nodeDistVec[node->splitDim];
				DistanceType b = query[node->splitDim] - node->splitVal;
				b *= b;
				DistanceType farDist = nodeDist - a + b;
				if (bestNeighbors.empty() ||
					farDist < bestNeighbors.top().first) {
					vltools::float3 farNodeDistVec = nodeDistVec;
					farNodeDistVec[node->splitDim] = b;
					bestLeafs.push(
						NodeBoxType(far,farNodeDistVec,farDist));
				}
			}

			// descend down near node
			if (near != NULL) {
				nearestHelper(
					bestNeighbors,
					bestLeafs,
					nodeDistVec, nodeDist,
					near, query, k);
			}
		}
	}

	void addPointsToQueue (
		KNearestQueue & queue, 
		const Node * leaf,
		const ElementType * query) {
		for (std::size_t i = leaf->beginIndex;
			i < leaf->endIndex; i++) {
			DistanceType dist = 0.0;
			for (std::size_t j = 0; j < _tree.getDim(); j++) {
				DistanceType temp = query[j] -
					_tree.getData()[i * _tree.getDim() + j];
				temp *= temp;
				dist += temp;
			}
			queue.push(std::pair<DistanceType,int>(
				dist,_tree.getIndices()[i]));
		}
	}


protected:
	std::size_t _numLeafsVisited;
	std::size_t _numInnerVisited;
	std::size_t _numPointsChecked;
};

} // namespace pointkd

#endif
