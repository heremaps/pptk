#ifndef __TRAVERSERBATCHED_H__
#define __TRAVERSERBATCHED_H__

// note: this file is only meant to be included from PointKdTree.h
#include "Accumulator.h"
#include "PointKdTree_decl.h"
#include "Traverser.h"
#include "vltools/pq.h"
#include "vltools/box3.h"
#include "tbb/scalable_allocator.h"
#include <limits>
#include <vector>
#include <iostream>
#include <fstream>
#if defined (__INTEL_COMPILER)
#include "xmmintrin.h"
#endif

namespace pointkd {

template <typename T> class PointKdTree;

template <typename T>
class TraverserBatched : public Traverser<T> {
	using Traverser<T>::_tree;
public:
	typedef T ElementType;
	typedef typename Accumulator<T>::Type DistanceType;
	typedef typename PointKdTree<T>::Node Node;
	typedef typename Traverser<T>::NodeDistPair NodeDistPair;
	typedef vltools::Box3<DistanceType> Box;

	struct NodeBoxDist {
		NodeBoxDist () : node(NULL), box(),
			dist(std::numeric_limits<DistanceType>::max()) {}
		NodeBoxDist (
			const Node * node,
			const Box & box,
			const DistanceType dist) :
			node(node), box(box), dist(dist) {}
		bool operator< (const NodeBoxDist & rhs) const {
			return this->dist < rhs.dist;
		}
		bool operator> (const NodeBoxDist & rhs) const {
			return this->dist > rhs.dist;
		}
		const Node * node;
		Box box;
		DistanceType dist;
	};

	typedef vltools::PriorityQueue<NodeDistPair> NodeDistMaxHeap;
	typedef vltools::PriorityQueue<
		NodeBoxDist, std::vector<NodeBoxDist>,
		std::greater<NodeBoxDist> > NodeBoxDistMinHeap;
	typedef vltools::PriorityQueue<
		NodeDistPair, std::vector<NodeDistPair>,
		std::greater<NodeDistPair> > NodeDistMinHeap;

//	struct NodeBoxDistPair {
//		NodeRecord (const Node * node, DistanceType dx,
//			DistanceType dy, DistanceType dz) : node(node) {
//			vec[0] = dx;
//			vec[1] = dy;
//			vec[2] = dz;
//			dist = dx * dx + dy * dy + dz * dz;
//		}
//		DistanceType vec[3];
//	};

	TraverserBatched(const PointKdTree<T> & tree) : 
		Traverser<T>(tree)
	{
		_box = Box();
		_box.addPoints (_tree.getData(), _tree.getNumPoints());
		getLeafNodes(_tree.getRoot(), _box);
	}

	void allNearest (
		const std::size_t k,
		std::vector<int> & nearestNeighbors,
		std::vector<DistanceType> & nearestDistances)
	{
		// allocate result arrays
		std::vector<int> indices(_tree.getNumPoints(), -1);
		nearestNeighbors.resize(k * _tree.getNumPoints(), -1);
		nearestDistances.resize(k * _tree.getNumPoints(), 
			std::numeric_limits<DistanceType>::max());
		int * p_indices = &indices[0];
		int * neighbors = &nearestNeighbors[0];
		DistanceType * distances = &nearestDistances[0];

		// for each leaf in tree
		for (std::size_t i = 0; i < _leafNodes.size(); i++) {
			Box & box = _leafExtents[i];
			_heapH.clear();
			_heapY.clear();
			_heapX.clear();
			//_heapH.push (NodeDistVecPair(_tree->root, 0.0, 0.0, 0.0));
			_heapH.push (NodeBoxDist(_tree.getRoot(), _box, 0.0));
			_heapY.push (NodeDistPair(_leafNodes[i],
				maxDist2(box, box)));
			unsigned int count = _leafNodes[i]->endIndex -
				_leafNodes[i]->beginIndex;
			while (!_heapH.empty()) {
				// get next node
				NodeBoxDist next = _heapH.top(); _heapH.pop();

				// descend to get leaf node
				NodeBoxDist leaf;
				descend (leaf, next, box);
				if (leaf.node == NULL)
					continue;
				DistanceType minD = leaf.dist;
				DistanceType maxD = maxDist2 (box, leaf.box);
				int leafSize = leaf.node->endIndex -
					leaf.node->beginIndex;
				NodeDistPair leafPair(leaf.node, maxD);

				// record leaf node
				_heapX.push(NodeDistPair(leaf.node, minD));

				// 
				if (count < k) {
					// 
					_heapY.push(leafPair);
					count += leafSize;
				} else if (maxD < _heapY.top().dist) {
					// leaf with smaller maxDist found
					_heapY.push(leafPair);
					count += leafSize;
					int sz = _heapY.top().node->endIndex -
						_heapY.top().node->beginIndex;
					if (count - sz >= k) {
						_heapY.pop();
						count -= sz;
					}
				} else if (minD >= _heapY.top().dist) {
					// can safely disregard all other leafs
					break;
				}
			}
			// collect leafs that might contain k-nn
			std::vector<const Node *> nodes;
			nodes.clear();
			while (!_heapX.empty() &&
					_heapX.top().dist <= _heapY.top().dist) {
				nodes.push_back (_heapX.top().node);
				_heapX.pop();
			}

			// compute all-pairs distances
			computeDistances (_leafNodes[i], nodes, k,
				p_indices, neighbors, distances);

			// prepare next iteration
			std::size_t nodeSize = _leafNodes[i]->endIndex -
				_leafNodes[i]->beginIndex;
			p_indices += nodeSize;
			neighbors += nodeSize * k;
			distances += nodeSize * k;
		}

		// reorder distances and neighbors
		std::vector<DistanceType> temp_distances (nearestDistances.size());
		std::vector<int> temp_neighbors (nearestNeighbors.size());
		for (std::size_t i = 0; i < indices.size(); i++) {
			int idx = indices[i];
			for (std::size_t j = 0; j < k; j++) {
				temp_distances[k * idx + j] = nearestDistances[k * i + j];
				temp_neighbors[k * idx + j] = nearestNeighbors[k * i + j];
			}
		}
		nearestDistances.swap(temp_distances);
		nearestNeighbors.swap(temp_neighbors);
	}

	void nearest(
		const std::vector<ElementType> & queries,
		const std::size_t k,
		std::vector<int> & nearestNeighbors,
		std::vector<DistanceType> & nearestDistances)
	{
	}

	void range(
		const std::vector<ElementType> & queries,
		const DistanceType r,
		std::vector<int> & rNeighbors,
		std::vector<DistanceType> & neighborDistances)
	{
	}
private:
	void getLeafNodes(const Node * node, Box box)
	// gets extents and pointers to all leaf nodes
	// assumes _leafExtents and _leafNodes are empty
	{
		if (node == NULL)
			return;
		if (node->isLeaf()) {
			_leafExtents.push_back(box);
			_leafNodes.push_back(node);
		} else {
			DistanceType temp = box.max(node->splitDim);
			box.max(node->splitDim) = node->splitVal;
			getLeafNodes(node->left, box);
			box.max(node->splitDim) = temp;
			temp = box.min(node->splitDim);
			box.min(node->splitDim) = node->splitVal;
			getLeafNodes(node->right, box);
			box.min(node->splitDim) = temp;
		}
	}

	void descend(
		NodeBoxDist & l,
		const NodeBoxDist & n,
		const Box q)
	{
		const Node * node = n.node;
		Box box = n.box;
		DistanceType vec[3];
		minDistVec(vec, box, q);
		DistanceType vec2[3];
		vec2[0] = vec[0] * vec[0];
		vec2[1] = vec[1] * vec[1];
		vec2[2] = vec[2] * vec[2];
		while (node != NULL && !node->isLeaf()) {
			int dim = node->splitDim;
			DistanceType leftMax = node->splitVal;
			DistanceType rightMin = node->splitVal;
			// compute distance to left child
			DistanceType dL = std::max(q.min(dim) - leftMax, 
				(DistanceType)0.0);
			dL = std::max(dL, vec[dim]);
			// compute distance to right child
			DistanceType dR = std::max(rightMin - q.max(dim),
				(DistanceType)0.0);
			dR = std::max(dR, vec[dim]);

			if (dL <= dR) {
				// push right child
				DistanceType delta2 = dR * dR;
				std::swap (box.min(dim), rightMin);
				std::swap (vec2[dim], delta2);
				DistanceType dist2 = vec2[0] + vec2[1] + vec2[2];
				_heapH.push(NodeBoxDist(node->right, box, dist2));
				box.min(dim) = rightMin;
				vec2[dim] = delta2;
				// desceond left child
				box.max(dim) = leftMax;
				node = node->left;
			} else {
				// push left child
				DistanceType delta2 = dL * dL;
				std::swap (box.max(dim), leftMax);
				std::swap (vec2[dim], delta2);
				DistanceType dist2 = vec2[0] + vec2[1] + vec2[2];
				_heapH.push(NodeBoxDist(node->left, box, dist2));
				box.max(dim) = leftMax;
				vec2[dim] = delta2;
				// descond right child
				box.min(dim) = rightMin;
				node = node->right;
			}
		}
		l.node = node;
		l.box = box;
		l.dist = n.dist;
	}

	void computeDistances(
		const Node * node,
		const std::vector<const Node *> & nodes,
		std::size_t k,
		int * indices,	// of query points
		int * neighbors,
		DistanceType * distances)
	{
		// for each query point
		for (std::size_t i = node->beginIndex;
				i < node->endIndex; i++) {
			ElementType q[3];
			q[0] = _tree.getData()[3 * i + 0];
			q[1] = _tree.getData()[3 * i + 1];
			q[2] = _tree.getData()[3 * i + 2];
			int count = 0;
			// for each leaf
			for (std::size_t j = 0; j < nodes.size(); j++) {
				// for each point in leaf
				for (std::size_t l = nodes[j]->beginIndex;
						l < nodes[j]->endIndex; l++) {
					if (nodes[j] == node && l == i)
						continue;	// exclude itself from k-nn
					ElementType p[3];
					p[0] = _tree.getData()[3 * l + 0];
					p[1] = _tree.getData()[3 * l + 1];
					p[2] = _tree.getData()[3 * l + 2];
					DistanceType dist = 0.0;
					DistanceType temp = p[0] - q[0];
					dist += temp * temp;
					temp = p[1] - q[1];
					dist += temp * temp;
					temp = p[2] - q[2];
					dist += temp * temp;

					// insert point index and distance
					int idx;
					if (count < k) {
						idx = count - 1;
						count ++;
					} else if (distances[count - 1] > dist) {
						// point is closer than k-th nearest
						// => replace k-th nearest
						distances[count - 1] = dist;
						neighbors[count - 1] = _tree.getIndices()[l];
						idx = count - 2;
					} else {
						// point is farther than k-th nearest
						// => continue to next point
						continue;
					}
					while (idx >= 0 && distances[idx] > dist) {
						distances[idx + 1] = distances[idx];
						neighbors[idx + 1] = neighbors[idx];
						idx --;
					}

					distances[idx + 1] = dist;
					neighbors[idx + 1] = _tree.getIndices()[l];
				}
			}
			*indices++ = _tree.getIndices()[i];
			neighbors += k;
			distances += k;
		}
	}
	std::vector<Box> _leafExtents; 
	std::vector<const Node *> _leafNodes;
	NodeBoxDistMinHeap _heapH;	// nodes in increasing MinDist order
	NodeDistMaxHeap _heapY;		// nodes in decreasing MaxDist order
	NodeDistMinHeap _heapX;			// nodes in increasing MinDist order
	Box _box;
};

} // namespace pointkd

#endif
