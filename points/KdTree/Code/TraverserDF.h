#ifndef __TRAVERSERDF_H__
#define __TRAVERSERDF_H__

#include "Traverser.h"
#include "ResultSet.h"

//#define PROFILE_TRAVERSAL

namespace pointkd {

template <typename T> class PointKdTree;

template <typename T>
class TraverserDF : public Traverser<T> {
	using Traverser<T>::_tree;
public:
	typedef typename Traverser<T>::ElementType ElementType;
	typedef typename Traverser<T>::DistanceType DistanceType;
	typedef typename Traverser<T>::Node Node;
	typedef typename Traverser<T>::KNearestQueue KNearestQueue;

	TraverserDF(const PointKdTree<T> & tree) :
		Traverser<T>(tree),
		_numLeafsVisited(0),
		_numInnerVisited(0), 
		_numPointsChecked(0) { }

	virtual void nearest(
		const std::vector<ElementType> & queries,
		const std::size_t k,
		std::vector<int> & nearestNeighbors,
		std::vector<DistanceType> & nearestDistances,
		const float initialMaxDist = 
			std::numeric_limits<DistanceType>::max())
	{
		std::size_t numQueries = queries.size() / _tree.getDim();
		nearestNeighbors.resize (k * numQueries);
		nearestDistances.resize (k * numQueries);
		std::fill(nearestNeighbors.begin(), nearestNeighbors.end(), -1);
		std::fill(nearestDistances.begin(), nearestDistances.end(),
			initialMaxDist);

		#ifdef PROFILE_TRAVERSAL
		_numLeafsVisited = 0;
		_numInnerVisited = 0;
		_numPointsChecked = 0;
		#endif
		#if 0
		KNearestQueue queue(k);
		std::vector<DistanceType> distVec (_tree.getDim(), 0.0);
		const ElementType * query = &queries[i * _tree.getDim()];
		nearestHelper (
			queue, distVec,
			_tree.getRoot(), query, k, 0.0);

		int * nhbrs = &nearestNeighbors[i * k];
		DistanceType * dists = &nearestDistances[i * k];
		for (std::size_t j = 0; !queue.empty(); j++) {
			nhbrs[j] = queue.top().second;
			dists[j] = queue.top().first;
			queue.pop();
		}
		#else
		if (k == 1) {
      // TODO: Parallelize here
		for (std::size_t i = 0; i < numQueries; i++) {
			std::vector<DistanceType> distVec (_tree.getDim(), 0.0);
			const ElementType * query = &queries[i * _tree.getDim()];
			oneNearestHelper (
				nearestNeighbors[i], nearestDistances[i], distVec,
				_tree.getRoot(), query, 0.0);
		}
		} else if (k < 250) {
		for (std::size_t i = 0; i < numQueries; i++) {
			ResultSetLinearInsert<DistanceType> resultSet(
				k, initialMaxDist);
			std::vector<DistanceType> distVec (_tree.getDim(), 0.0);
			const ElementType * query = &queries[i * _tree.getDim()];
			nearestHelper (
				resultSet, distVec,
				_tree.getRoot(), query, k, 0.0);

			int * nhbrs = &nearestNeighbors[i * k];
			DistanceType * dists = &nearestDistances[i * k];
			resultSet.copy(dists,nhbrs);
			for (std::size_t j = 0; j < k; j++) {
				if (nhbrs[j] == -1) break;
				nhbrs[j] = _tree.getIndices()[nhbrs[j]];
			}
		} 
		} else {
		for (std::size_t i = 0; i < numQueries; i++) {
			ResultSetHeap<DistanceType> resultSet(k, initialMaxDist);
			std::vector<DistanceType> distVec (_tree.getDim(), 0.0);
			const ElementType * query = &queries[i * _tree.getDim()];
			nearestHelper (
				resultSet, distVec,
				_tree.getRoot(), query, k, 0.0);

			int * nhbrs = &nearestNeighbors[i * k];
			DistanceType * dists = &nearestDistances[i * k];
			resultSet.copy(dists,nhbrs);
			for (std::size_t j = 0; j < k; j++) {
				if (nhbrs[j] == -1) break;
				nhbrs[j] = _tree.getIndices()[nhbrs[j]];
			}
		}
		}
		#endif
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

	void nearestSelf(
		const std::vector<int> & indices,
		const std::size_t k,
		std::vector<int> & nearestNeighbors,
		std::vector<DistanceType> & nearestDistances,
		const float initialMaxDist = 
			std::numeric_limits<DistanceType>::max())
	{
		std::size_t numQueries = indices.size();
		nearestNeighbors.resize (k * numQueries);
		nearestDistances.resize (k * numQueries);
		std::fill(nearestNeighbors.begin(), nearestNeighbors.end(), -1);
		std::fill(nearestDistances.begin(), nearestDistances.end(),
			initialMaxDist);
			
		ResultSet<DistanceType> * resultSet;
		if (k < 250) {
			resultSet = new ResultSetLinearInsert<DistanceType>(k, initialMaxDist);
		} else {
			resultSet = new ResultSetHeap<DistanceType>(k, initialMaxDist);
		}
		
		for (std::size_t i = 0; i < numQueries; i++) {
			resultSet->clear();
			std::vector<DistanceType> distVec (_tree.getDim(), 0.0);
			int queryIndex = _tree.getReverseIndices()[indices[i]];
			const ElementType * query = _tree.getData() + queryIndex * _tree.getDim();
			nearestHelper (
				*resultSet, distVec,
				_tree.getRoot(), query, k, 0.0, queryIndex);

			int * nhbrs = &nearestNeighbors[i * k];
			DistanceType * dists = &nearestDistances[i * k];
			resultSet->copy(dists,nhbrs);
			for (std::size_t j = 0; j < k; j++) {
				if (nhbrs[j] == -1) break;
				nhbrs[j] = _tree.getIndices()[nhbrs[j]];
			}
		}
		delete resultSet;
	}
	
private:
	void nearestHelper (
		KNearestQueue & bestNeighbors,
		std::vector<DistanceType> & nodeDistVec,
		const Node * node,
		const ElementType * query,
		const std::size_t k,
		const DistanceType nodeDist)
	{
		if (node->isLeaf()) {
			#ifdef PROFILE_TRAVERSAL
			_numPointsChecked += (node->endIndex-node->beginIndex);
			_numLeafsVisited ++;
			#endif
			addPointsToQueue(bestNeighbors,node,query);
		} else {
			#ifdef PROFILE_TRAVERSAL
			_numInnerVisited ++;
			#endif
			Node * nearNode, * farNode;
			if (query[node->splitDim] < node->splitVal) {
				nearNode = node->left;
				farNode = node->right;
			} else {
				nearNode = node->right;
				farNode = node->left;
			}

			// check near node
			if (nearNode != NULL) {
				nearestHelper (
					bestNeighbors, nodeDistVec,
					nearNode, query, k, nodeDist);
			}

			// check far node
			if (farNode != NULL) {
				DistanceType d = nodeDistVec[node->splitDim];
				DistanceType distToSplit = query[node->splitDim] -
					node->splitVal;
				distToSplit *= distToSplit;
				DistanceType distToFar = nodeDist - d + distToSplit;
				nodeDistVec[node->splitDim] = distToSplit;
				if (bestNeighbors.empty() ||
					distToFar < bestNeighbors.top().first) {
					nearestHelper (
						bestNeighbors,
						nodeDistVec, farNode, query, k, distToFar);
				}
				nodeDistVec[node->splitDim] = d; 
			}
		}	// if inner
	} // void nearestHelper

	void nearestHelper (
		ResultSet<DistanceType> & resultSet,
		std::vector<DistanceType> & nodeDistVec,
		const Node * node,
		const ElementType * query,
		const std::size_t k,
		const DistanceType nodeDist,
		const int queryIndex = -1)
	{
		if (node->isLeaf()) {
			#ifdef PROFILE_TRAVERSAL
			_numPointsChecked += (node->endIndex-node->beginIndex);
			_numLeafsVisited ++;
			#endif
			std::size_t dim = _tree.getDim();
			const ElementType * data = _tree.getData();
			for (std::size_t i = node->beginIndex;
				i < node->endIndex; i++) {
				if (i == queryIndex) continue;
				DistanceType dist = 0.0;
				const ElementType * p = data + i * dim; 
				for (std::size_t j = 0; j < dim; j++) {
					DistanceType temp = query[j] - p[j];
					temp *= temp;
					dist += temp;
				}
				resultSet.addPoint(dist,i);
			}
		} else {
			#ifdef PROFILE_TRAVERSAL
			_numInnerVisited ++;
			#endif
			Node * nearNode, * farNode;
			if (query[node->splitDim] < node->splitVal) {
				nearNode = node->left;
				farNode = node->right;
			} else {
				nearNode = node->right;
				farNode = node->left;
			}

			// check near node
			if (nearNode != NULL) {
				nearestHelper (
					resultSet, nodeDistVec,
					nearNode, query, k, nodeDist, queryIndex);
			}

			// check far node
			if (farNode != NULL) {
				DistanceType d = nodeDistVec[node->splitDim];
				DistanceType distToSplit = query[node->splitDim] -
					node->splitVal;
				distToSplit *= distToSplit;
				DistanceType distToFar = nodeDist - d + distToSplit;
				nodeDistVec[node->splitDim] = distToSplit;
				if (distToFar < resultSet.worstDist()) {
					nearestHelper (
						resultSet,
						nodeDistVec, farNode, query, k, distToFar, queryIndex);
				}
				nodeDistVec[node->splitDim] = d; 
			}
		}	// if inner
	} // void nearestHelper
	
	void oneNearestHelper (
		int & nearestNeighbor,
		DistanceType & nearestDistance,
		std::vector<DistanceType> & nodeDistVec,
		const Node * node,
		const ElementType * query,
		const DistanceType nodeDist)
	{
		if (node->isLeaf()) {
			#ifdef PROFILE_TRAVERSAL
			_numPointsChecked += (node->endIndex-node->beginIndex);
			_numLeafsVisited ++;
			#endif
			for (std::size_t i = node->beginIndex;
				i < node->endIndex; i++) {
				DistanceType dist = 0.0;
				for (std::size_t j = 0; j < _tree.getDim(); j++) {
					DistanceType temp = query[j] -
						_tree.getData()[i * _tree.getDim() + j];
					temp *= temp;
					dist += temp;
				}
				if (dist < nearestDistance) {
					nearestDistance = dist;
					nearestNeighbor = _tree.getIndices()[i];
				}
			}

		} else {
			#ifdef PROFILE_TRAVERSAL
			_numInnerVisited ++;
			#endif
			Node * nearNode, * farNode;
			if (query[node->splitDim] < node->splitVal) {
				nearNode = node->left;
				farNode = node->right;
			} else {
				nearNode = node->right;
				farNode = node->left;
			}

			// check near node
			if (nearNode != NULL) {
				oneNearestHelper (
					nearestNeighbor, nearestDistance, nodeDistVec,
					nearNode, query, nodeDist);
			}

			// check far node
			if (farNode != NULL) {
				DistanceType d = nodeDistVec[node->splitDim];
				DistanceType distToSplit = query[node->splitDim] -
					node->splitVal;
				distToSplit *= distToSplit;
				DistanceType distToFar = nodeDist - d + distToSplit;
				nodeDistVec[node->splitDim] = distToSplit;
				if (distToFar < nearestDistance) {
					oneNearestHelper (
						nearestNeighbor, nearestDistance,
						nodeDistVec, farNode, query, distToFar);
				}
				nodeDistVec[node->splitDim] = d; 
			}
		}	// if inner
	} // void oneNearestHelper

	void addPointsToQueue (
		KNearestQueue & queue, 
		const Node * leaf,
		const ElementType * query)
	{
		for (std::size_t i = leaf->beginIndex;
			i < leaf->endIndex; i++) {
			DistanceType dist = 0.0;
			for (std::size_t j = 0; j < _tree.getDim(); j++) {
				DistanceType temp = query[j] -
					_tree.getData()[i * _tree.getDim() + j];
				temp *= temp;
				dist += temp;
			}
			queue.push(
				std::pair<DistanceType,int>(
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
