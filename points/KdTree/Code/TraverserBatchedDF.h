#ifndef __TRAVERSERBATCHEDDF_H__
#define __TRAVERSERBATCHEDDF_H__

#include "Traverser.h"
#include "ResultSet.h"
#include "vltools/box3.h"
#include "vltools/timer.h"

//#define PROFILE_TRAVERSAL

namespace pointkd {

template <typename T> class PointKdTree;

template <typename T>
class TraverserBatchedDF : public Traverser<T> {
	using Traverser<T>::_tree;
public:
	typedef typename Traverser<T>::ElementType ElementType;
	typedef typename Traverser<T>::DistanceType DistanceType;
	typedef typename Traverser<T>::Node Node;
	typedef typename Traverser<T>::KNearestQueue KNearestQueue;
	typedef typename vltools::Box3<DistanceType> BoxType;

	TraverserBatchedDF(const PointKdTree<T> & tree) :
		Traverser<T>(tree),
		_numLeafsVisited(0),
		_numInnerVisited(0), 
		_numPointsChecked(0) { }

	void allNearest (
		const std::size_t k,
		std::vector<int> & nearestNeighbors,
		std::vector<DistanceType> & nearestDistances,
		const float initialMaxDist = 
			std::numeric_limits<DistanceType>::max())
	{
		std::size_t numQueries = _tree.getNumPoints();
		nearestNeighbors.resize (k * numQueries, -1);
		nearestDistances.resize (k * numQueries, initialMaxDist);

		#ifdef PROFILE_TRAVERSAL
		_numLeafsVisited = 0;
		_numInnerVisited = 0;
		_numPointsChecked = 0;
		_leafTime = 0.0;
		#endif
		const std::vector<const Node *> & leafPointers =
			_tree.getLeafPointers();
		const std::vector<BoxType> & leafBoxes = _tree.getLeafBoxes();
		if (k==1) {
		//if (false) {
		for (std::size_t i = 0; i < leafPointers.size(); i++) {
			// get leaf pointer
			const Node * leafPtr = leafPointers[i];
			// get leaf extent
			const BoxType & leafBox = leafBoxes[i];
			// initialize largest of distances to k-th nearest
			DistanceType maxDist = initialMaxDist;
			// number of points in leaf node
			std::size_t leafSize = leafPtr->endIndex -
				leafPtr->beginIndex;
			// keep track of each query points nearest neighbor
			std::vector<int> nhbr(leafSize,-1);
			std::vector<DistanceType> dist(leafSize,maxDist);
			// squared distance to "target" leaf node
			DistanceType nodeDist = 0.0;
			// '' along each axis
			std::vector<DistanceType> nodeDistVec(_tree.getDim(), 0.0);
			//begin recursion
			allOneNearestHelper(
				nhbr,
				dist,
				maxDist,
				nodeDist,
				nodeDistVec,
				_tree.getBoundingBox(),
				_tree.getRoot(),
				leafPtr,
				leafBox);
			// save results
			for (std::size_t j = 0; j < leafSize; j++) {
				int idx = _tree.getIndices()[leafPtr->beginIndex + j]; 
				nearestNeighbors[idx] = nhbr[j];
				nearestDistances[idx] = dist[j];
			}
		}
		} else {
		#if 1
		std::size_t maxLeafSize = this->_tree.getMaxLeafSize();
		std::vector<ResultSet<DistanceType> *> results(maxLeafSize);
		if (k >= 250) {
			for (std::size_t i = 0; i < maxLeafSize; i++)
				results[i] = new 
					ResultSetHeap<DistanceType>(k, initialMaxDist);
		} else {
			for (std::size_t i = 0; i < maxLeafSize; i++)
				results[i] = new 
					ResultSetLinearInsert<DistanceType>(k, initialMaxDist);
		}
		for (std::size_t i = 0; i < leafPointers.size(); i++) {
			// get leaf pointer
			const Node * leafPtr = leafPointers[i];
			// get leaf extent
			const BoxType & leafBox = leafBoxes[i];
			// initialize largest of distances to k-th nearest
			DistanceType maxDist = initialMaxDist;
			// number of points in leaf node
			std::size_t leafSize = leafPtr->endIndex -
				leafPtr->beginIndex;
			// vector of priority queues
			for (std::size_t j = 0; j < leafSize; j++)
				results[j]->clear();
			// squared distance to "target" leaf node
			DistanceType nodeDist = 0.0;
			// '' along each axis
			std::vector<DistanceType> nodeDistVec(_tree.getDim(), 0.0);
			// begin recursion
			allNearestHelper(
				results,
				maxDist,
				nodeDist,
				nodeDistVec,
				_tree.getRoot(),
				_tree.getBoundingBox(),
				leafPtr,
				leafBox);

			// save results (consider leaving this to
			// later reordering step)
			for (std::size_t j = 0; j < leafSize; j++) {
				int idx = _tree.getIndices()[leafPtr->beginIndex + j]; 
				int * nhbr = &nearestNeighbors[k * idx];
				DistanceType * dist = &nearestDistances[k * idx];
				results[j]->copy(dist, nhbr);
			}
		}
		for (std::size_t i = 0; i < maxLeafSize; i++)
			delete results[i];
		#else
		for (std::size_t i = 0; i < leafPointers.size(); i++) {
			// get leaf pointer
			const Node * leafPtr = leafPointers[i];
			// get leaf extent
			const BoxType & leafBox = leafBoxes[i];
			// initialize largest of distances to k-th nearest
			DistanceType maxDist = initialMaxDist;
			// number of points in leaf node
			std::size_t leafSize = leafPtr->endIndex -
				leafPtr->beginIndex;
			// vector of priority queues
			std::vector<KNearestQueue> results(
				leafSize, KNearestQueue(k));
			for (std::size_t j = 0; j < leafSize; j++)
				results[j].push(
					std::pair<DistanceType, int>(maxDist, -1));
			// squared distance to "target" leaf node
			DistanceType nodeDist = 0.0;
			// '' along each axis
			std::vector<DistanceType> nodeDistVec(_tree.getDim(), 0.0);
			// begin recursion
			allNearestHelper(
				results,
				maxDist,
				nodeDist,
				nodeDistVec,
				_tree.getRoot(),
				_tree.getBoundingBox(),
				leafPtr,
				leafBox);

			// save results (consider leaving this to
			// later reordering step)
			for (std::size_t j = 0; j < leafSize; j++) {
				int idx = _tree.getIndices()[leafPtr->beginIndex + j]; 
				int * nhbr = &nearestNeighbors[k * idx];
				DistanceType * dist = &nearestDistances[k * idx];
				for (std::size_t l = 0; !results[j].empty(); l++) {
					nhbr[l] = results[j].top().second;
					dist[l] = results[j].top().first;
					results[j].pop();
				}
			}
		}
		#endif
		} // k!=1
		#ifdef PROFILE_TRAVERSAL
		std::cout << "leafs: " << _numLeafsVisited << std::endl;
		std::cout << "inner: " << _numInnerVisited << std::endl;
		std::cout << "check: " << _numPointsChecked << std::endl;
		std::cout << "leaf times: " << _leafTime << std::endl;
		#endif
	}

	virtual void nearest(
		const std::vector<ElementType> & queries,
		const std::size_t k,
		std::vector<int> & nearestNeighbors,
		std::vector<DistanceType> & nearestDistances,
		const DistanceType initialMaxDist = 
			std::numeric_limits<DistanceType>::max()) {

	}

	virtual void range(
		const std::vector<ElementType> & queries,
		const DistanceType r,
		std::vector<int> & rNeighbors,
		std::vector<DistanceType> & neighborDistances) {

	}

private:
	void allOneNearestHelper(
		std::vector<int> & neighbors,
		std::vector<DistanceType> & distances,
		DistanceType & maxDist,
		DistanceType & nodeDist,
		std::vector<DistanceType> & nodeDistVec,
		const BoxType & nodeBox,
		const Node * node,
		const Node * leafPtr,
		const BoxType & leafBox)
	{
		if (node->isLeaf()) {
			maxDist = 0.0;
			std::size_t leafSize =
				leafPtr->endIndex - leafPtr->beginIndex;
			#ifdef PROFILE_TRAVERSAL
			_numLeafsVisited ++;
			#endif
			for (std::size_t i = 0;	i < leafSize; i++)
			{
				std::size_t idx = leafPtr->beginIndex + i;
				const ElementType * v = 
					&_tree.getData()[idx * _tree.getDim()];

				// test query point against node extent
				DistanceType distToLeaf = 0.0;
				bool leafTooFar = false;
				for (std::size_t j = 0; j < _tree.getDim(); j++) {
					DistanceType temp;
					((temp = v[j] - nodeBox.max(j)) > 0.0) ||
					((temp = nodeBox.min(j) - v[j]) > 0.0) ||
					(temp = 0.0);
					distToLeaf += temp * temp;
					if (leafTooFar = distToLeaf >= distances[i])
						break;
				}
				if (!leafTooFar) {
					#ifdef PROFILE_TRAVERSAL
					_numPointsChecked += 
						(node->endIndex-node->beginIndex);
					#endif
					// test query point against points inside node
					for (std::size_t j = node->beginIndex;
						j < node->endIndex; j++)
					{
						if (j == idx) continue;

						DistanceType dist = 0.0;
						const ElementType * w = 
							&_tree.getData()[j * _tree.getDim()];
						for (std::size_t k = 0; k < _tree.getDim(); k++) {
							DistanceType temp = v[k] - w[k];
							temp *= temp;
							dist += temp;
						}
						if (dist < distances[i]) {
							distances[i] = dist;
							neighbors[i] = _tree.getIndices()[j];
						}
					}
				}
				maxDist = distances[i] > maxDist ? distances[i] : maxDist;
			}
		} else {
			#ifdef PROFILE_TRAVERSAL
			_numInnerVisited ++;
			#endif
			const int splitDim = node->splitDim;
			const DistanceType splitVal = node->splitVal;
			Node * nearNode;
			Node * farNode;
			const ElementType * q = 
				&_tree.getData()[_tree.getDim() * leafPtr->beginIndex];
			BoxType nearBox = nodeBox;
			BoxType farBox = nodeBox;
			if (q[splitDim] < splitVal) {
				nearNode = node->left;
				farNode = node->right;
				nearBox.max(splitDim) = splitVal;
				farBox.min(splitDim) = splitVal;
			} else {
				nearNode = node->right;
				farNode = node->left;
				nearBox.min(splitDim) = splitVal;
				farBox.max(splitDim) = splitVal;
			}

			// descend down near node
			if (nearNode != NULL) {
				allOneNearestHelper(
					neighbors,
					distances,
					maxDist,
					nodeDist,
					nodeDistVec,
					nearBox,
					nearNode,
					leafPtr,
					leafBox);
			}

			// descend down far node
			if (farNode != NULL) {
				DistanceType d;
				(d = splitVal - leafBox.max(splitDim)) > 0.0 ||
				(d = leafBox.min(splitDim) - splitVal) > 0.0 || (d = 0.0);
				d *= d;
				DistanceType temp = nodeDistVec[splitDim];
				nodeDistVec[splitDim] = d;
				DistanceType distToFar = nodeDist - temp + d;
				if (distToFar < maxDist) { 
					allOneNearestHelper(
						neighbors,
						distances,
						maxDist,
						distToFar,
						nodeDistVec,
						farBox,
						farNode,
						leafPtr,
						leafBox);
				}
				nodeDistVec[splitDim] = temp;
			}
		}
	}

	void allNearestHelper(
		std::vector<KNearestQueue> & results,
		DistanceType & maxDist,
		DistanceType & nodeDist,
		std::vector<DistanceType> & nodeDistVec,
		const Node * node,
		const BoxType & nodeBox,
		const Node * leafPtr,
		const BoxType & leafBox)
	{
		if (node->isLeaf()) {
			std::size_t leafSize =
				leafPtr->endIndex - leafPtr->beginIndex;
			#ifdef PROFILE_TRAVERSAL
			_numLeafsVisited ++;
			double leafTime = vltools::getTime();
			#endif
			for (std::size_t i = 0;	i < leafSize; i++)
			{
				std::size_t idx = leafPtr->beginIndex + i;
				const ElementType * v = 
					&_tree.getData()[idx * _tree.getDim()];

				// test query point against node extent
				bool leafTooFar = false;
				if (results[i].size() == results[i].k()) {
					DistanceType distToLeaf = 0.0;
					for (std::size_t j = 0; j < _tree.getDim(); j++) {
						DistanceType temp;
						((temp = v[j] - nodeBox.max(j)) > 0.0) ||
						((temp = nodeBox.min(j) - v[j]) > 0.0) ||
						(temp = 0.0);
						distToLeaf += temp * temp;
						if (leafTooFar = distToLeaf >= results[i].top().first)
							break;
					}
				}
				if (!leafTooFar) {
					#ifdef PROFILE_TRAVERSAL
					_numPointsChecked += 
						(node->endIndex-node->beginIndex);
					#endif
					// test query point against points inside node
					for (std::size_t j = node->beginIndex;
						j < node->endIndex; j++)
					{
						if (j == idx) continue;

						DistanceType dist = 0.0;
						const ElementType * w = 
							&_tree.getData()[j * _tree.getDim()];
						for (std::size_t k = 0; k < _tree.getDim(); k++) {
							DistanceType temp = v[k] - w[k];
							temp *= temp;
							dist += temp;
						}
						results[i].push(std::pair<DistanceType, int>(
							dist, _tree.getIndices()[j]));
					}
				}
			}
			maxDist = results[0].top().first;
			for (std::size_t i = 1; i < leafSize; i++) { 
				maxDist = results[i].top().first > maxDist ?
					results[i].top().first : maxDist;
			}
			#ifdef PROFILE_TRAVERSAL
			_leafTime += vltools::getTime() - leafTime;
			#endif
		} else {
			#ifdef PROFILE_TRAVERSAL
			_numInnerVisited ++;
			#endif
			const int splitDim = node->splitDim;
			const DistanceType splitVal = node->splitVal;
			Node * nearNode, * farNode;
			const ElementType * q = 
				&_tree.getData()[_tree.getDim() * leafPtr->beginIndex];
			BoxType nearBox = nodeBox;
			BoxType farBox = nodeBox;
			if (q[splitDim] < splitVal) {
				nearNode = node->left;
				farNode = node->right;
				nearBox.max(splitDim) = splitVal;
				farBox.min(splitDim) = splitVal;
			} else {
				nearNode = node->right;
				farNode = node->left;
				nearBox.min(splitDim) = splitVal;
				farBox.max(splitDim) = splitVal;
			}

			// descend down near node
			if (nearNode != NULL) {
				allNearestHelper(
					results,
					maxDist,
					nodeDist,
					nodeDistVec,
					nearNode,
					nearBox,
					leafPtr,
					leafBox);
			}

			// descend down far node
			if (farNode != NULL) {
				DistanceType d;
				(d = splitVal - leafBox.max(splitDim)) > 0.0 ||
				(d = leafBox.min(splitDim) - splitVal) > 0.0 || (d = 0.0);
				d *= d;
				DistanceType temp = nodeDistVec[splitDim];
				nodeDistVec[splitDim] = d;
				DistanceType distToFar = nodeDist - temp + d;
				if (distToFar < maxDist) { 
					allNearestHelper(
						results,
						maxDist,
						distToFar,
						nodeDistVec,
						farNode,
						farBox,
						leafPtr,
						leafBox);
				}
				nodeDistVec[splitDim] = temp;
			}
		}

	}

	void allNearestHelper(
		std::vector<ResultSet<DistanceType> *> & results,
		DistanceType & maxDist,
		DistanceType & nodeDist,
		std::vector<DistanceType> & nodeDistVec,
		const Node * node,
		const BoxType & nodeBox,
		const Node * leafPtr,
		const BoxType & leafBox)
	{
		if (node->isLeaf()) {
			std::size_t leafSize =
				leafPtr->endIndex - leafPtr->beginIndex;
			#ifdef PROFILE_TRAVERSAL
			_numLeafsVisited ++;
			double leafTime = vltools::getTime();
			#endif
			for (std::size_t i = 0;	i < leafSize; i++)
			{
				std::size_t idx = leafPtr->beginIndex + i;
				const ElementType * v = 
					&_tree.getData()[idx * _tree.getDim()];

				// test query point against node extent
				bool leafTooFar = false;
				if (results[i]->full()) {
					DistanceType distToLeaf = 0.0;
					for (std::size_t j = 0; j < _tree.getDim(); j++) {
						DistanceType temp;
						((temp = v[j] - nodeBox.max(j)) > 0.0) ||
						((temp = nodeBox.min(j) - v[j]) > 0.0) ||
						(temp = 0.0);
						distToLeaf += temp * temp;
						if (leafTooFar = distToLeaf >= 
							results[i]->worstDist())
							break;
					}
				}
				if (!leafTooFar) {
					#ifdef PROFILE_TRAVERSAL
					_numPointsChecked += 
						(node->endIndex-node->beginIndex);
					#endif
					// test query point against points inside node
					for (std::size_t j = node->beginIndex;
						j < node->endIndex; j++)
					{
						if (j == idx) continue;

						DistanceType dist = 0.0;
						const ElementType * w = 
							&_tree.getData()[j * _tree.getDim()];
						for (std::size_t k = 0; k < _tree.getDim(); k++) {
							DistanceType temp = v[k] - w[k];
							temp *= temp;
							dist += temp;
						}
						results[i]->addPoint(dist, _tree.getIndices()[j]);
					}
				}
			}
			maxDist = results[0]->worstDist();
			for (std::size_t i = 1; i < leafSize; i++) { 
				maxDist = results[i]->worstDist() > maxDist ?
					results[i]->worstDist() : maxDist;
			}
			#ifdef PROFILE_TRAVERSAL
			_leafTime += vltools::getTime() - leafTime;
			#endif
		} else {
			#ifdef PROFILE_TRAVERSAL
			_numInnerVisited ++;
			#endif
			const int splitDim = node->splitDim;
			const DistanceType splitVal = node->splitVal;
			Node * nearNode, * farNode;
			const ElementType * q = 
				&_tree.getData()[_tree.getDim() * leafPtr->beginIndex];
			BoxType nearBox = nodeBox;
			BoxType farBox = nodeBox;
			if (q[splitDim] < splitVal) {
				nearNode = node->left;
				farNode = node->right;
				nearBox.max(splitDim) = splitVal;
				farBox.min(splitDim) = splitVal;
			} else {
				nearNode = node->right;
				farNode = node->left;
				nearBox.min(splitDim) = splitVal;
				farBox.max(splitDim) = splitVal;
			}

			// descend down near node
			if (nearNode != NULL) {
				allNearestHelper(
					results,
					maxDist,
					nodeDist,
					nodeDistVec,
					nearNode,
					nearBox,
					leafPtr,
					leafBox);
			}

			// descend down far node
			if (farNode != NULL) {
				DistanceType d;
				(d = splitVal - leafBox.max(splitDim)) > 0.0 ||
				(d = leafBox.min(splitDim) - splitVal) > 0.0 || (d = 0.0);
				d *= d;
				DistanceType temp = nodeDistVec[splitDim];
				nodeDistVec[splitDim] = d;
				DistanceType distToFar = nodeDist - temp + d;
				if (distToFar < maxDist) { 
					allNearestHelper(
						results,
						maxDist,
						distToFar,
						nodeDistVec,
						farNode,
						farBox,
						leafPtr,
						leafBox);
				}
				nodeDistVec[splitDim] = temp;
			}
		}
	}

protected:
	std::size_t _numLeafsVisited;
	std::size_t _numInnerVisited;
	std::size_t _numPointsChecked;
	double _leafTime;
};

} // namespace pointkd

#endif
