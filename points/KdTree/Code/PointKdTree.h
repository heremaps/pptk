#ifndef __POINTKDTREE_H__
#define __POINTKDTREE_H__

#include "PointKdTree_decl.h"
#include "tbb/scalable_allocator.h"
#include <iostream>
#include <stdlib.h>

//#define PROFILE_TRAVERSAL
namespace pointkd {

template <typename T> PointKdTree<T>::PointKdTree (
	const std::vector<ElementType> & points,
	std::size_t dim, std::size_t maxLeafSize,
	float emptySplitThreshold)
{
	if (dim != 3) {
		std::cout << "Currently only supprts dim==3" << std::endl;
		exit(0);
	}

	// initialize attributes
	_data = &points[0];
	_numPoints = points.size() / dim;
	_dim = dim;
	_maxLeafSize = maxLeafSize;
	_emptySplitThreshold = emptySplitThreshold;

	// initialize indices
	_indices = (int*)scalable_malloc(sizeof(int) * _numPoints);
	for (std::size_t i = 0; i < _numPoints; i++)
		_indices[i] = i;

	// allocate scratch
	_min = (ElementType*)scalable_malloc(
		sizeof(ElementType) * _dim);
	_max = (ElementType*)scalable_malloc(
		sizeof(ElementType) * _dim);

	// compute bounding box (assumes _dim == 3)
	_boundingBox.addPoints(_data, _numPoints);

	// begin recursive build
	_root = (Node*) scalable_malloc (sizeof(Node));
	build (_boundingBox, _root, _indices, _numPoints);

	// record tree depth
	_height = 1;
	computeTreeDepth (_root, 1);

	// make reordered copy of data
	ElementType * data = (ElementType*) scalable_malloc (
		sizeof(ElementType) * points.size());
	for (std::size_t i = 0; i < _numPoints; i++) {
		for (std::size_t j = 0; j < _dim; j++) {
			data[_dim * i + j] = points[_dim * _indices[i] + j];
		}
	}
	_data = (const ElementType *)data;

	// record reverse indices
	_reverseIndices = (int*)scalable_malloc(sizeof(int) * _numPoints);
	for (std::size_t i = 0; i < _numPoints; i++)
		_reverseIndices[_indices[i]] = (int)i;
		
	// linearize node array
#if 0
	Node * nodesLinear = (Node*)scalable_malloc (
		sizeof(Node) * (2 * _numPoints - 1));
	Node * nodesLinearOrig = nodesLinear;
	linearizeHelper (_root, nodesLinear);
	freeNode(_root);
	_root = nodesLinearOrig;
#endif

	// clean scratch
	//scalable_free(_indices);
	scalable_free(_min);
	scalable_free(_max);
}

template <typename T> PointKdTree<T>::~PointKdTree()
{
	freeNode(_root);
	//scalable_free(_root);
	scalable_free(_indices);
	scalable_free(_reverseIndices);
	scalable_free((ElementType*)_data);
}

template <typename T> void PointKdTree<T>::build (
	const BoxType & nodeExtent,
	Node * node,
	int * indices,
	std::size_t numPoints)
{
	if (numPoints <= _maxLeafSize) {
		//node->left = node->right = NULL;
		//node->splitDim = *indices;
		node->beginIndex = (std::size_t)(indices - _indices);
		node->endIndex = node->beginIndex + numPoints;
		node->splitDim = -1;
		// save leaf node pointers and extent
		_leafPointers.push_back(node);
		_leafBoxes.push_back(nodeExtent);
	} else {
		// fill in node->splitDim and node->splitVal
		int splitType;
		computeSplit (
			indices,
			nodeExtent,
			numPoints,
			node->splitDim,
			node->splitVal,
			splitType);
		// reorder indices
		std::size_t numLeft, numRight;
		if (splitType == 3) {
			numLeft = partitionIndices (
				indices, numPoints,
				node->splitDim,
				node->splitVal);
			numRight = numPoints - numLeft;
		} else if (splitType == 1) {
			numLeft = 0;
			numRight = numPoints;
		} else {	// splitType == 2
			numLeft = numPoints;
			numRight = 0;
		}
		// recurse
		if (numLeft > 0) {
			node->left = (Node*)scalable_malloc (sizeof(Node));
			BoxType nodeExtentLeft(nodeExtent);
			nodeExtentLeft.max(node->splitDim) = node->splitVal;
			build (nodeExtentLeft,
				node->left, indices, numLeft);
		} else {
			node->left = NULL;
		}
		if (numRight > 0) {
			node->right = (Node*)scalable_malloc (sizeof(Node));
			BoxType nodeExtentRight(nodeExtent);
			nodeExtentRight.min(node->splitDim) = node->splitVal;
			build (nodeExtentRight,
				node->right, indices + numLeft, numRight);
		} else {
			node->right = NULL;
		}
	}
}	

template <typename T> void PointKdTree<T>::computeSplit (
	const int * indices,
	const BoxType & nodeExtent,
	const std::size_t numPoints,
	int & splitDim,
	DistanceType & splitVal,
	int & splitType)
{
	//memset(_min, 0, sizeof(ElementType) * _dim);
	//memset(_max, 0, sizeof(ElementType) * _dim);

	// initialize _min, _max for all dimensions
	for (std::size_t i = 0; i < _dim; i++) {
		_min[i] = _max[i] = _data[indices[0] * _dim + i];
	}

	// compute _min, _max over all points in indices
	for (std::size_t i = 1; i < numPoints; i++) {
		const ElementType * v = &_data[indices[i] * _dim];
		for (std::size_t j = 0; j < _dim; j++) {
			_min[j] = v[j] < _min[j] ? v[j] : _min[j];
			_max[j] = v[j] > _max[j] ? v[j] : _max[j];
		}
	}

	// determine largest empty split gap
	// assumes _dim == 3
	DistanceType widestGapSize = (DistanceType)(0.0);
	DistanceType widestGapRatio = (DistanceType)(0.0);
	DistanceType emptySplitVal = (DistanceType)0.0;
	int emptySplitDim = 0;
	splitType = 3;
		// 1 - left empty
		// 2 - right empty
		// 3 - non-empty split
	for (std::size_t i = 0; i < this->_dim; i++) {
		DistanceType gapSize = nodeExtent.max(i) - _max[i];
		if (gapSize > widestGapSize) {
			widestGapSize = gapSize;
			widestGapRatio = gapSize /
				(nodeExtent.max(i) - nodeExtent.min(i));
			emptySplitVal = _max[i];
			emptySplitDim = i;
			splitType = 2;
		}
		gapSize = _min[i] - nodeExtent.min(i);
		if (gapSize > widestGapSize) {
			widestGapSize = gapSize;
			widestGapRatio = gapSize /
				(nodeExtent.max(i) - nodeExtent.min(i));
			emptySplitVal = _min[i];
			emptySplitDim = i;
			splitType = 1;
		}
	}

	// decide whether to perform empty split or not
	if (splitType != 3 && widestGapRatio > _emptySplitThreshold) {
		// choose empty split
		splitDim = emptySplitDim;
		splitVal = emptySplitVal;
		
	} else {
		// determine dim with widest spread
		DistanceType maxSpread = 0.0;
		for (std::size_t i = 0; i < _dim; i++) {
			DistanceType spread = _max[i] - _min[i];
			if (spread >= maxSpread) {
				maxSpread = spread;
				splitDim = i;
				if (_max[i] == _min[i])
					splitVal = _max[i];
				else
					splitVal = 0.5 * (_max[i] + _min[i]);
			}
		}
		splitType = 3;
	}
}

template <typename T> std::size_t PointKdTree<T>::partitionIndices (
	int * indices,
	const std::size_t numPoints,
	const int splitDim,
	const DistanceType splitVal)
{
	int left = 0;
	int right = numPoints - 1;
	for (;;) {
		while (left <= right &&
			_data[indices[left]*_dim+splitDim]<splitVal) {
			left++;
		}
		while (left <= right &&
			_data[indices[right]*_dim+splitDim]>=splitVal) {
			right--;
		}
		if (left >= right) break;
		std::swap (indices[left], indices[right]);
		left++;
		right--;
	}
	int lim1 = left;
	right = numPoints - 1;
	for (;;) {
		while (left <= right &&
			_data[indices[left]*_dim+splitDim]<=splitVal) {
			left++;
		}
		while (left <= right &&
			_data[indices[right]*_dim+splitDim]>splitVal) {
			right--;
		}
		if (left >= right) break;
		std::swap(indices[left], indices[right]);
		left++;
		right--;
	}
	int lim2 = left;

	std::size_t numLeft;
	if (lim1 > numPoints / 2) numLeft = lim1;
	else if (lim2 < numPoints / 2) numLeft = lim2;
	else numLeft = numPoints / 2;

	if (lim1 == numPoints && lim2 == 0) numLeft = numPoints / 2;

	return numLeft;
}

template <typename T> void PointKdTree<T>::computeTreeDepth (
	Node * node,
	unsigned int currentDepth)
{
	if (node == NULL) {
		return;
	}
	if (node->isLeaf()) { // leaf
		if (currentDepth > _height)
			_height = currentDepth;
	} else {
		computeTreeDepth (node->left, currentDepth + 1);
		computeTreeDepth (node->right, currentDepth + 1);
	}
}

template <typename T> void PointKdTree<T>::freeNode (Node * node)
{
	if (node == NULL)
		return;
	if (!node->isLeaf()) {
		freeNode (node->left);
		freeNode (node->right);
	}
	scalable_free (node);
}

template <typename T>
std::size_t PointKdTree<T>::countLeafNodesHelper (Node * node)
{
	if (node == NULL)
		return 0;
	else if (node->isLeaf())
		return 1;
	else
		return 
			countLeafNodesHelper(node->left) + 
			countLeafNodesHelper(node->right);
}

template <typename T>
std::size_t PointKdTree<T>::countInnerNodesHelper (Node * node)
{
	if (node == NULL || node->isLeaf())
		return 0;
	else
		return 1 +
			countInnerNodesHelper(node->left) +
			countInnerNodesHelper(node->right);
}

template <typename T>
bool PointKdTree<T>::compareHelper (
	const Node * thisNode,
	const Node * otherNode,
	const PointKdTree<T> & otherTree) const
{
	if (thisNode == NULL && otherNode == NULL) {
		return true;
	} else {
		if (thisNode->isLeaf() && otherNode->isLeaf() &&
			(thisNode->endIndex - thisNode->beginIndex ==
			otherNode->endIndex - otherNode->beginIndex)) {
			std::size_t n = thisNode->endIndex - thisNode->beginIndex;
			for (std::size_t i = 0; i < n; i++) {
				if (this->_indices[thisNode->beginIndex + i] !=
					otherTree._indices[otherNode->beginIndex + i])
					return false;
			}
			return true;
		} else if (!thisNode->isLeaf() && !otherNode->isLeaf() &&
			thisNode->splitVal == otherNode->splitVal &&
			thisNode->splitDim == otherNode->splitDim)
			return
				compareHelper(thisNode->left,
					otherNode->left, otherTree) &&
				compareHelper(thisNode->right,
					otherNode->right, otherTree);
		else
			return false;
	}
}

/*
template <typename T> void PointKdTree<T>::linearizeHelper (
	Node * node,
	Node * & nextNode)
	// writes nodes in subtree rooted at node to array starting at
	// nextNode.  Nodes are written in depth-first order.
{
	Node * newNode = nextNode;
	nextNode++;
	if (node->isLeaf()) {
		newNode->splitDim = node->splitDim;
		newNode->beginIndex = node->beginIndex;
		newNode->endIndex = node->endIndex;
	} else {
		newNode->leftMax = node->leftMax;
		newNode->rightMin = node->rightMin;
		newNode->splitDim = node->splitDim;
		newNode->left = nextNode;
		linearizeHelper (node->left, nextNode);
		newNode->right = nextNode;
		linearizeHelper (node->right, nextNode);
	}
}
*/
} // namespace pointkd
#endif
