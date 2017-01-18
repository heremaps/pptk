#ifndef __BUILDERSERIAL_H__
#define __BUILDERSERIAL_H__

#include "Builder.h"
#include "tbb/scalable_allocator.h"

namespace pointkd {

template <typename T>
class BuilderSerial : public Builder<T> {
public:
	typedef typename Builder<T>::ElementType ElementType;
	typedef typename Builder<T>::DistanceType DistanceType;
	typedef typename Builder<T>::Node Node;
	typedef typename Builder<T>::BoxType BoxType;

	using Builder<T>::_data;
	using Builder<T>::_numPoints;
	using Builder<T>::_dim;
	using Builder<T>::_maxLeafSize;
	using Builder<T>::_emptySplitThreshold;
	using Builder<T>::_indices;
	using Builder<T>::_boundingBox;
	using Builder<T>::_root;
	using Builder<T>::_height;
	using Builder<T>::_leafPointers;
	using Builder<T>::_leafBoxes;

	BuilderSerial(PointKdTree<ElementType> & tree):
		Builder<ElementType>(tree), _min(NULL), _max(NULL) {}

	void build(
		const ElementType * points,
		const std::size_t numPoints,
		const std::size_t dim,
		const std::size_t maxLeafSize = 10,
		const float emptySplitThreshold = 0.2f)
	{
		if (dim != 3) {
			std::cout << "Currently only supprts dim==3" << std::endl;
			exit(0);
		}

		// initialize attributes
		_data = points;
		_numPoints = numPoints;
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
		buildHelper (_boundingBox, _root, _indices, _numPoints);

		// record tree depth
		_height = 1;
		this->computeTreeDepth (_height, _root, 1);

		// make reordered copy of data
		ElementType * data = (ElementType*) scalable_malloc (
			sizeof(ElementType) * _numPoints * _dim);
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
private:
	Node * buildHelper(
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
				buildHelper (nodeExtentLeft,
					node->left, indices, numLeft);
			} else {
				node->left = NULL;
			}
			if (numRight > 0) {
				node->right = (Node*)scalable_malloc (sizeof(Node));
				BoxType nodeExtentRight(nodeExtent);
				nodeExtentRight.min(node->splitDim) = node->splitVal;
				buildHelper (nodeExtentRight,
					node->right, indices + numLeft, numRight);
			} else {
				node->right = NULL;
			}
		}
	}	
	void computeSplit (
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
					if (_min[i] == _max[i])
						splitVal = _min[i];
					else
						splitVal = 0.5 * (_max[i] + _min[i]);
				}
			}
			splitType = 3;
		}
	}

	std::size_t partitionIndices (
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
		std::size_t lim1 = left;
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
		std::size_t lim2 = left;

		std::size_t numLeft;
		if (lim1 > numPoints / 2) numLeft = lim1;
		else if (lim2 < numPoints / 2) numLeft = lim2;
		else numLeft = numPoints / 2;

		if (lim1 == numPoints && lim2 == 0) numLeft = numPoints / 2;

		return numLeft;
	}

	// scratch space for computing tight bounding box
	ElementType * _min;
	ElementType * _max;
};


}	// namespace pointkd
#endif	// ifndef __BUILDERSERIAL_H__
