#ifndef __BUILDERPARALLEL_H__
#define __BUILDERPARALLEL_H__

#include "Builder.h"
#include "tbb/task.h"
#include "tbb/scalable_allocator.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/enumerable_thread_specific.h"
#include "vltools/timer.h"
#undef min
#undef max
namespace pointkd {

template <typename T>
class BuilderParallel : public Builder<T> {
public:
	typedef typename Builder<T>::ElementType ElementType;
	typedef typename Builder<T>::DistanceType DistanceType;
	typedef typename Builder<T>::Node Node;
	typedef typename Builder<T>::BoxType BoxType;
	typedef typename vltools::Box3<ElementType> EltBoxType;

	class Scratch {
	public:
		Scratch () {}
		Scratch (unsigned int dim) : _min(dim), _max(dim) {}
		void reset () {
			std::fill(_min.begin(), _min.end(), (ElementType)0);
			std::fill(_max.begin(), _max.end(), (ElementType)0);
		}
		std::vector<const Node*, tbb::scalable_allocator<const Node*> > 
			_leafPointers;
		std::vector<BoxType, tbb::scalable_allocator<BoxType> >
			_leafBoxes;
		std::vector<ElementType> _min;
		std::vector<ElementType> _max;
	};
	typedef tbb::enumerable_thread_specific<Scratch> ScratchType;

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
	using Builder<T>::_reverseIndices;

	BuilderParallel(PointKdTree<ElementType> & tree):
		Builder<ElementType>(tree) {}

	friend class BuildTask;
	class BuildTask : public tbb::task {
	public:
		BuildTask(
			const BoxType & nodeExtent,
			Node * node,
			int * indices,
			std::size_t numPoints,
			BuilderParallel<ElementType> & builder):
			nodeExtent(nodeExtent), node(node), indices(indices),
			numPoints(numPoints), _builder(builder), 
			_scratch(builder._perThreadScratch) {}

		tbb::task * execute() {
			if (numPoints < _builder._serialCutoff) {
				// serial cutoff
				_builder.buildHelper(
					nodeExtent, node, indices, numPoints);
				return NULL;
			} else if (numPoints <= _builder._maxLeafSize) {
				//node->left = node->right = NULL;
				//node->splitDim = *indices;
				node->beginIndex = (std::size_t)
					(indices - _builder._indices);
				node->endIndex = node->beginIndex + numPoints;
				node->splitDim = -1;
				// save leaf node pointers and extent
				_scratch.local()._leafPointers.push_back(node);
				_scratch.local()._leafBoxes.push_back(nodeExtent);
				return NULL;
			} else {
				// fill in node->splitDim and node->splitVal
				#if 0
				int splitType;
				_builder.computeSplit (
					indices,
					nodeExtent,
					numPoints,
					node->splitDim,
					node->splitVal,
					splitType);
				BoxType nodeExtentLeft(nodeExtent);
				nodeExtentLeft.max(node->splitDim) = node->splitVal;
				BoxType nodeExtentRight(nodeExtent);
				nodeExtentRight.min(node->splitDim) = node->splitVal;
				#else
				BoxType childNodeExtent(nodeExtent);
				EltBoxType boundingBox;
				_builder.computeBoundingBox (
					boundingBox,
					indices,
					numPoints);
				while (true) {
					DistanceType widestGapRatio;
					int emptySplitType;
					int emptySplitDim;
					DistanceType emptySplitVal;
					_builder.computeEmptySplit (
						widestGapRatio, emptySplitType,
						emptySplitDim, emptySplitVal,
						childNodeExtent, boundingBox);
					if (widestGapRatio <= _builder._emptySplitThreshold)
						break;
					node->splitDim = emptySplitDim;
					node->splitVal = emptySplitVal;
					Node * childNode =
						(Node*)scalable_malloc(sizeof(Node));
					if (emptySplitType == 0) {	// left empty
						node->left = NULL;
						node->right = childNode;
					} else {	// emptySplitType == 1, right empty
						node->left = childNode;
						node->right = NULL;
					}
					node = childNode;
					childNodeExtent._data[emptySplitType][emptySplitDim] = 
						emptySplitVal;
				}
				_builder.computeSplit (
					node->splitDim,
					node->splitVal,
					boundingBox);
				BoxType nodeExtentLeft(childNodeExtent);
				nodeExtentLeft.max(node->splitDim) = node->splitVal;
				BoxType nodeExtentRight(childNodeExtent);
				nodeExtentRight.min(node->splitDim) = node->splitVal;
				int splitType = 3;
				#endif
				// reorder indices
				std::size_t numLeft, numRight;
				if (splitType == 3) {
					numLeft = _builder.partitionIndices (
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

				// create left task
				BuildTask * leftTask;
				if (numLeft > 0) {
					node->left = (Node*)scalable_malloc (sizeof(Node));
					leftTask = new(tbb::task::allocate_child())
						BuildTask(nodeExtentLeft, node->left,
						indices, numLeft, _builder);
				} else
					node->left = NULL;

				// create right task
				BuildTask * rightTask = NULL;
				if (numRight > 0) {
					node->right = (Node*)scalable_malloc (sizeof(Node));
					rightTask = new(tbb::task::allocate_child())
						BuildTask(nodeExtentRight, node->right,
						indices + numLeft, numRight, _builder);
				} else
					node->right = NULL;

				// spawn tasks
				if (splitType == 3) {
					set_ref_count(3);
					spawn(*leftTask);
					spawn_and_wait_for_all(*rightTask);
				} else if (splitType == 1) { // left empty
					set_ref_count(2);
					spawn_and_wait_for_all(*rightTask);
				} else { // right empty
					set_ref_count(2);
					spawn_and_wait_for_all(*leftTask);
				}
				return NULL;
			}
		}
	private:
		const BoxType & nodeExtent;
		Node * node;
		int * indices;
		std::size_t numPoints;
		BuilderParallel<ElementType> & _builder;
		ScratchType & _scratch;
	};

	void build(
		const ElementType * points,
		const std::size_t numPoints,
		const std::size_t dim,
		const std::size_t maxLeafSize = 10,
		const float emptySplitThreshold = 0.2f,
		const int numProcs = -1,
		const std::size_t serialCutoff = 0)
	{
		if (dim != 3) {
			std::cout << "Currently only supprts dim==3" << std::endl;
			exit(0);
		}
		_perThreadScratch = ScratchType(Scratch((int)dim));

		// initialize attributes
		_data = points;
		_numPoints = numPoints;
		_dim = dim;
		_maxLeafSize = maxLeafSize;
		_emptySplitThreshold = emptySplitThreshold;
		_serialCutoff = serialCutoff;

		// initialize indices
		_indices = (int*)scalable_malloc(sizeof(int) * _numPoints);
		for (std::size_t i = 0; i < _numPoints; i++)
			_indices[i] = (int)i;

		// compute bounding box (assumes _dim == 3)
		_boundingBox.addPoints(_data, _numPoints);

		// begin recursive build
		_root = (Node*) scalable_malloc (sizeof(Node));
		BuildTask & rootTask = *new(tbb::task::allocate_root())
			BuildTask(_boundingBox, _root, _indices, _numPoints, *this);
		tbb::task::spawn_root_and_wait(rootTask);

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

		// combine per thread leaf pointers and boxes
		for (typename ScratchType::const_iterator
			i = _perThreadScratch.begin();
			i < _perThreadScratch.end(); ++i) {
			const std::vector<const Node *,
				tbb::scalable_allocator<const Node *> > & pointers = 
				i->_leafPointers;
			const std::vector<BoxType,
				tbb::scalable_allocator<BoxType> > & boxes =
				i->_leafBoxes;
			_leafPointers.insert(_leafPointers.end(),
				pointers.begin(), pointers.end());
			_leafBoxes.insert(_leafBoxes.end(),
				boxes.begin(), boxes.end());
		}
	}
private:
	void buildHelper(
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
			_perThreadScratch.local()._leafPointers.push_back(node);
			_perThreadScratch.local()._leafBoxes.push_back(nodeExtent);
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
	void computeBoundingBox (
		EltBoxType & boundingBox,
		const int * indices,
		const std::size_t numPoints)
	{
		for (std::size_t i = 0; i < _dim; i++) {
			boundingBox.min(i) = boundingBox.max(i) =
				_data[indices[0] * _dim + i];
		}
		for (std::size_t i = 1; i < numPoints; i++) {
			const ElementType * v = &_data[indices[i] * _dim];
			for (std::size_t j = 0; j < _dim; j++) {
				boundingBox.min(j) = v[j] < boundingBox.min(j) ?
					v[j] : boundingBox.min(j);
				boundingBox.max(j) = v[j] > boundingBox.max(j) ?
					v[j] : boundingBox.max(j);
			}
		}
	}
	void computeEmptySplit (
		DistanceType & widestGapRatio,
		int & splitType,
		int & splitDim,
		DistanceType & splitVal,
		const BoxType & nodeExtent,
		const EltBoxType & boundingBox)
	{
		widestGapRatio = 0.0f;
		splitType = 0;
		splitDim = 0;
		splitVal = 0.0;
		DistanceType widestGapSize = 0.0f;
		for (std::size_t i = 0; i < _dim; i++) {
			DistanceType gapSize = 
				nodeExtent.max(i) - boundingBox.max(i);
			if (gapSize > widestGapSize) {
				widestGapSize = gapSize;
				widestGapRatio = gapSize /
					(nodeExtent.max(i) - nodeExtent.min(i));
				splitType = 1;
				splitDim = (int)i;
				splitVal = boundingBox.max(i);
			}
			gapSize = boundingBox.min(i) - nodeExtent.min(i);
			if (gapSize > widestGapSize) {
				widestGapSize = gapSize;
				widestGapRatio = gapSize /
					(nodeExtent.max(i) - nodeExtent.min(i));
				splitType = 0;
				splitDim = (int)i;
				splitVal = boundingBox.min(i);
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
		_perThreadScratch.local().reset();
		ElementType * _min = &_perThreadScratch.local()._min[0];
		ElementType * _max = &_perThreadScratch.local()._max[0];

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
				emptySplitDim = (int)i;
				splitType = 2;
			}
			gapSize = _min[i] - nodeExtent.min(i);
			if (gapSize > widestGapSize) {
				widestGapSize = gapSize;
				widestGapRatio = gapSize /
					(nodeExtent.max(i) - nodeExtent.min(i));
				emptySplitVal = _min[i];
				emptySplitDim = (int)i;
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
					splitDim = (int)i;
					if (_max[i] == _min[i])
						splitVal = _max[i];
					else
						splitVal = (DistanceType)0.5 * (_max[i] + _min[i]);
				}
			}
			splitType = 3;
		}
	}
	void computeSplit(
		int & splitDim,
		DistanceType & splitVal,
		const EltBoxType & box)
	{
		DistanceType maxSpread = 0.0;
		for (std::size_t i = 0; i < _dim; i++) {
			DistanceType spread = box.max(i) - box.min(i);
			if (spread >= maxSpread) {
				maxSpread = spread;
				splitDim = (int)i;
				if (box.min(i) == box.max(i))
					splitVal = box.min(i);
				else
					splitVal = (DistanceType)0.5 * (box.max(i) + box.min(i));
			}
		}
	}
	std::size_t partitionIndices (
		int * indices,
		const std::size_t numPoints,
		const int splitDim,
		const DistanceType splitVal)
	{
		int left = 0;
		int right = (int)numPoints - 1;
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
		right = (int)numPoints - 1;
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

	// scratch space for computing tight bounding box
	ScratchType _perThreadScratch;
	std::size_t _serialCutoff;
};


}	// namespace pointkd
#endif	// ifndef __BUILDERPARALLEL_H__
