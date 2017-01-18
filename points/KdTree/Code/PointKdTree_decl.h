#ifndef __POINTKDTREE_DECL_H__
#define __POINTKDTREE_DECL_H__

#include "Accumulator.h"
#include "vltools/box3.h"
#include <vector>
#include <string.h>
#include <limits>

// Design choices:
// How many points per leaf?
// - also need to store number of points in leaf
// - more points => less traversal steps
// Keep reordered local copy of data?
// - regular access pattern when testing points
// - store as SoA or AoS?
// Maintain bounding box?
// - avoid always computing min/max for all dims 
// - allows for storing splitLeft and splitRight
// - parent task must wait for children tasks to finish
// (to combine children bounding boxes)
// Have splitLeft, splitRight as opposed to splitVal?
// Reorder point data during construction?

namespace pointkd {

template <typename T>
class PointKdTree {
public:
	// typedefs for more descriptive names
	typedef T ElementType;
	typedef typename Accumulator<T>::Type DistanceType;
	typedef vltools::Box3<DistanceType> BoxType;

	// node struct
	struct Node {
		DistanceType splitVal;
		int splitDim;
		union {
			Node * left;
			std::size_t beginIndex;
		};
		union {
			Node * right;
			std::size_t endIndex;
		};
		inline bool isLeaf() const {return splitDim == -1;}
	};

	PointKdTree () : _data(NULL), _numPoints(0), _dim(0),
		_indices(NULL), _reverseIndices(NULL), _root(NULL),
		_maxLeafSize(10), _emptySplitThreshold(0.2f) {}

	PointKdTree (
		const std::vector<ElementType> & points,
		std::size_t dim, std::size_t maxLeafSize = 10,
		float emptySplitThreshold = 0.2f);

	~PointKdTree ();

	void kNearest_BBF (
		std::vector<int> & neighbors,
		std::vector<DistanceType> & distances,
		const std::vector<ElementType> & queries,
		const std::size_t k);

	void kNearest_DF (
		std::vector<int> & neighbors,
		std::vector<DistanceType> & distances,
		const std::vector<ElementType> & queries,
		const std::size_t k);

	void kNearestI (
		std::vector<int> & neighbors,
		std::vector<DistanceType> & distances,
		const std::vector<ElementType> & queries,
		const std::size_t k);

	void allkNearestB (
		std::vector<int> & neighbors,
		std::vector<DistanceType> & distances,
		const std::size_t k);

	std::size_t countNodes() {
		return 
			countLeafNodesHelper(_root) + 
			countInnerNodesHelper(_root);
	}
	std::size_t countLeafNodes() {
		return countLeafNodesHelper(_root);
	}
	std::size_t countInnerNodes() {
		return countInnerNodesHelper(_root);
	}
	bool compare(const PointKdTree<ElementType> & other) const {
		return compareHelper(this->_root, other._root, other);
	}

	// reference getting functions (used by builder)
	const ElementType * & data() {return _data;}
	std::size_t & numPoints() {return _numPoints;}
	std::size_t & dim() {return _dim;}
	std::size_t & maxLeafSize() {return _maxLeafSize;}
	float & emptySplitThreshold() {return _emptySplitThreshold;}
	int * & indices() {return _indices;}
	int * & reverseIndices() {return _reverseIndices;}
	BoxType & boundingBox() {return _boundingBox;}
	Node * & root() {return _root;}
	unsigned int & height() {return _height;}
	std::vector<const Node*> & leafPointers() {return _leafPointers;}
	std::vector<BoxType> & leafBoxes() {return _leafBoxes;}

	// getter functions
	const Node * getRoot() const {return _root;}
	const ElementType * getData() const {return _data;}
	const int * getIndices() const {return _indices;}
	const int * getReverseIndices() const {return _reverseIndices;}
	const std::size_t getDim() const {return _dim;}
	const std::size_t getNumPoints() const {return _numPoints;}
	const std::size_t getMaxLeafSize() const {return _maxLeafSize;}
	const unsigned int getHeight() const {return _height;}
	const BoxType & getBoundingBox() const {return _boundingBox;};
	const std::vector<const Node *> & getLeafPointers() const {
		return _leafPointers;
	}	
	const std::vector<BoxType> & getLeafBoxes() const {
		return _leafBoxes;
	}

private: 
	void build (
		const BoxType & nodeExtent,
		Node * node,
		int * indices,
		std::size_t numPoints);

	void computeSplit (
		const int * indices,
		const BoxType & nodeExtent,
		const std::size_t numPoints,
		int & splitDim,
		DistanceType & splitVal,
		int & splitType);

	std::size_t partitionIndices (
		int * indices,
		const std::size_t numPoints,
		const int splitDim,
		const DistanceType splitVal);

	void computeTreeDepth (
		Node * node,
		unsigned int currentDepth);

	void freeNode (Node * node);

	void linearizeHelper (
		Node * node,
		Node * & nextNode);

	std::size_t countLeafNodesHelper (Node * node);
	std::size_t countInnerNodesHelper (Node * node);
	bool compareHelper (const Node * thisNode,
		const Node * otherNode, const PointKdTree<T> & otherTree) const;

	const ElementType * _data;
	std::size_t _numPoints;
	std::size_t _dim;
	std::size_t _maxLeafSize;
	float _emptySplitThreshold;
	int * _indices;
	int * _reverseIndices;	// i == _indices[_reverse_indices[i]]
	BoxType _boundingBox;
	// _min and _max are scratch space
	ElementType * _min;
	ElementType * _max;
	Node * _root;

	unsigned int _height;
	// leaf nodes
	std::vector<const Node *> _leafPointers;
	std::vector<BoxType> _leafBoxes;
};

} // namespace pointkd
#endif
