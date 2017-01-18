#ifndef __BUILDER_H__
#define __BUILDER_H__

#include "PointKdTree_decl.h"

namespace pointkd {

template <typename T>
class Builder {
public:
	typedef T ElementType;
	typedef typename Accumulator<T>::Type DistanceType;
	typedef typename PointKdTree<T>::BoxType BoxType;
	typedef typename PointKdTree<T>::Node Node;

	Builder(PointKdTree<ElementType> & tree) :
		_data(tree.data()),
		_numPoints(tree.numPoints()),
		_dim(tree.dim()),
		_maxLeafSize(tree.maxLeafSize()),
		_emptySplitThreshold(tree.emptySplitThreshold()),
		_indices(tree.indices()),
		_reverseIndices(tree.reverseIndices()),
		_boundingBox(tree.boundingBox()),
		_root(tree.root()),
		_height(tree.height()),
		_leafPointers(tree.leafPointers()),
		_leafBoxes(tree.leafBoxes()) {}

protected:

	static void computeTreeDepth (
		unsigned int & maxDepth,
		const Node * node,
		const unsigned int currentDepth)
	{
		if (node == NULL) {
			return;
		}
		if (node->isLeaf()) { // leaf
			if (currentDepth > maxDepth)
				maxDepth = currentDepth;
		} else {
			computeTreeDepth (maxDepth, node->left, currentDepth + 1);
			computeTreeDepth (maxDepth, node->right, currentDepth + 1);
		}
	}

	// "map" _tree member variables into scope of Builder<T>
	const ElementType * & _data;
	std::size_t & _numPoints; 
	std::size_t & _dim;
	std::size_t & _maxLeafSize;
	float & _emptySplitThreshold;
	int * & _indices;
	int * & _reverseIndices;
	BoxType & _boundingBox;
	Node * & _root;
	unsigned int & _height;
	std::vector<const Node *> & _leafPointers;
	std::vector<BoxType> & _leafBoxes;
};

}	// namespace pointkd
#endif	// ifndef __BUILDER_H__
