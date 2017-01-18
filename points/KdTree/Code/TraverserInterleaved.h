#ifndef __TRAVERSERINTERLEAVED_H__
#define __TRAVERSERINTERLEAVED_H__

// note: this file is only meant to be included from PointKdTree.h
#include "Accumulator.h"
#include "vltools/Stack.h"
#include "PointKdTree_decl.h"
#include "Traverser.h"
#include "tbb/scalable_allocator.h"
#include <limits>
#include <vector>
#if defined (__INTEL_COMPILER)
#include "xmmintrin.h"
#endif

#include <iostream>
namespace pointkd {

template <typename T> class PointKdTree;

//#define NUMTRAVS 4 // number of simultaneous traversals
template <typename T>
class TraverserInterleaved : public Traverser<T> {
	using Traverser<T>::_tree;
public:
	typedef T ElementType;
	typedef typename Accumulator<T>::Type DistanceType;
	typedef typename PointKdTree<T>::Node Node;
	typedef typename Traverser<T>::NodeDistPair NodeDistPair;

	struct State {
		State () :
			currentNode(NULL),
			//nearestNeighbor(-1),
			//nearestDistance(std::numeric_limits<DistanceType>::max()),
			nearestNeighbor(NULL),
			nearestDistance(NULL),
			query(NULL),
			active(false) { }
		const Node * currentNode;
		vltools::Stack<NodeDistPair> farNodes;
		int * nearestNeighbor;
		DistanceType * nearestDistance;
		const ElementType * query;
		bool active;
	};

	TraverserInterleaved(const PointKdTree<T> & tree) :
		Traverser<T>(tree)
	{
		_states.resize(NUMTRAVS);
		for (std::size_t i = 0; i < NUMTRAVS; i++) {
			_states[i].farNodes.resize(_tree.getHeight());
		}
	}

	void nearest(
		const std::vector<ElementType> & queries,
		const std::size_t k,
		std::vector<int> & nearestNeighbors,
		std::vector<DistanceType> & nearestDistances)
	{
		// todo:
		// - support k > 1

		// save query parameters
		_queries = &queries[0];
		_numQueries = queries.size() / _tree.getDim();
		// expect queries.size() % _tree.getDim() == 0
		_k = k;

		// resize and initialize result arrays
		nearestNeighbors.clear();
		nearestDistances.clear();
		nearestNeighbors.resize(_numQueries * _k, -1);
		nearestDistances.resize(_numQueries * _k,
			std::numeric_limits<DistanceType>::max());
		_nearestNeighbors = &nearestNeighbors[0];
		_nearestDistances = &nearestDistances[0];

		// initialize traversal states
		std::size_t nextQuery =
			_numQueries < NUMTRAVS ? _numQueries : NUMTRAVS;
		for (std::size_t i = 0; i < nextQuery; i++) {
			_states[i].currentNode = _tree.getRoot();
			_states[i].query = _queries + i * _tree.getDim();
			_states[i].nearestNeighbor = _nearestNeighbors + i;
			_states[i].nearestDistance = _nearestDistances + i;
			_states[i].active = true;
		}

		// start traversal
		bool done = false;
		while (!done) {
			done = true;
			for (std::size_t i = 0; i < NUMTRAVS; i++) {
				if (!_states[i].active)
					continue;
				done = false;
				const Node * node = _states[i].currentNode;
				State & state = _states[i];
				if (node == NULL) { // empty
					if (!backtrack(state))
						// current query done
						setNext(nextQuery++,
							state);
				} else if (node->isLeaf()) { // leaf
					updateNearest(state);
					if (!backtrack(state))
						// current query done
						setNext(nextQuery++,
							state);
				} else { // inner
					descend(state);
				}

				// prefetch next node here:
				#if defined(_MSR_VER) || defined(__INTEL_COMPILER)
				_mm_prefetch (
					(char*) state.currentNode, _MM_HINT_T0);
				#elif defined(__GNUC__)
				__builtin_prefetch (
					(const void*) state.currentNode);
				#endif
			}
		}
	}

	void range(
		const std::vector<ElementType> & queries,
		const DistanceType r,
		std::vector<int> & rNeighbors,
		std::vector<DistanceType> & neighborDistances) {

	}
private:
	inline void updateNearest (State & state)
	{
		const Node * node = state.currentNode;
		for (std::size_t i = node->beginIndex; i < node->endIndex; i++) {
			DistanceType dist = distance (
				state.query, _tree.getData() + i * _tree.getDim());
			if (dist < state.nearestDistance[0]) {
				state.nearestDistance[0] = dist;
				state.nearestNeighbor[0] = _tree.getIndices()[i];
			}
		}
	}
	inline DistanceType distance (const ElementType * a, const ElementType * b)
	{
		DistanceType dist = 0.0;
		for (std::size_t i = 0; i < _tree.getDim(); i++) {
			DistanceType temp = a[i] - b[i];
			temp *= temp;
			dist += temp;
		}
		return dist;
	}
	inline bool backtrack (State & state)
	{
		while (!state.farNodes.empty() &&
			state.farNodes.top().dist >= state.nearestDistance[0]) {
			state.farNodes.pop();
		}
		if (state.farNodes.empty())
			return false;
		else {
			state.currentNode = state.farNodes.top().node;
			state.farNodes.pop();
			return true;
		}
	}
	inline void setNext (
		const std::size_t nextQuery,
		State & state)
	{
		if (nextQuery < _numQueries) {
			state.currentNode = _tree.getRoot();
			state.query = _queries + nextQuery * _tree.getDim();
			state.nearestNeighbor = _nearestNeighbors + nextQuery;
			state.nearestDistance = _nearestDistances + nextQuery;
		} else
			state.active = false;
	}

	inline void descend (State & state) {
		Node * near, * far;
		const Node * node = state.currentNode;
		if (state.query[state.currentNode->splitDim] <
			state.currentNode->splitVal) {
			near = node->left;
			far = node->right;
		} else {
			near = node->right;
			far = node->left;
		}

		DistanceType distToSplit = 
			state.query[state.currentNode->splitDim] -
			state.currentNode->splitVal;
		distToSplit *= distToSplit;
		state.farNodes.push(NodeDistPair(far, distToSplit));

		state.currentNode = near;

	}

	const ElementType * _queries;
	int * _nearestNeighbors;
	DistanceType * _nearestDistances;
	std::size_t _numQueries;
	std::size_t _k;
	std::vector<State> _states;
};

} // namespace pointkd

#endif
