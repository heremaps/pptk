#ifndef __PQ_H__
#define __PQ_H__
#include <algorithm>
#include <functional>
#include <vector>

namespace vltools {

template <typename T, class Container = std::vector<T>,
	class Compare = std::less<T> >
class PriorityQueue
{
	Container _heap;
public:
	PriorityQueue () {}
	PriorityQueue (const Container & v)
	{
		_heap = v;
		std::make_heap (_heap.begin(), _heap.end(), Compare());
	}
	void push (const T & item)
	{
		_heap.push_back (item);
		std::push_heap (_heap.begin(), _heap.end(), Compare());
	}
	void pop ()
	{
		std::pop_heap (_heap.begin(), _heap.end(), Compare());
		_heap.pop_back ();
	}
	const T & top ()
	{
		return _heap[0];
	}
	void clear ()
	{
		_heap.clear();
	}
	bool empty ()
	{
		return _heap.empty();
	}
	size_t size ()
	{
		return _heap.size();
	}
};
} // namespace vltools
#endif
