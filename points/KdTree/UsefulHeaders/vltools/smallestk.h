#ifndef _SMALLESTK_H_
#define _SMALLESTK_H_

#include "pq.h"
namespace vltools {

template <typename T>
class SmallestK : private PriorityQueue<T> {
// note: using private inheritance, so that users use bool push(...)
// instead of void push(...)
	typedef PriorityQueue<T> PQT;
	std::size_t _k;
public:
	SmallestK() : _k(0) {}
	SmallestK(std::size_t k) : _k(k) {}

	bool push(const T & item) {
		if (PQT::size() < _k) {
			PQT::push(item);
			return true;
		} else if (PQT::size() == _k && item < PQT::top()) {
			PQT::pop();
			PQT::push(item);
			return true;
		} else {
			return false;
		}
	}

	void popAll (std::vector<T> & v) {
		while (!PQT::empty()) {
			v.push_back(PQT::top());
			PQT::pop();
		}
	}
	void pop() {PQT::pop();}
	const T & top() {return PQT::top();}
	void clear() {PQT::clear();}
	bool empty() {return PQT::empty();}
	std::size_t size() {return PQT::size();}
	std::size_t k() {return _k;}
};

} // namespace vltools
#endif
