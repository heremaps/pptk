#ifndef __STACK_H__
#define __STACK_H__
// this is simply an adapter of the stl vector class
// not using std::stack because I want fixed size stack
// implementation is fragile, not suitable for general usage
#include <vector>

namespace vltools {

template <typename T>
class Stack {
public:
	Stack () : _numItems(0) {}
	std::size_t size() {
		return _numItems;
	}
	std::size_t capacity() {
		return _data.size();
	}
	void clear() {
		_numItems = 0;
	}
	void resize(std::size_t size) {
		// resizes stack capacity
		_data.resize(size);
	}

	T & top() {
		// assumes user does not call this when stack is empty
		return _data[_numItems-1];
	}
	void pop() {
		// assumes user does not pop when stack is empty
		_numItems--;
	}
	void push(const T & x) {
		// assumes user does not push when capacity is reached
		_data[_numItems++] = x;
	}
	bool empty() {
		return _numItems == 0;
	}
private:
	std::vector<T> _data;
	std::size_t _numItems;
};
} // namespace vltools
#endif
