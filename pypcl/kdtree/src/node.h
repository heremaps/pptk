/** TODO: license boiler plate here
  *
  * By Victor Lu (victor.1.lu@here.com)
*/

#ifndef __NODE_H__
#define __NODE_H__

#include <cstdint>
#include <iostream>
#include "accumulator.h"

namespace pointkd {
template <typename T>
struct Node {
  Node() {}
  Node(T split_val, int split_dim, int split_idx, Node<T>* left, Node<T>* right)
      : split_value(split_val),
        split_dim(split_dim),
        split_index(split_idx),
        left(left),
        right(right) {}

  T split_value;
  unsigned int split_dim : 3;
  unsigned int split_index : 29;
  Node<T>* left;
  Node<T>* right;
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const Node<T>& node) {
  os << "Split(addr=" << &node << ", dim=" << node.split_dim
     << ", val=" << node.split_value << ", idx=" << node.split_index
     << ", left=" << node.left << ", right=" << node.right << ")";
  return os;
}
}  // namespace pointkd

#endif  // __NODE_H__