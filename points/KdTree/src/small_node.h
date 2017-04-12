/** TODO: license boiler plate here
  *
  * By Victor Lu (victor.1.lu@here.com)
*/

#ifndef __SMALL_NODE_H__
#define __SMALL_NODE_H__

#include "accumulator.h"

namespace pointkd {

template <typename T>
struct SmallNode {
  SmallNode() : split_value(0), split_index(0), child_offset(0) {}
  SmallNode(T split_value_, int split_dim_, int split_index_, int child_offset_,
            bool has_left_child, bool has_right_child)
      : split_value(split_value_),
        split_index(split_index_),
        child_offset(child_offset_) {
    split_index = (split_index << 3) | (split_dim_ & 7);
    child_offset = (child_offset << 2);
    child_offset |= has_left_child ? 2 : 0;
    child_offset |= has_right_child ? 1 : 0;
  }
  int GetSplitDim() const { return (int)(split_index & 7); }
  int GetSplitIndex() const { return (int)(split_index >> 3); }
  int LeftChildIndex(int current_index) const {
    if (child_offset & 2)
      return (child_offset >> 2) + current_index;
    else
      return -1;
  }
  int RightChildIndex(int current_index) const {
    if ((child_offset & 3) == 3)
      return (child_offset >> 2) + 1 + current_index;
    else if ((child_offset & 3) == 1)
      return (child_offset >> 2) + current_index;
    else
      return -1;
  }
  T split_value;
  unsigned int split_index;   // last 3 bits used to encode split dim
  unsigned int child_offset;  // last 2 bits used to encode the presence of a
                              // left and right child node
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const SmallNode<T>& node) {
  os << "SmallNode(dim=" << node.GetSplitDim() << ", value=" << node.split_value
     << ", index=" << node.GetSplitIndex()
     << ", left_offset=" << node.LeftChildIndex(0)
     << ", right_offset=" << node.RightChildIndex(0) << ")";
  return os;
}

}  // namespace pointkd

#endif  // __SMALL_NODE_H__