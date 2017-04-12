/** TODO: license boiler plate here
  *
  * By Victor Lu (victor.1.lu@here.com)
*/

#ifndef __ACCUMULATOR_H__
#define __ACCUMULATOR_H__

#include <type_traits>

namespace pointkd {
// specify how element types are converted to floating point number
template <typename T, bool is_int = std::is_integral<T>::value>
struct Accumulator {};

template <typename T>
struct Accumulator<T, true> {
  typedef float Type;
};
template <>
struct Accumulator<float, false> {
  typedef float Type;
};
template <>
struct Accumulator<double, false> {
  typedef double Type;
};
}  // namespace pointkd

#endif  //__ACCUMULATOR_H__
