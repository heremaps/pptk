#ifndef _FLOATCAST_H_
#define _FLOATCAST_H_
#include <stdint.h>
namespace vltools {

// specifies that all integer types are converted to float,
// float to float, and double to double.
template <typename T>
struct FloatCast {typedef T Type;};
template <>
struct FloatCast<uint8_t> {typedef float Type;};
template <>
struct FloatCast<uint16_t> {typedef float Type;};
template <>
struct FloatCast<uint32_t> {typedef float Type;};
template <>
struct FloatCast<uint64_t> {typedef float Type;};
template <>
struct FloatCast<int8_t> {typedef float Type;};
template <>
struct FloatCast<int16_t> {typedef float Type;};
template <>
struct FloatCast<int32_t> {typedef float Type;};
template <>
struct FloatCast<int64_t> {typedef float Type;};

} // namespace vltools

#endif
