#ifndef __BOX3_H__
#define __BOX3_H__
#include <iostream>
#include <limits>
#include <vector>

namespace vltools {

template <typename T>
struct Box3 {
  Box3() {
    _data[0][0] = _data[0][1] = _data[0][2] = std::numeric_limits<T>::max();
    _data[1][0] = _data[1][1] = _data[1][2] = -std::numeric_limits<T>::max();
  }
  Box3(const Box3<T>& box) {
    _data[0][0] = box._data[0][0];
    _data[0][1] = box._data[0][1];
    _data[0][2] = box._data[0][2];
    _data[1][0] = box._data[1][0];
    _data[1][1] = box._data[1][1];
    _data[1][2] = box._data[1][2];
  }
  Box3(T xmin, T xmax, T ymin, T ymax, T zmin, T zmax) {
    _data[0][0] = xmin;
    _data[0][1] = ymin;
    _data[0][2] = zmin;
    _data[1][0] = xmax;
    _data[1][1] = ymax;
    _data[1][2] = zmax;
  }
  void addPoint(T x, T y, T z) {
    _data[0][0] = std::min(x, _data[0][0]);
    _data[0][1] = std::min(y, _data[0][1]);
    _data[0][2] = std::min(z, _data[0][2]);
    _data[1][0] = std::max(x, _data[1][0]);
    _data[1][1] = std::max(y, _data[1][1]);
    _data[1][2] = std::max(z, _data[1][2]);
  }
  void addPoints(const T* points, std::size_t numPoints) {
    for (std::size_t i = 0; i < numPoints; i++) {
      addPoint(points[3 * i + 0], points[3 * i + 1], points[3 * i + 2]);
    }
  }
  void addBox(const Box3<T>& other) {
    _data[0][0] = std::min(other.min(0), _data[0][0]);
    _data[0][1] = std::min(other.min(1), _data[0][1]);
    _data[0][2] = std::min(other.min(2), _data[0][2]);
    _data[1][0] = std::max(other.max(0), _data[1][0]);
    _data[1][1] = std::max(other.max(1), _data[1][1]);
    _data[1][2] = std::max(other.max(2), _data[1][2]);
  }

  // const getter methods
  const T& min(std::size_t i) const { return _data[0][i]; }
  const T& max(std::size_t i) const { return _data[1][i]; }
  const T& x(std::size_t i) const { return _data[i][0]; }
  const T& y(std::size_t i) const { return _data[i][1]; }
  const T& z(std::size_t i) const { return _data[i][2]; }
  // non-const versions
  T& min(std::size_t i) { return _data[0][i]; }
  T& max(std::size_t i) { return _data[1][i]; }
  T& x(std::size_t i) { return _data[i][0]; }
  T& y(std::size_t i) { return _data[i][1]; }
  T& z(std::size_t i) { return _data[i][2]; }
  T _data[2][3];
};

template <typename T>
T minDist2(const Box3<T>& a, const Box3<T>& b) {
  T dist = 0.0;
  for (std::size_t i = 0; i < 3; i++) {
    T temp = std::max(a.min(i) - b.max(i), (T)0.0);
    temp = std::max(b.min(i) - a.max(i), temp);
    dist += temp * temp;
  }
  return dist;
}

template <typename T>
T maxDist2(const Box3<T>& a, const Box3<T>& b) {
  T dist = 0.0;
  for (std::size_t i = 0; i < 3; i++) {
    T temp = a.max(i) - b.min(i);
    temp = std::max(b.max(i) - a.min(i), temp);
    dist += temp * temp;
  }
  return dist;
}

template <typename T>
void minDistVec(T* vec, const Box3<T>& a, const Box3<T>& b) {
  for (std::size_t i = 0; i < 3; i++) {
    T temp = std::max(a.min(i) - b.max(i), (T)0.0);
    temp = std::max(b.min(i) - a.max(i), temp);
    vec[i] = temp;
  }
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const Box3<T>& b) {
  os << "["
     << b.min(0) << ", " << b.max(0) << "]x["
     << b.min(1) << ", " << b.max(1) << "]x["
     << b.min(2) << ", " << b.max(2) << "]";
  return os;
}
}  // namespace vltools
#endif  // __BOX3_H__
