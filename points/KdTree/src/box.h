/** TODO: license boiler plate here
  *
  * By Victor Lu (victor.1.lu@here.com)
*/

#ifndef __BOX_H__
#define __BOX_H__

#include <iostream>
#include <limits>
#include <vector>
#include "accumulator.h"

#ifdef max
#undef max
#endif

#ifdef min
#undef min
#endif

namespace pointkd {
template <typename T, int dim>
class Box {
 public:
  Box() { Reset(); }

  Box(const std::vector<T>& points) {
    Reset();
    AddPoints(points);
  }

  Box(const std::vector<T>& points, const std::vector<int>& indices) {
    Reset();
    AddPoints(points, indices);
  }

  Box(const T* points, const std::vector<int>& indices) {
    Reset();
    AddPoints(points, indices);
  }

  Box(const T* points, std::size_t num_points) {
    Reset();
    AddPoints(points, num_points);
  }

  Box(const Box<T, dim>& other) {
    for (int i = 0; i < dim; i++) {
      min_[i] = other.min_[i];
      max_[i] = other.max_[i];
    }
  }

  template <typename U>
  Box(const Box<U, dim>& other) {
    for (int i = 0; i < dim; i++) {
      min_[i] = (U)other.min(i);
      max_[i] = (U)other.max(i);
      if (min_[i] > max_[i])  // guard against numerical errors that may have
                              // occurred during type conversion from T->U
        std::swap(min_[i], max_[i]);
    }
  }

  ~Box() {}

  Box& operator=(const Box<T, dim>& other) {
    for (int i = 0; i < dim; i++) {
      min_[i] = other.min_[i];
      max_[i] = other.max_[i];
    }
    return *this;
  }

  void AddPoints(const T* points, std::size_t num_points) {
    // assumes points is array of size num_points * dim
    for (std::size_t i = 0; i < num_points; i++) {
      this->AddPoint(&points[i * dim]);
    }
  }

  void AddPoints(const std::vector<T>& points) {
    // assumes points.size() % dim == 0
    AddPoints(&points[0], points.size() / dim);
  }

  void AddPoints(const T* points, const std::vector<int>& indices) {
    for (std::size_t i = 0; i < indices.size(); i++) {
      // assumes indices[i]*dim+(0...dim-1) is not out of bounds
      this->AddPoint(&points[indices[i] * dim]);
    }
  }

  void AddPoints(const std::vector<T>& points,
                 const std::vector<int>& indices) {
    for (std::size_t i = 0; i < indices.size(); i++) {
      // assumes indices[i] is between 0 and points.size() / dim
      this->AddPoint(&points[indices[i] * dim]);
    }
  }

  void AddPoint(const T* point) {
    // point is array of type T[dim]
    for (int i = 0; i < dim; i++) {
      min_[i] = std::min(point[i], min_[i]);
      max_[i] = std::max(point[i], max_[i]);
    }
  }

  void Reset() {
    for (int i = 0; i < dim; i++) min_[i] = std::numeric_limits<T>::max();
    for (int i = 0; i < dim; i++) max_[i] = std::numeric_limits<T>::lowest();
  }

  bool IsPoint() const {
    bool is_point = true;
    for (int i = 0; i < dim; i++) {
      if (min_[i] != max_[i]) {
        is_point = false;
        break;
      }
    }
    return is_point;
  }

  template <typename U>
  bool Contains(const U* point) const {
    for (int i = 0; i < dim; i++) {
      if ((T)point[i] < min_[i]) return false;
      if ((T)point[i] > max_[i]) return false;
    }
    return true;
  }

  // const getter methods
  double width(int i) const { return (double)max_[i] - (double)min_[i]; }
  T min(int i) const { return min_[i]; }
  T max(int i) const { return max_[i]; }
  T val(int i, int side) const { return (side == 0 ? min_[i] : max_[i]); }

  // non-const getter methods
  T& min(int i) { return min_[i]; }
  T& max(int i) { return max_[i]; }
  T& val(int i, int side) {
    if (side == 0)
      return min_[i];
    else
      return max_[i];
  }

 private:
  T min_[dim];
  T max_[dim];
};  // class Box

template <typename T, typename Q, int dim, typename DistT>
void MinDist2Vec(DistT* v, const Q* q, const Box<T, dim>& box) {
  for (int i = 0; i < dim; i++) {
    DistT d_min = (DistT)box.min(i) - (DistT)q[i];
    DistT d_max = (DistT)box.max(i) - (DistT)q[i];
    if (d_min > (DistT)0.0)
      v[i] = d_min * d_min;
    else if (d_max < (DistT)0.0)
      v[i] = d_max * d_max;
    else
      v[i] = (DistT)0.0;
  }
}

template <typename T, typename Q, int dim, typename DistT>
void MaxDist2Vec(DistT* v, const Q* q, const Box<T, dim>& box) {
  for (int i = 0; i < dim; i++) {
    DistT d_min = (DistT)box.min(i) - (DistT)q[i];
    d_min = d_min * d_min;
    DistT d_max = (DistT)box.max(i) - (DistT)q[i];
    d_max = d_max * d_max;
    v[i] = std::max(d_min, d_max);
  }
}

template <typename Q, typename T, int dim, typename DistT>
DistT MinDist2(const Q* p, const Box<T, dim>& box) {
  DistT min_vector[dim];
  MinDist2Vec<T, Q, dim, DistT>(min_vector, p, box);
  DistT dist2 = (DistT)0.0;
  for (int i = 0; i < dim; i++) dist2 += min_vector[i];
  return dist2;
}

template <typename Q, typename T, int dim, typename DistT>
DistT MaxDist2(const Q* p, const Box<T, dim>& box) {
  DistT max_vector[dim];
  MaxDist2Vec<T, Q, dim, DistT>(max_vector, p, box);
  DistT dist2 = (DistT)0.0;
  for (int i = 0; i < dim; i++) dist2 += max_vector[i];
  return dist2;
}

template <typename T, int dim>
typename Accumulator<T>::Type MinDist2(const Box<T, dim>& a,
                                       const Box<T, dim>& b) {
  typedef typename Accumulator<T>::Type DistT;
  DistT dist = 0.0;
  for (int i = 0; i < dim; i++) {
    DistT temp = std::max((DistT)b.min(i) - (DistT)a.max(i),
                          (DistT)a.min(i) - (DistT)b.max(i));
    temp = std::max(temp, (DistT)0.0);  // temp is 0.0 => intervals overlap
    dist += temp * temp;
  }
  return dist;
}

template <typename T, int dim>
typename Accumulator<T>::Type MaxDist2(const Box<T, dim>& a,
                                       const Box<T, dim>& b) {
  typedef typename Accumulator<T>::Type DistT;
  DistT dist = 0.0;
  for (int i = 0; i < dim; i++) {
    T temp = std::max((DistT)a.max(i) - (DistT)b.min(i),
                      (DistT)b.max(i) - (DistT)a.min(i));
    dist += temp * temp;
  }
  return dist;
}

template <typename T, int dim>
void MinDistVec(typename Accumulator<T>::Type* vec, const Box<T, dim>& a,
                const Box<T, dim>& b) {
  // assumes vec has already been allocated
  typedef typename Accumulator<T>::Type DistT;
  for (int i = 0; i < dim; i++) {
    DistT temp = std::max((DistT)b.min(i) - (DistT)a.max(i),
                          (DistT)a.min(i) - (DistT)b.max(i));
    temp = std::max(temp, (DistT)0.0);  // temp is 0.0 => intervals overlap
    vec[i] = temp;
  }
}

template <typename T, int dim>
std::ostream& operator<<(std::ostream& os, const Box<T, dim>& b) {
  os << "[" << b.min(0) << ", " << b.max(0) << "]x[" << b.min(1) << ", "
     << b.max(1) << "]x[" << b.min(2) << ", " << b.max(2) << "]";
  return os;
}
}  // namespace pointkd

#endif  // __BOX_H__
