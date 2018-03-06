#ifndef __POINTATTRIBUTES_H__
#define __POINTATTRIBUTES_H__
#include <QVector3D>
#include <QVector4D>
#include <QtGlobal>
#include <algorithm>
#include <vector>
#include "octree.h"

class PointAttributes {
 private:
  std::vector<std::vector<float> > _attr;
  std::vector<quint64> _attr_size;
  std::vector<quint64> _attr_dim;
  std::size_t _curr_idx;

 public:
  PointAttributes()
      : _attr(1, std::vector<float>(4, 1.0f)),
        _attr_size(1, 1),
        _attr_dim(1, 4),
        _curr_idx(0) {}

  bool set(const std::vector<float>& attr, quint64 attr_size,
           quint64 attr_dim) {
    if (attr.size() != attr_size * attr_dim) return false;
    _attr.clear();
    _attr_size.clear();
    _attr_dim.clear();
    _attr.resize(1, attr);
    _attr_size.resize(1, attr_size);
    _attr_dim.resize(1, attr_dim);
    _curr_idx = 0;
    return true;
  }

  bool set(const std::vector<char>& data, const Octree& octree) {
    // overwrites existing attributes
    unsigned int num_points = octree.getNumPoints();

    // fill in _attr* arrays
    if (!_unpack(data, num_points)) return false;

    // nothing to do if there are no points in octree
    if (num_points == 0) return true;

    // set points to white if no attributes received
    if (_attr.empty()) {
      reset();
      return true;
    }

    // for each attribute set
    for (std::size_t i = 0; i < _attr.size(); i++) {
      if (_attr_size[i] == 1) continue;

      // reorder attributes according to octree
      _reorder(i, octree);

      // compute LOD averages
      _compute_LOD(i, octree);
    }
    _curr_idx = 0;
    return true;
  }

  void reset() {
    _attr.clear();
    _attr_size.clear();
    _attr_dim.clear();
    _attr.resize(1, std::vector<float>(4, 1.0f));
    _attr_size.resize(1, 1);
    _attr_dim.resize(1, 4);
    _curr_idx = 0;
  }

  const std::vector<float>& operator[](int i) const { return _attr[i]; }

  float operator()(int i, int j) const {
    // return j-th component of i-th attribute in current attribute set
    quint64 dim = _attr_dim[_curr_idx];
    quint64 size = _attr_size[_curr_idx];
    if (size == 1)
      return _attr[_curr_idx][j];
    else
      return _attr[_curr_idx][i * dim + j];
  }

  float operator()(int k, int i, int j) const {
    // return j-th component of i-th attribute in k-th attribute set
    quint64 dim = _attr_dim[k];
    quint64 size = _attr_size[k];
    if (size == 1)
      return _attr[k][j];
    else
      return _attr[k][i * dim + j];
  }

  std::size_t currentIndex() const { return _curr_idx; }
  quint64 size(int i) const { return _attr_size[i]; }
  quint64 dim(int i) const { return _attr_dim[i]; }
  std::size_t numAttributes() const { return _attr.size(); }
  void setCurrentIndex(std::size_t i) {
    if (i < _attr.size() && i >= 0) _curr_idx = i;
  }

 private:
  bool _unpack(const std::vector<char>& data, unsigned int expected_size) {
    if (data.empty()) return false;

    // initialize ptr into data stream
    const char* ptr = (char*)&data[0];
    const char* ptr_end = ptr + data.size();

    // get number of attribute sets
    quint64 num_attr;
    if (!_unpack_number(num_attr, ptr, ptr_end)) return false;

    // parse attribute sets
    std::vector<std::vector<float> > attr(num_attr);
    std::vector<quint64> attr_size(num_attr);
    std::vector<quint64> attr_dim(num_attr);
    for (quint64 i = 0; i < num_attr; i++) {
      // record attribute size of current set
      if (!_unpack_number(attr_size[i], ptr, ptr_end)) return false;
      if (attr_size[i] != expected_size && attr_size[i] != 1) return false;

      // record attribute dimension of current set
      if (!_unpack_number(attr_dim[i], ptr, ptr_end)) return false;

      // record attribute values
      attr[i].resize(attr_size[i] * attr_dim[i]);
      if (!_unpack_array(attr[i], ptr, ptr_end)) return false;
    }
    _attr.swap(attr);
    _attr_size.swap(attr_size);
    _attr_dim.swap(attr_dim);
    return true;
  }

  void _reorder(std::size_t attr_idx, const Octree& octree) {
    // assumes _attr_size[attr_idx] > 1
    std::vector<float>& attr = _attr[attr_idx];
    std::vector<float> temp(attr.size());
    quint64 dim = _attr_dim[attr_idx];
    for (quint64 j = 0; j < _attr_size[attr_idx]; j++)
      for (quint64 k = 0; k < dim; k++)
        temp[j * dim + k] = attr[octree.getIndices()[j] * dim + k];
    attr.swap(temp);
  }

  void _compute_LOD(std::size_t attr_idx, const Octree& octree) {
    // make space for LOD averages
    std::size_t num_centroids =
        octree.getPointPos().size() / 3 - octree.getNumPoints();
    _attr[attr_idx].resize(_attr[attr_idx].size() +
                           _attr_dim[attr_idx] * num_centroids);
    if (false && _attr_dim[attr_idx] == 4)
      // disable LOD approximation of alpha transparency for now
      _compute_rgba_LOD_helper(attr_idx, octree.getRoot());
    else
      _compute_LOD_helper(attr_idx, octree.getRoot());
  }

  void _compute_rgba_LOD_helper(std::size_t attr_idx,
                                const Octree::Node* node) {
    if (node == NULL) return;

    std::vector<float>& attr = _attr[attr_idx];
    quint64 dim = _attr_dim[attr_idx];
    float* dst = &attr[dim * node->centroid_index];
    dst[0] = dst[1] = dst[2] = 0.0f;
    dst[3] = 1.0f;
    if (node->is_leaf) {
      QVector3D x = QVector3D(0.0f, 0.0f, 0.0f);
      float w = 1.0f;
      for (unsigned int i = 0; i < node->point_count; i++)
        _accumulate_rgba(x, w, &attr[4 * (node->point_index + i)]);
      _xw_to_rgba(dst, x, w, node->point_count);
    } else {  // !node->is_leaf
      //_compute_inner_rgba(dst, node);
      QVector3D x = QVector3D(0.0f, 0.0f, 0.0f);
      float w = 1.0f;
      unsigned int num_children = 0;
      for (unsigned int i = 0; i < 8; i++) {
        if (node->children[i] == NULL) continue;
        num_children++;
        _compute_rgba_LOD_helper(attr_idx, node->children[i]);
        _accumulate_rgba(x, w, &attr[4 * node->children[i]->centroid_index]);
      }
      _xw_to_rgba(dst, x, w, num_children);
    }
  }

  void _compute_LOD_helper(std::size_t attr_idx, const Octree::Node* node) {
    if (node == NULL) return;

    std::vector<float>& attr = _attr[attr_idx];
    quint64 dim = _attr_dim[attr_idx];
    float* dst = &attr[dim * node->centroid_index];
    std::fill_n(dst, dim, 0.0f);
    if (node->is_leaf) {
      for (unsigned int i = 0; i < node->point_count; i++)
        for (quint64 j = 0; j < dim; j++)
          dst[j] += attr[(node->point_index + i) * dim + j];
      for (quint64 i = 0; i < dim; i++) dst[i] /= node->point_count;
    } else {  // !node->is_leaf
      std::vector<float> w(8, 0.0f);
      for (std::size_t i = 0; i < 8; i++) {
        if (!node->children[i]) continue;
        w[i] = (float)node->children[i]->point_count / node->point_count;
      }
      for (unsigned int i = 0; i < 8; i++) {
        Octree::Node* child = node->children[i];
        if (!child) continue;
        _compute_LOD_helper(attr_idx, child);
        float w = (float)node->children[i]->point_count / node->point_count;
        for (quint64 j = 0; j < dim; j++)
          dst[j] += w * attr[child->centroid_index * dim + j];
      }
    }
  }

  inline void _accumulate_rgba(QVector3D& x, float& w, const float* v) {
    for (int i = 0; i < 3; i++) x[i] += v[i];
    w *= (1.0f - v[3]);
  }

  inline void _xw_to_rgba(float* dst, const QVector3D& x, const float& w,
                          unsigned int n) const {
    dst[0] = x[0] / n;
    dst[1] = x[1] / n;
    dst[2] = x[2] / n;
    dst[3] = 1.0f - w;
  }

  template <typename T>
  bool _unpack_number(T& v, const char*& ptr, const char* const ptr_end) {
    // returns false if attempting to read beyond end of stream [ptr, ptr_end)
    if (ptr + sizeof(T) > ptr_end) {
      return false;
    } else {
      v = *(T*)ptr;
      ptr += sizeof(T);
      return true;
    }
  }

  template <typename T>
  bool _unpack_array(std::vector<T>& v, const char*& ptr,
                     const char* const ptr_end) {
    if (ptr + sizeof(T) * v.size() > ptr_end) {
      return false;
    } else {
      std::copy((const T*)ptr, (const T*)(ptr + sizeof(T) * v.size()),
                v.begin());
      ptr += sizeof(T) * v.size();
      return true;
    }
  }
};

#endif  // __POINTATTRIBUTES_H__
