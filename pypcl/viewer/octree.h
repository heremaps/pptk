#ifndef __OCTREE_H__
#define __OCTREE_H__

#include <algorithm>
#include <limits>
#include <vector>
#include "box3.h"
#include "camera.h"

class Octree {
 public:
  struct Node {
    unsigned int centroid_index;
    unsigned int point_index;
    unsigned int point_count;
    bool is_leaf;
    Node* children[8];  // index given by 3-bit code: xyz,
                        // where z is least significant bit
  };

  struct CameraFrustum {
    float eye[3];
    float right[3];
    float up[3];
    float view[3];
    float image_r;
    float image_t;
    float z_near;  // z_near < 0

    void setImagePlane(float vfov, float aspect_ratio) {
      // vfov in radians
      image_t = tan(0.5f * vfov);
      image_r = aspect_ratio * image_t;
    }

    void xyz2ruv(float (&ruv)[3], const float (&xyz)[3]) const {
      // note: handles case where xyz and ruv refer to same vector
      float xyz_[3];
      for (unsigned int dim = 0; dim < 3; dim++) {
        xyz_[dim] = xyz[dim] - eye[dim];
      }
      ruv[0] = ruv[1] = ruv[2] = 0.0f;
      for (unsigned int k = 0; k < 3; k++) ruv[0] += right[k] * xyz_[k];
      for (unsigned int k = 0; k < 3; k++) ruv[1] += up[k] * xyz_[k];
      for (unsigned int k = 0; k < 3; k++) ruv[2] += view[k] * xyz_[k];
    }

    static void setupCameraFrustum(CameraFrustum& frustum, const Camera& camera,
                                   const float z_near, const float vfov,
                                   const float aspect_ratio) {
      camera.getCameraPosition(frustum.eye);
      camera.getRightVector(frustum.right);
      camera.getUpVector(frustum.up);
      camera.getViewVector(frustum.view);
      frustum.setImagePlane(vfov, aspect_ratio);
      frustum.z_near = z_near;
    }

    static void setupOrthoCamera(CameraFrustum& frustum, const Camera& camera,
                                 const float vfov, const float aspect_ratio) {
      camera.getCameraPosition(frustum.eye);
      camera.getRightVector(frustum.right);
      camera.getUpVector(frustum.up);
      camera.getViewVector(frustum.view);
      frustum.setImagePlane(vfov, aspect_ratio);
      frustum.image_r *= camera.getCameraDistance();
      frustum.image_t *= camera.getCameraDistance();
      frustum.z_near = std::numeric_limits<float>::max();
    }
  };

  Octree()
      : _max_leaf_size(64),
        _num_points(0),
        _root(NULL),
        _ptr_point_xyz(NULL),
        _cube_size(0.0f) {
    _lower_left_corner[0] = _lower_left_corner[1] = _lower_left_corner[2] =
        0.0f;
  }

  ~Octree() { deleteTree(_root); }

  const Node* getRoot() const { return _root; }
  // TODO: getPointPos() segfaults if called before buildTree()
  const std::vector<float>& getPointPos() const { return *_ptr_point_xyz; }
  const std::vector<unsigned int>& getIndices() const { return _indices; }
  const std::vector<unsigned int>& getIndicesR() const { return _indices_r; }

  unsigned int countNodes() {
    // sanity check: two ways of computing the same thing
    // return (*_ptr_point_xyz).size() / 3 - _num_points;
    return countNodesHelper(_root);
  }
  unsigned int countInnerNodes() { return countNodesHelper(_root, false); }

  unsigned int getNumPoints() const { return _num_points; }

  void buildTree(std::vector<float>& point_xyz, std::vector<float>& point_size,
                 unsigned int max_leaf_size = 64) {
    deleteTree(_root);

    _max_leaf_size = max_leaf_size;
    _num_points = (unsigned int)point_xyz.size() / 3;
    _ptr_point_xyz = &point_xyz;
    _ptr_point_size = &point_size;

    // let non-centroid points have zero size
    point_size.resize(_num_points, 0.0f);

    if (_num_points == 0) {
      _root = NULL;
      return;
    }

    // compute bounding cube
    vltools::Box3<float> box;
    box.addPoints(&point_xyz[0], _num_points);
    float cube_center[3] = {0.5f * (box.x(0) + box.x(1)),
                            0.5f * (box.y(0) + box.y(1)),
                            0.5f * (box.z(0) + box.z(1))};
    float cube_size =
        (std::max)(box.x(1) - box.x(0),
                   (std::max)(box.y(1) - box.y(0), box.z(1) - box.z(0)));
    float cube_corner[3] = {cube_center[0] - 0.5f * cube_size,
                            cube_center[1] - 0.5f * cube_size,
                            cube_center[2] - 0.5f * cube_size};
    for (unsigned int dim = 0; dim < 3; dim++)
      _lower_left_corner[dim] = cube_corner[dim];
    _cube_size = cube_size;

    // prepare scratch space
    _indices.clear();
    _indices.reserve(_num_points);
    for (unsigned int i = 0; i < _num_points; i++) _indices.push_back(i);
    _labels.resize(_num_points);

    // begin recursion
    _root = buildTreeHelper(&_indices[0], &_labels[0], _num_points,
                            _lower_left_corner, _cube_size);

    // reorder points
#if 0
    std::vector<float> temp_xyz(3 * _num_points);
    for (unsigned int i = 0; i < _num_points; i++) {
      for (unsigned int dim = 0; dim < 3; dim++) {
        temp_xyz[3 * i + dim] = point_xyz[3 * _indices[i] + dim];
      }
    }
    temp_xyz.insert(temp_xyz.end(), &point_xyz[3 * _num_points], &point_xyz[0] + point_xyz.size());
    point_xyz.swap(temp_xyz);
#else
    _indices_r.resize(_num_points);
    std::vector<float> temp_xyz;
    temp_xyz.reserve(point_xyz.size());
    for (unsigned int i = 0; i < _num_points; i++) {
      for (unsigned int dim = 0; dim < 3; dim++) {
        temp_xyz.push_back(point_xyz[3 * _indices[i] + dim]);
      }
      _indices_r[_indices[i]] = i;
    }
    for (unsigned int i = _num_points; i < point_xyz.size() / 3; i++) {
      for (unsigned int dim = 0; dim < 3; dim++) {
        temp_xyz.push_back(point_xyz[3 * i + dim]);
      }
    }
    point_xyz.swap(temp_xyz);
#endif
  }

  enum ProjectionMode { PERSPECTIVE, ORTHOGRAPHIC };
  void getIndices(std::vector<unsigned int>& indices, const Camera& camera,
                  float vfov, float z_near, unsigned int width,
                  unsigned int height, float fudge_factor = 0.25f) const {
    indices.clear();
    if (!_root) return;

    // set frustum
    CameraFrustum frustum;
    camera.getCameraPosition(frustum.eye);
    camera.getRightVector(frustum.right);
    camera.getUpVector(frustum.up);
    camera.getViewVector(frustum.view);
    frustum.setImagePlane(vfov, (float)width / height);
    frustum.z_near = z_near;

    // compute eps (pixel size in image plane at z = -1)
    float eps = frustum.image_t * 2.0f / height;

    indices.reserve(_num_points);
    getIndicesHelper(indices, _root, frustum, _lower_left_corner, _cube_size,
                     eps, PERSPECTIVE, fudge_factor);
  }

  void getIndicesOrtho(std::vector<unsigned int>& indices, const Camera& camera,
                       float image_r, float image_t, unsigned int height,
                       float fudge_factor = 0.25f) const {
    indices.clear();
    if (!_root) return;

    // set frustum
    CameraFrustum frustum;
    camera.getCameraPosition(frustum.eye);
    camera.getRightVector(frustum.right);
    camera.getUpVector(frustum.up);
    camera.getViewVector(frustum.view);
    frustum.image_t = image_t;
    frustum.image_r = image_r;
    frustum.z_near = 0.0;  // doesn't matter for orthographic projection

    // compute eps (pixel size in image plane at z = -1)
    float eps = frustum.image_t * 2.0f / height;

    indices.reserve(_num_points);
    getIndicesHelper(indices, _root, frustum, _lower_left_corner, _cube_size,
                     eps, ORTHOGRAPHIC, fudge_factor);
  }

  void getClickIndices(std::vector<unsigned int>& indices, const float screen_x,
                       const float screen_y, const float screen_radius,
                       const float screen_width, const float screen_height,
                       const float vfov,
                       const float near_clip,  // positive, distance along -view
                       const Camera& camera,
                       const ProjectionMode projection_mode) const {
    // returned indices include:
    // 1. index of nearest point along -view whose projection
    //    is within screen_radius of (screen_x, screen_y)
    // 2. indices of centroids that "include" this index
    indices.clear();

    float d_min = std::numeric_limits<float>::max();
    CameraFrustum frustum;
    if (projection_mode == PERSPECTIVE)
      CameraFrustum::setupCameraFrustum(frustum, camera, -near_clip, vfov,
                                        screen_width / screen_height);
    else
      CameraFrustum::setupOrthoCamera(frustum, camera, vfov,
                                      screen_width / screen_height);

    // compute coordinates on z = -1 image plane
    float click_pos[2] = {
        (2.0f * screen_x / screen_width - 1.0f) * frustum.image_r,
        (2.0f * screen_y / screen_height - 1.0f) * frustum.image_t};
    float click_radius = screen_radius / screen_width * 2.0f * frustum.image_r;

    getClickIndicesHelper(indices, d_min, _root, _lower_left_corner, _cube_size,
                          click_pos, click_radius, frustum, projection_mode);

    // reverse indices[1..] to give sorted order
    if (indices.size() > 1)
      std::reverse(&indices[1], &indices[indices.size() - 1]);
  }

  void getClickIndicesBrute(
      std::vector<unsigned int>& indices, const float screen_x,
      const float screen_y, const float screen_radius, const float screen_width,
      const float screen_height, const float vfov,
      const float near_clip,  // positive, distance along -view
      const Camera& camera, const ProjectionMode projection_mode) const {
    float d_min = std::numeric_limits<float>::max();
    CameraFrustum frustum;
    if (projection_mode == PERSPECTIVE)
      CameraFrustum::setupCameraFrustum(frustum, camera, -near_clip, vfov,
                                        screen_width / screen_height);
    else
      CameraFrustum::setupOrthoCamera(frustum, camera, vfov,
                                      screen_width / screen_height);

    // compute coordinates on z = -1 image plane
    float click_pos[2] = {
        (2.0f * screen_x / screen_width - 1.0f) * frustum.image_r,
        (2.0f * screen_y / screen_height - 1.0f) * frustum.image_t};
    float click_radius = screen_radius / screen_width * 2.0f * frustum.image_r;

    int min_idx;
    std::vector<float>& points_xyz = *_ptr_point_xyz;
    getClickIndicesBruteHelper(min_idx, d_min, &points_xyz[0], _num_points,
                               click_pos, click_radius, frustum,
                               projection_mode);

    indices.clear();
    if (min_idx != -1) indices.push_back(min_idx);
  }

 private:
  static void traversalOrder(unsigned int (&nodeIndices)[8],
                             const float (&view)[3]) {
    int b[3][2];
    for (int dim = 0; dim < 3; dim++) {
      if (view[dim] > 0.0f) {
        b[dim][0] = 0;
        b[dim][1] = 1;
      } else {
        b[dim][0] = 1;
        b[dim][1] = 0;
      }
    }
    int idx = 0;
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++)
        for (int k = 0; k < 2; k++)
          nodeIndices[idx++] = 4 * b[0][i] + 2 * b[1][j] + b[2][k];
  }

  static float pointToPointDistanceSquared(const float (&x)[2],
                                           const float (&y)[2]) {
    float distance = 0.0f;
    for (int dim = 0; dim < 2; dim++) {
      float temp = x[dim] - y[dim];
      distance += temp * temp;
    }
    return distance;
  }

  static float dotProduct(const float (&x)[3], const float (&y)[3]) {
    float result = 0.0f;
    for (int dim = 0; dim < 3; dim++) result += x[dim] * y[dim];
    return result;
  }

  static float boxToPointDistance(const float (&box_center)[2],
                                  const float (&box_size)[2],
                                  const float (&point)[2]) {
    float d = 0.0f;
    for (int i = 0; i < 2; i++) {
      float a = box_center[i] - 0.5f * box_size[i];
      float b = box_center[i] + 0.5f * box_size[i];
      float x = (std::max)(0.0f, (std::max)(point[i] - b, a - point[i]));
      d += x * x;
    }
    return sqrtf(d);
  }

  static void projectNode(float& v_min, float& v_max,
                          const float (&node_corner)[3], const float node_size,
                          const float (&view)[3], const float (&eye)[3]) {
    v_min = v_max = 0.0f;
    for (int dim = 0; dim < 3; dim++) {
      float a = view[dim] * (node_corner[dim] - eye[dim]);
      float b = view[dim] * (node_corner[dim] + node_size - eye[dim]);
      if (view[dim] > 0.0f) {
        v_min += a;
        v_max += b;
      } else {
        v_min += b;
        v_max += a;
      }
    }
  }

  static void getClickIndicesBruteHelper(
      int& min_idx, float& d_min, const float* points, const int count,
      const float (&click_pos)[2], const float click_radius,
      const CameraFrustum& frustum, const ProjectionMode& projection_mode) {
    min_idx = -1;
    float click_radius_squared = click_radius * click_radius;
    for (int i = 0; i < count; i++) {
      float xyz_eye[3];
      for (int dim = 0; dim < 3; dim++)
        xyz_eye[dim] = points[3 * i + dim] - frustum.eye[dim];

      // compute point distance along -view vector
      float d = -dotProduct(xyz_eye, frustum.view);

      // skip near-clipped points
      if (d < -frustum.z_near) continue;

      // compute distance from point's projection to click position
      float point_proj[2] = {dotProduct(xyz_eye, frustum.right),
                             dotProduct(xyz_eye, frustum.up)};
      if (projection_mode == PERSPECTIVE) {
        point_proj[0] /= d;
        point_proj[1] /= d;
      }
      float point_proj_d_2 = pointToPointDistanceSquared(point_proj, click_pos);

      // skip if projection is too far from click
      if (point_proj_d_2 > click_radius_squared) continue;

      // point is in "selection cylinder"
      if (d < d_min) {
        d_min = d;
        min_idx = i;
      }
    }
  }

  void getClickIndicesHelper(
      std::vector<unsigned int>& indices, float& d_min, const Node* node,
      const float (&node_corner)[3], const float node_size,
      const float (&click_pos)[2],  // on z = -1 image plane
      const float click_radius, const CameraFrustum& frustum,
      const ProjectionMode projection_mode = PERSPECTIVE) const {
    if (!node) return;

    // calculate node center in camera coordinates
    float node_center[3];
    for (int dim = 0; dim < 3; dim++)
      node_center[dim] = node_corner[dim] + 0.5f * node_size;
    float node_center_ruv[3];
    frustum.xyz2ruv(node_center_ruv, node_center);

    // return if node is too far behind eye (camera faces -z direction)
    if (node_center_ruv[2] > node_size * sqrtf(3.0f) * 0.5f + frustum.z_near)
      return;

    // compute conservative bound around projection of node onto z = -1 plane
    float bound_center[2];
    float bound_size[2];
    if (projection_mode == PERSPECTIVE)
      boundProjectedAABB(bound_center, bound_size, node_center_ruv, node_size,
                         frustum.z_near);
    else
      boundOrthoProjectedAABB(bound_center, bound_size, node_center_ruv,
                              node_size);

    // return if node is too far from click position
    if (boxToPointDistance(bound_center, bound_size, click_pos) > click_radius)
      return;

    // compute projection along view
    float v_min, v_max;
    projectNode(v_min, v_max, node_corner, node_size, frustum.view,
                frustum.eye);

    // return if node cannot possibly contain a point closer than d_min
    if (-v_min > d_min) return;

    if (node->is_leaf) {
      std::vector<float>& point_xyz = *_ptr_point_xyz;
      int min_idx;
      getClickIndicesBruteHelper(
          min_idx, d_min, &point_xyz[3 * node->point_index], node->point_count,
          click_pos, click_radius, frustum, projection_mode);
      if (min_idx != -1) {
        // a closer point has been found
        indices.clear();
        indices.push_back(node->point_index + min_idx);
      }
    } else {
      unsigned int trav_order[8];
      if (projection_mode == PERSPECTIVE) {
        float node_center_eye[3] = {node_center[0] - frustum.eye[0],
                                    node_center[1] - frustum.eye[1],
                                    node_center[2] - frustum.eye[2]};
        traversalOrder(trav_order, node_center_eye);
      } else {
        traversalOrder(trav_order, frustum.view);
      }

      for (int i = 0; i < 8; i++) {
        unsigned int child_index = trav_order[i];
        Node* child_node = node->children[child_index];
        float child_node_corner[3] = {
            (child_index & 4) ? node_center[0] : node_corner[0],
            (child_index & 2) ? node_center[1] : node_corner[1],
            (child_index & 1) ? node_center[2] : node_corner[2]};
        getClickIndicesHelper(indices, d_min, child_node, child_node_corner,
                              0.5f * node_size, click_pos, click_radius,
                              frustum, projection_mode);
      }
      if (!indices.empty()) indices.push_back(node->centroid_index);
    }
  }

  Octree(const Octree&);  // prohibit copy constructor

  Octree& operator=(Octree);  // prohibit copy assignment constructor

  void computeCentroid(float (&pos)[3], const unsigned int* indices,
                       const unsigned int count) {
    std::vector<float>& point_xyz = *_ptr_point_xyz;
    pos[0] = pos[1] = pos[2] = 0.0f;
    for (unsigned int i = 0; i < count; i++) {
      for (unsigned int dim = 0; dim < 3; dim++) {
        pos[dim] += point_xyz[3 * indices[i] + dim];
      }
    }
    for (unsigned int dim = 0; dim < 3; dim++) {
      pos[dim] /= count;
    }
  }

  void partitionXYZ(unsigned int* child_counts, unsigned int* indices,
                    unsigned char* labels, unsigned int count,
                    unsigned int bit = 2) {
    unsigned int left = partition(indices, labels, count, bit);
    // bit: 2 - x, 1 - y, 0 - z
    if (bit == 0) {
      child_counts[0] = left;
      child_counts[1] = count - left;
    } else {
      partitionXYZ(child_counts, indices, labels, left, bit - 1);
      unsigned int offset = 1 << bit;
      partitionXYZ(child_counts + offset, indices + left, labels + left,
                   count - left, bit - 1);
    }
  }
  unsigned int partition(unsigned int* indices, unsigned char* labels,
                         const unsigned int count, const unsigned int bit) {
    if (count == 0) return 0;
    int left = 0;
    int right = count - 1;
    unsigned char mask = 1 << bit;
    for (;;) {
      while (left < (int)count && (labels[left] & mask) == 0) left++;
      while (right >= 0 && (labels[right] & mask)) right--;
      if (left > right) break;
      std::swap(indices[left], indices[right]);
      std::swap(labels[left], labels[right]);
      left++;
      right--;
    }
    return left;
  }
  unsigned int addCentroid(const float (&xyz)[3]) {
    std::vector<float>& point_xyz = *_ptr_point_xyz;
    unsigned int index = (unsigned int)point_xyz.size() / 3;
    for (unsigned int dim = 0; dim < 3; dim++) {
      point_xyz.push_back(xyz[dim]);
    }
    return index;
  }
  void computeCentroid(float (&centroid_xyz)[3], Node* children[]) {
    // compute centroid from child node centroids
    // assumes children are valid
    std::vector<float>& point_xyz = *_ptr_point_xyz;
    centroid_xyz[0] = centroid_xyz[1] = centroid_xyz[2] = 0.0f;
    unsigned int count = 0;
    for (unsigned int i = 0; i < 8; i++) {
      if (children[i] == NULL) continue;
      float* child_centroid_xyz = &point_xyz[3 * children[i]->centroid_index];
      unsigned int child_count = children[i]->point_count;
      for (unsigned int dim = 0; dim < 3; dim++) {
        centroid_xyz[dim] += child_centroid_xyz[dim] * child_count;
      }
      count += child_count;
    }
    for (unsigned int dim = 0; dim < 3; dim++) {
      centroid_xyz[dim] /= count;
    }
  }

  bool pointsAreIdentical(const unsigned int* indices,
                          const unsigned int count) {
    if (count == 1) return true;
    std::vector<float>& points = *_ptr_point_xyz;
    int first_index = indices[0];
    float first_point[3] = {points[3 * first_index + 0],
                            points[3 * first_index + 1],
                            points[3 * first_index + 2]};
    for (unsigned int i = 1; i < count; i++)
      for (int dim = 0; dim < 3; dim++)
        if (points[3 * indices[i] + dim] != first_point[dim]) return false;
    return true;
  }

  Node* buildTreeHelper(unsigned int* indices, unsigned char* labels,
                        const unsigned int count, const float (&cube_corner)[3],
                        const float cube_size) {
    std::vector<float>& point_xyz = *_ptr_point_xyz;
    std::vector<float>& point_size = *_ptr_point_size;
    Node* node;
    if (count == 0) {
      node = NULL;
    } else if (pointsAreIdentical(indices, count) || count <= _max_leaf_size) {
      // create leaf node
      node = new Node();
      node->point_index = indices - &_indices[0];
      node->point_count = count;
      node->is_leaf = true;

      // compute centroid
      float centroid_xyz[3];
      computeCentroid(centroid_xyz, indices, count);
      node->centroid_index = addCentroid(centroid_xyz);
      point_size.push_back(cube_size);

      return node;
    } else {
      // create inner node
      node = new Node();
      node->point_index = indices - &_indices[0];
      node->point_count = count;
      node->is_leaf = false;

      // compute cube center
      float cube_center[3];
      for (unsigned int dim = 0; dim < 3; dim++)
        cube_center[dim] = cube_corner[dim] + 0.5f * cube_size;

      // label points by child nodes (xyz encoding)
      for (unsigned int i = 0; i < count; i++) {
        labels[i] = 0;
        for (unsigned int dim = 0; dim < 3; dim++)
          labels[i] |=
              (point_xyz[3 * indices[i] + dim] > cube_center[dim] ? 1 : 0)
              << (2 - dim);
      }

      // partition points by labels
      unsigned int child_counts[8];
      partitionXYZ(child_counts, indices, labels, count);

      // recurse on children
      unsigned int* ptr_indices = indices;
      unsigned char* ptr_labels = labels;
      for (unsigned int i = 0; i < 8; i++) {
        float child_corner[3] = {
            (i & 4) == 0 ? cube_corner[0] : cube_center[0],
            (i & 2) == 0 ? cube_corner[1] : cube_center[1],
            (i & 1) == 0 ? cube_corner[2] : cube_center[2]};
        node->children[i] =
            buildTreeHelper(ptr_indices, ptr_labels, child_counts[i],
                            child_corner, 0.5f * cube_size);
        ptr_indices += child_counts[i];
        ptr_labels += child_counts[i];
      }

      // compute centroid
      float centroid_xyz[3] = {0.0f, 0.0f, 0.0f};
      computeCentroid(centroid_xyz, node->children);
      node->centroid_index = addCentroid(centroid_xyz);
      point_size.push_back(cube_size);
    }
    return node;
  }

  static void deleteTree(Node* root) {
    if (!root) return;
    if (root->is_leaf) {
      delete root;
    } else {
      for (int i = 0; i < 8; i++) deleteTree(root->children[i]);
    }
  }

  static void boundProjectedAABB(
      float (&bound_center)[2],  // on image plane
      float (&bound_size)[2],
      const float (&cube_center)[3],  // in ruv (camera) coordinates
      const float cube_size, float z_near) {
    // following assumes cube_center[2] < -sqrt(3)/2*cube_size
    // also assumes z_near < 0
    float delta = 0.5f * sqrtf(3.0f) * cube_size;
    float z_back = (std::min)(z_near, cube_center[2] - delta);
    float z_front = (std::min)(z_near, cube_center[2] + delta);
    for (int i = 0; i < 2; i++) {
      float right = (std::max)((cube_center[i] + delta) / -z_back,
                               (cube_center[i] + delta) / -z_front);
      float left = (std::min)((cube_center[i] - delta) / -z_back,
                              (cube_center[i] - delta) / -z_front);
      bound_center[i] = 0.5f * (right + left);
      bound_size[i] = (right - left);
    }
  }

  static void boundOrthoProjectedAABB(
      float (&bound_center)[2],  // on image plane
      float (&bound_size)[2],
      const float (&cube_center)[3],  // in ruv (camera) coordinates
      const float cube_size) {
    float delta = 0.5f * sqrtf(3.0f) * cube_size;
    for (int i = 0; i < 2; i++) {
      bound_center[i] = cube_center[i];
      bound_size[i] = 2.0f * delta;
    }
  }

  enum IntersectResult { OUTSIDE = 0, UNCERTAIN = 1, INSIDE = 2 };
  static IntersectResult intersectBoxes2D(const float (&box_1_center)[2],
                                          const float (&box_1_size)[2],
                                          const float (&box_2_center)[2],
                                          const float (&box_2_size)[2]) {
    if (box_1_center[0] + 0.5f * box_1_size[0] <
            box_2_center[0] - 0.5f * box_2_size[0] ||
        box_1_center[0] - 0.5f * box_1_size[0] >
            box_2_center[0] + 0.5f * box_2_size[0] ||
        box_1_center[1] + 0.5f * box_1_size[1] <
            box_2_center[1] - 0.5f * box_2_size[1] ||
        box_1_center[1] - 0.5f * box_1_size[1] >
            box_2_center[1] + 0.5f * box_2_size[1])
      return OUTSIDE;
    else if (box_1_center[0] + 0.5f * box_1_size[0] <
                 box_2_center[0] + 0.5f * box_2_size[0] &&
             box_1_center[0] - 0.5f * box_1_size[0] >
                 box_2_center[0] - 0.5f * box_2_size[0] &&
             box_1_center[1] + 0.5f * box_1_size[1] <
                 box_2_center[1] + 0.5f * box_2_size[1] &&
             box_1_center[1] - 0.5f * box_1_size[1] >
                 box_2_center[1] - 0.5f * box_2_size[1])
      return INSIDE;
    else
      return UNCERTAIN;
  }

  bool cubeInFrustum(const float (&cube_corner)[3], const float cube_size,
                     const CameraFrustum& frustum) {
    // note: does not work
    float cube_bounds[3][2] = {{cube_corner[0], cube_corner[0] + cube_size},
                               {cube_corner[1], cube_corner[1] + cube_size},
                               {cube_corner[2], cube_corner[2] + cube_size}};
    for (int k = 0; k < 2; k++) {
      for (int j = 0; j < 2; j++) {
        for (int i = 0; i < 2; i++) {
          float ruv[3];
          float corner[3] = {cube_bounds[0][i], cube_bounds[1][j],
                             cube_bounds[2][k]};
          frustum.xyz2ruv(ruv, corner);
          ruv[0] /= -ruv[2];
          ruv[1] /= -ruv[2];
          if (-ruv[2] >
                  0.0f &&  // note: view vector is opposite of view direction
              ruv[0] < frustum.image_r && ruv[0] > -frustum.image_r &&
              ruv[1] < frustum.image_t && ruv[1] > -frustum.image_t)
            return true;
        }
      }
    }
    return false;
  }

  void getIndicesHelper(
      std::vector<unsigned int>& indices, const Node* node,
      const CameraFrustum& frustum, const float (&cube_corner)[3],
      const float cube_size,
      const float eps,  // measured on z = 1 image plane
      const ProjectionMode projection_mode = PERSPECTIVE,
      const float fudge_factor =
          0.25f,  // factor by which projected area is reduced
      const IntersectResult parent_intersect_result = UNCERTAIN) const {
    // TODO: this results in redundant calls to xyz2ruv
    // consider performing frustum culling prior to recursing
    // bool in_frustum = cubeInFrustum(cube_corner, cube_size, frustum);
    // if (!in_frustum) return;

    // compute cube center
    float cube_center[3];
    for (int i = 0; i < 3; i++)
      cube_center[i] = cube_corner[i] + 0.5f * cube_size;

    // compute cube center in camera coordinates (ruv)
    float ruv[3];
    frustum.xyz2ruv(ruv, cube_center);

    // compute conservative bound on projected cube
    float bound_center[2];
    float bound_size[2];
    if (projection_mode == PERSPECTIVE)
      boundProjectedAABB(bound_center, bound_size, ruv, cube_size,
                         frustum.z_near);
    else
      boundOrthoProjectedAABB(bound_center, bound_size, ruv, cube_size);

    // test cube against frustum
    // note: at this point, parent_intersect_result is never OUTSIDE
    IntersectResult intersect_result;
    if (parent_intersect_result == INSIDE)
      // if parent was entirely in frustum,
      // then this node is also entirely in frustum
      intersect_result = INSIDE;
    else {  // parent_intersect_result == UNCERTAIN
      if (projection_mode == PERSPECTIVE &&
          -ruv[2] + 0.5f * cube_size * sqrtf(3.0f) < 0.0f)
        return;
      float image_center[2] = {0.0f, 0.0f};
      float image_size[2] = {2.0f * frustum.image_r, 2.0f * frustum.image_t};
      intersect_result =
          intersectBoxes2D(bound_center, bound_size, image_center, image_size);
    }
#if 0
    std::cout << "cube center:       (" << 
      cube_center[0] << ", " << cube_center[1] << ", " << cube_center[2] << "), " << cube_size << std::endl;
    std::cout << "cube center (ruv): (" << ruv[0] << ", " << ruv[1] << ", " << ruv[2] << "), " << std::endl;
    std::cout << "cube bound:  [" << 
      bound_center[0] - 0.5f * bound_size[0] << ", " << bound_center[0] + 0.5f * bound_size[0] << "]x[" << 
      bound_center[1] - 0.5f * bound_size[1] << ", " << bound_center[1] + 0.5f * bound_size[1] << "]" << std::endl;
    std::cout << "bound size: " << (std::max)(bound_size[0], bound_size[1]) << std::endl;
    std::cout << "image bound: [" << 
      -frustum.image_r << ", " << frustum.image_r << "]x[" << 
      -frustum.image_t << ", " << frustum.image_t << "]" << std::endl;
    std::cout << "# points: " << node->point_count << std::endl;
    std::cout << "is leaf? " << node->is_leaf << std::endl;
    std::cout << "centroid id: " << node->centroid_index << std::endl;
    std::cout << "intersection: " << intersect_result << std::endl;
    int x;
    std::cin >> x;
#endif
    if (intersect_result == OUTSIDE) return;
    float adjusted_cube_size;
    if (projection_mode == PERSPECTIVE)
      adjusted_cube_size = fudge_factor * cube_size / fabs(ruv[2]);
    else
      adjusted_cube_size = fudge_factor * cube_size;
    if (adjusted_cube_size < eps) {
      indices.push_back(node->centroid_index);
    } else if (node->is_leaf) {
      for (unsigned int i = 0; i < node->point_count; i++)
        indices.push_back(node->point_index + i);
    } else {  // inner node
      unsigned int b[3][2];
      if (projection_mode == PERSPECTIVE) {
        for (int i = 0; i < 3; i++) {
          if (frustum.eye[i] < cube_center[i]) {
            // visit right side first
            b[i][0] = 1;
            b[i][1] = 0;
          } else {
            // visit left side first
            b[i][0] = 0;
            b[i][1] = 1;
          }
        }
      } else {  // ORTHOGRAPHIC
        for (int i = 0; i < 3; i++) {
          // note: view is opposite viewing direction
          if (frustum.view[i] < 0.0f) {
            // visit right side first
            b[i][0] = 1;
            b[i][1] = 0;
          } else {
            // visit left side first
            b[i][0] = 0;
            b[i][1] = 1;
          }
        }
      }
      // visit child nodes in back to front order
      for (int k = 0; k < 2; k++) {
        for (int j = 0; j < 2; j++) {
          for (int i = 0; i < 2; i++) {
            unsigned int node_index = (b[0][i] << 2) + (b[1][j] << 1) + b[2][k];
            float child_corner[3] = {
                b[0][i] == 0 ? cube_corner[0] : cube_center[0],
                b[1][j] == 0 ? cube_corner[1] : cube_center[1],
                b[2][k] == 0 ? cube_corner[2] : cube_center[2]};
            if (!node->children[node_index]) continue;
            getIndicesHelper(indices, node->children[node_index], frustum,
                             child_corner, 0.5f * cube_size, eps,
                             projection_mode, fudge_factor, intersect_result);
          }
        }
      }
    }
  }

  unsigned int countNodesHelper(const Node* node, bool count_leaf = true) {
    if (!node)
      return 0;
    else if (node->is_leaf)
      return count_leaf ? 1 : 0;
    else {
      unsigned int count = 0;
      for (int i = 0; i < 8; i++) count += countNodesHelper(node->children[i]);
      return count + 1;
    }
  }

  unsigned int _max_leaf_size;
  unsigned int _num_points;
  Node* _root;
  std::vector<float>* _ptr_point_xyz;
  std::vector<float>* _ptr_point_size;

  float _lower_left_corner[3];
  float _cube_size;

  // scratch space for buildTreeHelper
  std::vector<unsigned int> _indices;
  std::vector<unsigned int> _indices_r;
  std::vector<unsigned char> _labels;
};

#endif  // __OCTREE_H__
