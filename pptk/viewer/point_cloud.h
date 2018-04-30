#ifndef __POINTCLOUD_H__
#define __POINTCLOUD_H__
#include <QOpenGLContext>
#include <QOpenGLShaderProgram>
#include <QWindow>
#include <vector>
#include "box3.h"
#include "octree.h"
#include "opengl_funcs.h"
#include "point_attributes.h"
#include "qt_camera.h"
#include "selection_box.h"
#include "timer.h"

class PointCloud : protected OpenGLFuncs {
 public:
  PointCloud(QWindow* window, QOpenGLContext* context)
      : _context(context),
        _window(window),
        _point_size(0.0f),
        _num_points(0),
        _buffer_positions(0),
        _buffer_colors(0),
        _buffer_sizes(0),
        _buffer_selection_mask(0),
        _buffer_octree_ids(0),
        _color_map(4, 1.0f),
        _color_map_min(0.0f),
        _color_map_max(1.0f),
        _color_map_auto(true) {
    _context->makeCurrent(_window);
    initializeOpenGLFunctions();
    _context->doneCurrent();
    compileProgram();
  }

  ~PointCloud() { clearPoints(); }

  void loadPoints(std::vector<float>& positions) {
    // warning: this function modifies positions and colors
    _positions.swap(positions);
    _num_points = _positions.size() / 3;

    _octree.buildTree(_positions, _sizes, 32);
    _octree_ids.reserve(_num_points);

    _full_box = vltools::Box3<float>();
    _full_box.addPoints(&_positions[0], _num_points);

    _context->makeCurrent(_window);

    // create a buffer for storing position vectors
    glGenBuffers(1, &_buffer_positions);
    glBindBuffer(GL_ARRAY_BUFFER, _buffer_positions);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * _positions.size(),
                 (GLvoid*)&_positions[0], GL_STATIC_DRAW);

    // create a buffer for storing color vectors
    glGenBuffers(1, &_buffer_colors);

    // create a buffer for storing per point scalars
    glGenBuffers(1, &_buffer_scalars);

    // create buffer for storing centroid sizes
    glGenBuffers(1, &_buffer_sizes);
    glBindBuffer(GL_ARRAY_BUFFER, _buffer_sizes);
    glBufferData(GL_ARRAY_BUFFER, _sizes.size() * sizeof(float),
                 (GLvoid*)&_sizes[0], GL_STATIC_DRAW);

    // create buffer for storing selection mask
    glGenBuffers(1, &_buffer_selection_mask);
    glBindBuffer(GL_ARRAY_BUFFER, _buffer_selection_mask);
    glBufferData(GL_ARRAY_BUFFER, _positions.size() / 3 * sizeof(float), NULL,
                 GL_DYNAMIC_DRAW);

    // create buffer for storing point indices obtained from octree
    glGenBuffers(1, &_buffer_octree_ids);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _buffer_octree_ids);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, _num_points * sizeof(unsigned int),
                 NULL, GL_DYNAMIC_DRAW);

    _context->doneCurrent();
    _attributes.reset();
    initColors();
  }

  void clearPoints() {
    clearAttributes();
    if (_num_points == 0) return;
    _num_points = 0;
    _positions.clear();
    _sizes.clear();
    _octree_ids.clear();
    _selected_ids.clear();
    _full_box = vltools::Box3<float>();
    _context->makeCurrent(_window);
    glDeleteBuffers(1, &_buffer_positions);
    glDeleteBuffers(1, &_buffer_colors);
    glDeleteBuffers(1, &_buffer_scalars);
    glDeleteBuffers(1, &_buffer_sizes);
    glDeleteBuffers(1, &_buffer_selection_mask);
    glDeleteBuffers(1, &_buffer_octree_ids);
    _context->doneCurrent();
    _octree.buildTree(_positions, _sizes, 32);
    _attributes.reset();
  }

  void loadAttributes(const std::vector<char>& data) {
    _attributes.set(data, _octree);
    initColors();
  }

  void loadAttributes(const std::vector<float>& attr, quint64 attr_size,
                      quint64 attr_dim) {
    _attributes.set(attr, attr_size, attr_dim);
    initColors();
  }

  void clearAttributes() { _attributes = PointAttributes(); }

  // render methods
  void draw(const QtCamera& camera, const SelectionBox* box = NULL) {
    queryLOD(_octree_ids, camera, 0.25f);
    if (_octree_ids.empty()) return;
    draw(&_octree_ids[0], (unsigned int)_octree_ids.size(), camera, box);
  }

  void draw(const unsigned int* indices, const unsigned int num_points,
            const QtCamera& camera, const SelectionBox* box = NULL) {
    if (_num_points == 0) return;

    // box should be in normalized device coordinates
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glEnable(GL_BLEND);

    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ZERO, GL_ONE);
    glBlendEquation(GL_FUNC_ADD);

    _program.bind();
    _program.setUniformValue(
        "width", (float)_window->devicePixelRatio() * _window->width());
    _program.setUniformValue(
        "height", (float)_window->devicePixelRatio() * _window->height());
    _program.setUniformValue("point_size", _point_size);
    _program.setUniformValue("mvpMatrix", camera.computeMVPMatrix(_full_box));
    _program.setUniformValue(
        "box_min", box ? box->getBox().topLeft()
                       : QPointF());  // topLeft in Qt is bottom left in NDC
    _program.setUniformValue("box_max", box ? box->getBox().bottomRight()
                                            : QPointF());  // top right in NDC
    _program.setUniformValue("eye", camera.getCameraPosition());
    _program.setUniformValue("view", camera.getViewVector());
    _program.setUniformValue("image_t", camera.getTop());
    _program.setUniformValue("box_select_mode",
                             box ? box->getType() : SelectionBox::NONE);
    _program.setUniformValue("projection_mode", camera.getProjectionMode());
    _program.setUniformValue("color_map", 0);
    _program.setUniformValue("scalar_min", _color_map_min);
    _program.setUniformValue("scalar_max", _color_map_max);
    _program.setUniformValue("color_map_n", _color_map.size() / 4.0f);

    _program.enableAttributeArray("position");
    _program.enableAttributeArray("size");
    _program.enableAttributeArray("selected");

    glBindBuffer(GL_ARRAY_BUFFER, _buffer_positions);
    _program.setAttributeArray("position", GL_FLOAT, 0, 3);
    glBindBuffer(GL_ARRAY_BUFFER, _buffer_sizes);
    _program.setAttributeArray("size", GL_FLOAT, 0, 1);
    glBindBuffer(GL_ARRAY_BUFFER, _buffer_selection_mask);
    _program.setAttributeArray("selected", GL_FLOAT, 0, 1);

    int curr_attr_idx = (int)_attributes.currentIndex();
    bool use_color_map = _attributes.dim(curr_attr_idx) == 1;
    bool broadcast_attr = _attributes.size(curr_attr_idx) == 1;
    if (use_color_map) {
      _program.setAttributeValue("color", QVector4D(1.0f, 1.0f, 1.0f, 1.0f));
    } else if (broadcast_attr) {
      const std::vector<float>& v = _attributes[curr_attr_idx];
      _program.setAttributeValue("color", QVector4D(v[0], v[1], v[2], v[3]));
    } else {
      glBindBuffer(GL_ARRAY_BUFFER, _buffer_colors);
      _program.enableAttributeArray("color");
      _program.setAttributeArray("color", GL_FLOAT, 0, 4);
    }
    if (!use_color_map) {
      _program.setAttributeValue("scalar", 1.0f);
    } else if (broadcast_attr) {
      _program.setAttributeValue("scalar", _attributes[curr_attr_idx][0]);
    } else {
      glBindBuffer(GL_ARRAY_BUFFER, _buffer_scalars);
      _program.enableAttributeArray("scalar");
      _program.setAttributeArray("scalar", GL_FLOAT, 0, 1);
    }

    glActiveTexture(GL_TEXTURE0 + 0);
    glBindTexture(GL_TEXTURE_1D, _texture_color_map);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _buffer_octree_ids);
    glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0,
                    sizeof(unsigned int) * num_points, (GLvoid*)indices);
    glDrawElements(GL_POINTS, num_points, GL_UNSIGNED_INT, 0);

    if (!use_color_map && !broadcast_attr) {
      _program.disableAttributeArray("color");
    }
    if (use_color_map && !broadcast_attr) {
      _program.disableAttributeArray("scalar");
    }

    _program.disableAttributeArray("position");
    _program.disableAttributeArray("size");
    _program.disableAttributeArray("selected");

    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
    glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
  }

  void queryLOD(std::vector<unsigned int>& indices, const QtCamera& camera,
                const float fudge_factor = 1.0f) {
    float min_z_near = 0.1f;
    if (camera.getProjectionMode() == QtCamera::PERSPECTIVE)
      _octree.getIndices(indices, camera, camera.getVerticalFOV(), -min_z_near,
                         _window->width(), _window->height(), fudge_factor);
    else {
      float t =
          camera.getCameraDistance() * tan(0.5f * camera.getVerticalFOV());
      float r = (float)_window->width() / _window->height() * t;
      _octree.getIndicesOrtho(indices, camera, r, t, _window->height(),
                              fudge_factor);
    }
  }

  void initColors() {
    // prepare OpenGL buffers and textures for current attribute set
    // four cases:            use colormap   upload array to gpu
    //    1. scalar           Y              N
    //    2. rgba             N              N
    //    3. array of scalar  Y              Y
    //    4. array of rgba    N              Y
    _context->makeCurrent(_window);
    int curr_attr_idx = (int)_attributes.currentIndex();
    bool use_color_map = _attributes.dim(curr_attr_idx) == 1;
    bool broadcast_attr = _attributes.size(curr_attr_idx) == 1;
    const std::vector<float>& attr = _attributes[curr_attr_idx];
    glEnable(GL_TEXTURE_1D);
    glActiveTexture(GL_TEXTURE0 + 0);
    glDeleteTextures(1, &_texture_color_map);
    glGenTextures(1, &_texture_color_map);
    glBindTexture(GL_TEXTURE_1D, _texture_color_map);
    if (use_color_map) {
      // use client provided color map
      glTexImage1D(GL_TEXTURE_1D, 0, 4, (int)_color_map.size() / 4, 0, GL_RGBA,
                   GL_FLOAT, (GLvoid*)&_color_map[0]);
      // not sure why this is needed, but it is
      glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAX_LEVEL, 0);
    } else {
      // use color map that always returns white
      float white[4] = {1.0f, 1.0f, 1.0f, 1.0f};
      glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, 1, 0, GL_RGBA, GL_FLOAT,
                   (GLvoid*)white);
    }
    // load color/scalar buffer if size > 1
    GLuint attr_buffer = use_color_map ? _buffer_scalars : _buffer_colors;
    if (!broadcast_attr) {
      glBindBuffer(GL_ARRAY_BUFFER, attr_buffer);
      glBufferData(GL_ARRAY_BUFFER, sizeof(float) * attr.size(),
                   (GLvoid*)&attr[0], GL_STATIC_DRAW);
    }
    _context->doneCurrent();
    if (_color_map_auto) {
      setColorMapScale(1.0f, 0.0f);
    }
  }

  void setColorMap(const std::vector<float>& color_map) {
    _color_map.resize(color_map.size());
    std::copy(color_map.begin(), color_map.end(), _color_map.begin());
    initColors();
  }

  void setColorMapScale(float scale_min, float scale_max) {
    // scale_min >= scale_max understood to mean auto
    if (scale_min >= scale_max) {
      _color_map_auto = true;
      // automatically set [_color_map_min, _color_map_max]
      // according to the attribute type
      int curr_attr_idx = (int)_attributes.currentIndex();
      bool use_color_map = _attributes.dim(curr_attr_idx) == 1;
      bool broadcast_attr = _attributes.size(curr_attr_idx) == 1;
      const std::vector<float>& attr = _attributes[curr_attr_idx];
      if (!use_color_map) {
        _color_map_min = 0.0f;
        _color_map_max = 1.0f;
      } else if (broadcast_attr) {
        _color_map_min = attr[0] - 1.0f;
        _color_map_max = attr[0] + 1.0f;
      } else {
        _color_map_min = std::numeric_limits<float>::max();
        _color_map_max = -std::numeric_limits<float>::max();
        for (std::size_t i = 0; i < attr.size(); i++) {
          if (attr[i] == attr[i]) {  // skip if attr[i] is NaN
            _color_map_min = qMin(_color_map_min, attr[i]);
            _color_map_max = qMax(_color_map_max, attr[i]);
          }
        }
      }
    } else {
      _color_map_auto = false;
      _color_map_min = scale_min;
      _color_map_max = scale_max;
    }
  }

  // selection methods
  void selectInBox(const SelectionBox& box, const QtCamera& camera) {
    if (box.getType() == SelectionBox::NONE) return;
    QMatrix4x4 mvp = camera.computeMVPMatrix(_full_box);
    QRectF rect = box.getBox();
    std::vector<unsigned int> new_indices;
    // check all centroids in addition to all points
    for (unsigned int i = 0; i < _positions.size() / 3; i++) {
      float* v = &_positions[3 * i];
      QVector4D p(v[0], v[1], v[2], 1);
      p = mvp * p;
      p /= p.w();
      if (rect.contains(QPointF(p.x(), p.y())) && p.z() > -1.0f && p.z() < 1.0f)
        new_indices.push_back(i);
    }
    if (box.getType() == SelectionBox::ADD)
      mergeIndices(_selected_ids, new_indices);
    else  // box.getType() == SelectionBox::SUB
      removeIndices(_selected_ids, new_indices);
    updateSelectionMask();
  }

  void queryNearPoint(std::vector<unsigned int>& indices, const QPointF& point,
                      const QtCamera& camera) {
    Octree::ProjectionMode projection_mode =
        camera.getProjectionMode() == QtCamera::PERSPECTIVE
            ? Octree::PERSPECTIVE
            : Octree::ORTHOGRAPHIC;
    _octree.getClickIndices(
        indices, point.x(), _window->height() - point.y() - 1.0f, 5.0f,
        _window->width(), _window->height(), camera.getVerticalFOV(), 0.1f,
        camera, projection_mode);
  }

  void selectNearPoint(const QPointF& point, const QtCamera& camera,
                       bool deselect = false) {
    std::vector<unsigned int> new_indices;
    queryNearPoint(new_indices, point, camera);
    if (new_indices.empty()) return;
    if (deselect)
      removeIndices(_selected_ids, new_indices);
    else
      mergeIndices(_selected_ids, new_indices);
    updateSelectionMask();
  }

  void deselectNearPoint(const QPointF& point, const QtCamera& camera) {
    selectNearPoint(point, camera, true);
  }

  void getSelected(std::vector<unsigned int>& indices) const {
    // returns indices into original array of points
    // (prior to reshuffling by octree)
    indices.reserve(_selected_ids.size());
    indices.clear();
    for (std::size_t i = 0; i < _selected_ids.size(); i++) {
      if (_selected_ids[i] < _num_points)
        indices.push_back(_octree.getIndices()[_selected_ids[i]]);
      else
        break;
    }
  }

  void setSelected(const std::vector<unsigned int>& indices) {
    // expects indices into original array of points
    _selected_ids.clear();
    for (std::size_t i = 0; i < indices.size(); i++) {
      _selected_ids.push_back(_octree.getIndicesR()[indices[i]]);
    }
    std::sort(_selected_ids.begin(), _selected_ids.end());  // increasing order
    updateSelectionMask();
  }

  void clearSelected() {
    _selected_ids.clear();
    updateSelectionMask();
  }

  QVector3D computeSelectionCentroid() {
    // returns bounding box centroid if no points are selected
    QVector3D centroid;
    std::size_t num_selected = 0;
    for (std::size_t i = 0; i < _selected_ids.size(); i++) {
      if (_selected_ids[i] >= _num_points) break;
      num_selected++;
      float* v = &_positions[3 * _selected_ids[i]];
      centroid += QVector3D(v[0], v[1], v[2]);
    }
    if (num_selected == 0)
      return QVector3D(0.5f * (_full_box.x(0) + _full_box.x(1)),
                       0.5f * (_full_box.y(0) + _full_box.y(1)),
                       0.5f * (_full_box.z(0) + _full_box.z(1)));
    else
      return centroid / num_selected;
  }

  // getters and setters
  std::size_t getNumPoints() const { return _num_points; }
  std::size_t getNumSelected() const {
    return countSelected(_selected_ids, (unsigned int)_num_points);
  }
  std::size_t getNumAttributes() const { return _attributes.numAttributes(); }
  std::size_t getCurrentAttributeIndex() const {
    return _attributes.currentIndex();
  }
  const PointAttributes& getAttributes() const { return _attributes; }
  const std::vector<float>& getPositions() const { return _positions; }
  const std::vector<unsigned int>& getSelectedIds() const {
    return _selected_ids;
  }
  const vltools::Box3<float>& getBox() const { return _full_box; }
  float getFloor() const { return _num_points == 0 ? 0.0f : _full_box.min(2); }
  void setPointSize(float point_size) { _point_size = point_size; }
  void setCurrentAttributeIndex(std::size_t i) {
    bool index_changed = i != _attributes.currentIndex();
    _attributes.setCurrentIndex(i);
    if (index_changed) initColors();
  }

 private:
  void compileProgram() {
    std::string vsCode =
        "#version 120\n"
        "\n"
        "uniform float point_size;\n"
        "uniform float width;\n"
        "uniform float height;\n"
        "uniform vec2 box_min;\n"
        "uniform vec2 box_max;\n"
        "uniform int draw_selection_box;\n"
        "uniform int box_select_mode;  // 0 - add, 1 - remove, 2 - no box\n"
        "uniform mat4 mvpMatrix;\n"
        "uniform sampler1D color_map;\n"
        "uniform float scalar_min;\n"
        "uniform float scalar_max;\n"
        "uniform float color_map_n;\n"

        "uniform int projection_mode;\n"
        "uniform vec3 eye;\n"
        "uniform vec3 view;\n"
        "uniform float image_t;\n"
        "varying float inner_radius;\n"
        "varying float outer_radius;\n"

        "attribute vec3 position;\n"
        "attribute vec4 color;\n"
        "attribute float scalar;\n"
        "attribute float size;\n"
        "attribute float selected;\n"
        "varying vec4 frag_color;\n"
        "varying vec2 frag_center;\n"
        "\n"
        "void main() {\n"
        "  vec4 p = mvpMatrix * vec4(position, 1.0);\n"
        "  frag_center = 0.5 * (p.xy / p.w + 1.0) * vec2(width, height);\n"
        "  gl_Position = p;\n"
        "  p /= p.w;\n"
        "  float tex_coord = clamp((scalar - scalar_min) / (scalar_max - scalar_min), 0.0, 1.0);\n"
        "  tex_coord = (tex_coord - 0.5) * (color_map_n - 1.0) / color_map_n + 0.5;\n"
        "  vec4 color_s = tex_coord != tex_coord ? vec4(0, 0, 0, 1) : texture1D(color_map, tex_coord);\n"
        "  vec4 color_r = color_s * color;\n"
        "  if (box_select_mode == 2)\n"
        "    frag_color = selected == 1.0 ? vec4(1, 1, 0, 1) : color_r;\n"
        "  else {\n"
        "    bool inBox = p.x < box_max.x && p.x > box_min.x && p.y < box_max.y && p.y > box_min.y && p.z < 1.0 && p.z > -1.0;\n"
        "    if (box_select_mode == 0)\n"
        "      frag_color = (inBox || selected == 1.0) ? vec4(1, 1, 0, 1) : color_r;\n"
        "    else\n"
        "      frag_color = (!inBox && selected == 1.0) ? vec4(1, 1, 0, 1) : color_r;\n"
        "  }\n"
        "  float d = abs(dot(position.xyz - eye,view));\n"
        "  if (projection_mode == 1) d = 1.0;\n"
        "  if (size == 0.0) {\n"
        "    inner_radius = point_size / d * height / (2.0 * image_t);\n"
        "    outer_radius = inner_radius + 1.0;\n"
        "  } else {\n"
        "    inner_radius = 0.5 * size / d * height / (2.0 * image_t);\n"
        "    outer_radius = max(1.0, 2.0 * inner_radius);\n"
        "  }\n"
        "  gl_PointSize = outer_radius * 2.0;\n"
        "}\n";
    std::string fsCode =
        "#version 120\n"
        "\n"
        "uniform float point_size;\n"
        "varying vec4 frag_color;\n"
        "varying vec2 frag_center;\n"
        "varying float inner_radius;\n"
        "varying float outer_radius;\n"
        "\n"
        "void main() {\n"
        "  float weight = clamp((outer_radius - length(frag_center - gl_FragCoord.xy)) / (outer_radius - inner_radius), 0, 1);\n"
        "  gl_FragColor = frag_color * vec4(1, 1, 1, weight);\n"
        "}\n";
    _context->makeCurrent(_window);
    _program.addShaderFromSourceCode(QOpenGLShader::Vertex, vsCode.c_str());
    _program.addShaderFromSourceCode(QOpenGLShader::Fragment, fsCode.c_str());
    _program.link();
    _context->doneCurrent();
  }

  static void mergeIndices(std::vector<unsigned int>& x,
                           const std::vector<unsigned int>& y,
                           bool xor_merge = false) {
    // assumes x and y are sorted in increasing order
    std::vector<unsigned int> temp;
    temp.reserve(x.size() + y.size());
    std::size_t y_idx = 0;
    for (std::size_t i = 0; i < x.size(); i++) {
      while (y_idx < y.size() && y[y_idx] < x[i]) temp.push_back(y[y_idx++]);
      if (y_idx == y.size()) {
        temp.insert(temp.end(), &x[i], &x[i] + (x.size() - i));
        break;
      }
      if (y[y_idx] == x[i]) {
        y_idx++;
        if (xor_merge) continue;
      }
      temp.push_back(x[i]);
    }
    if (y_idx < y.size())
      temp.insert(temp.end(), &y[y_idx], &y[y_idx] + (y.size() - y_idx));
    x.swap(temp);
  }

  static void removeIndices(std::vector<unsigned int>& x,
                            const std::vector<unsigned int>& y) {
    std::vector<unsigned int> temp;
    temp.reserve(x.size());
    std::size_t y_idx = 0;
    for (std::size_t i = 0; i < x.size(); i++) {
      while (y_idx < y.size() && y[y_idx] < x[i]) y_idx++;
      if (y_idx == y.size()) {
        temp.insert(temp.end(), &x[i], &x[i] + (x.size() - i));
        break;
      }
      if (y[y_idx] == x[i])
        y_idx++;
      else
        temp.push_back(x[i]);
    }
    x.swap(temp);
  }

  void updateSelectionMask() {
    std::vector<float> selection_mask(_positions.size() / 3, 0.0f);
    for (std::size_t i = 0; i < _selected_ids.size(); i++)
      selection_mask[_selected_ids[i]] = 1.0f;
    _context->makeCurrent(_window);
    glBindBuffer(GL_ARRAY_BUFFER, _buffer_selection_mask);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float) * _positions.size() / 3,
                    (GLvoid*)&selection_mask[0]);
    _context->doneCurrent();
  }

  static std::size_t countSelected(const std::vector<unsigned int>& x,
                                   unsigned int y) {
    // note _selected_ids may contain centroid, we desire non-centroid ids
    // find number of items in _selected_ids (sorted)
    // that is less than _num_points
    if (x.empty()) return 0;
    std::size_t a = 0;
    std::size_t b = x.size() - 1;
    if (x[b] < y) return b + 1;
    if (x[a] >= y) return 0;
    // at this point, we know x.size() >= 2
    // invariances:
    // 1. x[b] >= y
    // 2. x[a] < y
    while (b > a + 1) {
      std::size_t c = (a + b) / 2;
      if (x[c] < y)
        a = c;
      else
        b = c;
    }
    return a + 1;
  }

  QOpenGLContext* _context;
  QWindow* _window;
  QOpenGLShaderProgram _program;

  float _point_size;
  std::size_t _num_points;
  std::vector<float> _positions;
  std::vector<float> _colors;
  std::vector<float> _sizes;
  std::vector<unsigned int> _octree_ids;    // LOD-selected points dumped here
  std::vector<unsigned int> _selected_ids;  // maintain list of selected points
  vltools::Box3<float> _full_box;
  GLuint _buffer_positions;
  GLuint _buffer_colors;
  GLuint _buffer_scalars;
  GLuint _buffer_sizes;
  GLuint _buffer_selection_mask;
  GLuint _buffer_octree_ids;
  GLuint _texture_color_map;
  Octree _octree;
  PointAttributes _attributes;

  std::vector<float> _color_map;
  float _color_map_min;
  float _color_map_max;
  bool _color_map_auto;
};

#endif  // __POINTCLOUD_H__
