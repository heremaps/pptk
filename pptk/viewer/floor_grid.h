#ifndef __FLOORGRID_H__
#define __FLOORGRID_H__
#include <math.h>
#include <QOpenGLContext>
#include <QOpenGLShaderProgram>
#include <QWindow>
#include <exception>
#include <iostream>
#include "opengl_funcs.h"
#include "qt_camera.h"

class SectorCreateException : public std::exception {
  const char* what() const throw() { return "Failed to create sector.\n"; }
};

/*! \brief
 *  Acute (angle < pi) sector class (in radians)
 *  Restricting to acute angles to ensure sector intersection results in at
 *    most one sector
 */
class Sector {
 public:
  Sector() : _start(0.0f), _end(-1.0f) {}
  Sector(float start, float end) : _start(start), _end(end) {
    normalize(_start, _end);
  }

  operator bool() { return !empty(); }

  bool empty() const { return _start == 0.0f && _end == -1.0f; }

  /* \brief
   * returns true if angle is in sector (including _start and _end)
   */
  bool contains(float angle) const {
    angle = normalize(angle);
    if (angle >= _start)
      return angle <= _end;
    else
      return angle + 2.0f * pi() <= _end;
  }

  /* \brief
   * returns the sector intersection between this and other
   */
  Sector intersect(const Sector& other) const {
    const Sector* a = this;
    const Sector* b = &other;
    if (a->_start > b->_start) std::swap(a, b);

    if (b->_start <= a->_end) {
      return Sector(b->_start, std::min(a->_end, b->_end));
    } else if (b->_end >= a->_start + 2.0f * pi()) {
      return Sector(a->_start + 2.0f * pi(),
                    std::min(a->_end + 2.0f * pi(), b->_end));
    } else
      return Sector();
  }

  float getStart() const { return _start; }
  float getEnd() const { return _end; }
  static float pi() { return atan2(0.0f, -1.0f); }
  static float rad2deg(float rad) { return rad / pi() * 180.0f; }
  static float deg2rad(float deg) { return deg / 180.0f * pi(); }
  friend std::ostream& operator<<(std::ostream&, const Sector&);

 private:
  /* \brief
   * _start is in [0, 2*pi) and _end is in [_start, _start+2*pi]
   */
  static void normalize(float& start, float& end) {
    bool full_circle = end - start >= 2.0f * pi();
    if (full_circle) {
      start = 0;
      end = 2.0f * pi();
    } else {
      start = normalize(start);
      end = normalize(end);
      if (end < start) end += 2.0f * pi();
    }
  }

  /* \brief
   * normalizes angle x to be in [0, 2*pi)
   */
  static float normalize(float x) {
    return x - floor(x / (2.0f * pi())) * 2.0f * pi();
  }

  float _start;
  float _end;
};

class FloorGrid : protected OpenGLFuncs {
 public:
  FloorGrid(QWindow* window, QOpenGLContext* context)
      : _context(context),
        _window(window),
        _visible(true),
        _grid_line_color(0.7f, 0.7f, 0.7f, 1.0f),
        _grid_floor_color(0.3f, 0.3f, 0.3f, 0.5f),
        _grid_floor_z(0.0f),
        _cell_size(1.0f),
        _line_weight(0.0f) {
    _context->makeCurrent(_window);
    initializeOpenGLFunctions();
    _context->doneCurrent();

    compilePerspProgram();
    compileOrthoProgram();
    loadSquare();
  }

  ~FloorGrid() { unloadSquare(); }

  void draw(const QtCamera& camera) { draw(camera, _grid_floor_z); }

  void draw(const QtCamera& camera, float z_floor) {
    if (!_visible) return;
    if (camera.getProjectionMode() == QtCamera::PERSPECTIVE)
      drawPersp(camera, z_floor);
    else
      drawOrtho(camera, z_floor);
  }

  float getCellSize() const { return _cell_size; }
  float getLineWeight() const { return _line_weight; }
  QVector4D getLineColor() const { return _grid_line_color; }
  QVector4D getFloorColor() const { return _grid_floor_color; }
  float getFloorLevel() const { return _grid_floor_z; }
  bool getVisible() const { return _visible; }
  void setLineColor(QVector4D line_color) { _grid_line_color = line_color; }
  void setFloorColor(QVector4D floor_color) { _grid_floor_color = floor_color; }
  void setFloorLevel(float z) { _grid_floor_z = z; }
  void setVisible(bool visible) { _visible = visible; }

 private:
  void compilePerspProgram() {
    std::string vsCode =
        "#version 120\n"
        "\n"
        "// camera coordinate frame\n"
        "uniform vec3 eye;\n"
        "uniform vec3 right;\n"
        "uniform vec3 up;\n"
        "uniform vec3 view;\n"
        "uniform float height;  // relative to floor\n"
        "\n"
        "// image dimensions\n"
        "uniform float h_lo;\n"
        "uniform float h_hi;\n"
        "uniform float r;  // right\n"
        "uniform float t;  // top\n"
        "\n"
        "attribute vec3 position;\n"
        "varying vec2 floor_coord;\n"
        "varying float distance;\n"
        "\n"
        "void main() {\n"
        "  vec2 image_coord = position.xy * vec2(2.0 * r, h_hi - h_lo)+vec2(-r, h_lo);\n"
        "  mat3 R = mat3(right,up,view);\n"
        "  vec3 p_world = R * vec3(image_coord,-1);\n"
        "  p_world *= -height/p_world.z;\n"
        "  vec3 p_camera = transpose(R) * p_world;\n"
        "  gl_Position = vec4(p_camera.xy * vec2(1.0 / r, 1.0 / t), 0, -p_camera.z);\n"
        "  floor_coord = p_world.xy;\n"
        "  distance = length(p_world);\n"
        "}\n";
    std::string fsCode =
        "#version 120\n"
        "\n"
        "uniform vec3 eye;\n"
        "uniform vec3 right;\n"
        "uniform vec3 up;\n"
        "uniform vec3 view;\n"
        "uniform float height;\n"
        "\n"
        "uniform vec4 line_color;\n"
        "uniform vec4 floor_color;\n"
        "uniform float cell_size;\n"
        "uniform float line_weight;\n"
        "uniform float line_width;\n"
        "\n"
        "uniform float max_dist_in_focus;\n"
        "uniform float max_dist_visible;\n"
        "\n"
        "varying vec2 floor_coord;\n"
        "varying float distance;\n"
        "\n"
        "float compute_weight(vec3 n, vec2 image_coord) {\n"
        "  vec3 line = transpose(mat3(right, up, view))*n;\n"
        "  line /= length(line.xy);\n"
        "  float eps = -abs(dot(vec3(image_coord, -1), line));\n"
        "  return (eps+line_width) / line_width;\n"
        "}\n"
        "void main() {\n"
        "  vec2 cell_idx = floor((floor_coord + eye.xy) / cell_size);\n"
        "  vec2 cell_min = cell_idx * cell_size - eye.xy;\n"
        "  vec2 cell_max = cell_min + cell_size;\n"
        "  float i = mod(cell_idx.x, 10.0);\n"
        "  float j = mod(cell_idx.y, 10.0);\n"
        "  float x_min_weight = i == 0.0 ? 1.0 : line_weight;\n"
        "  float x_max_weight = i == 9.0 ? 1.0 : line_weight;\n"
        "  float y_min_weight = j == 0.0 ? 1.0 : line_weight;\n"
        "  float y_max_weight = j == 9.0 ? 1.0 : line_weight;\n"

        "  vec3 temp = transpose(mat3(right, up, view)) * vec3(floor_coord, -height);\n"
        "  vec2 image_coord = -temp.xy / temp.z;\n"
        "  float weight = 0.0;\n"
        "  weight = max(weight, x_min_weight * compute_weight(vec3(height, 0, cell_min.x), image_coord));\n"
        "  weight = max(weight, x_max_weight * compute_weight(vec3(height, 0, cell_max.x), image_coord));\n"
        "  weight = max(weight, y_min_weight * compute_weight(vec3(0, height, cell_min.y), image_coord));\n"
        "  weight = max(weight, y_max_weight * compute_weight(vec3(0, height, cell_max.y), image_coord));\n"
        "  weight *= 0.7;\n"

        "  float blur_weight = clamp((max_dist_visible - distance) / (max_dist_visible - max_dist_in_focus), 0.0, 1.0);\n"
        "  vec4 c = line_color * weight+floor_color*(1.0 - weight);\n"
        "  gl_FragColor = vec4(c.xyz, c.w * blur_weight);\n"
        "}\n";
    _context->makeCurrent(_window);
    _persp_program.addShaderFromSourceCode(QOpenGLShader::Vertex,
                                           vsCode.c_str());
    _persp_program.addShaderFromSourceCode(QOpenGLShader::Fragment,
                                           fsCode.c_str());
    _persp_program.link();
    _context->doneCurrent();
  }
  void compileOrthoProgram() {
    std::string vsCode =
        "#version 110\n"
        "uniform vec3 eye;\n"
        "uniform vec3 right;\n"
        "uniform vec3 up;\n"
        "uniform vec3 view;\n"
        "uniform float z_floor;\n"
        "uniform float r;\n"
        "uniform float t;\n"
        "attribute vec3 position;\n"
        "varying vec2 floor_coord;\n"
        "void main() {\n"
        "  vec2 image_coord = (position.xy - 0.5) * 2.0;\n"
        "  vec3 o = eye + image_coord.x * r * right + image_coord.y * t * up;\n"
        "  float d = (o.z - z_floor) / view.z;\n"
        "  floor_coord = -view.xy * d + o.xy;\n"
        "  gl_Position = vec4(image_coord, 0, 1);\n"
        "}\n";
    std::string fsCode =
        "#version 110\n"
        "uniform float eps_x;\n"
        "uniform float eps_y;\n"
        "uniform float cell_size;    // for minor grid cells\n"
        "uniform float line_weight;  // for minor grid lines\n"
        "uniform vec4 floor_color;\n"
        "uniform vec4 line_color;\n"
        "varying vec2 floor_coord;\n"
        "void main() {\n"
        "  vec2 cell_idx = floor(floor_coord / cell_size);\n"
        "  vec2 cell_min = cell_idx * cell_size;\n"
        "  vec2 cell_max = cell_min + cell_size;\n"
        "  vec2 ij = mod(cell_idx, 10.0);\n"
        "  float x_min_weight = ij.x == 0.0 ? 1.0 : line_weight;\n"
        "  float x_max_weight = ij.x == 9.0 ? 1.0 : line_weight;\n"
        "  float y_min_weight = ij.y == 0.0 ? 1.0 : line_weight;\n"
        "  float y_max_weight = ij.y == 9.0 ? 1.0 : line_weight;\n"

        "  float weight = 0.0;\n"
        "  weight = max(weight, x_min_weight * (1.0 - (floor_coord.x - cell_min.x) / eps_x));\n"
        "  weight = max(weight, y_min_weight * (1.0 - (floor_coord.y - cell_min.y) / eps_y));\n"
        "  weight = max(weight, x_max_weight * (1.0 + (floor_coord.x - cell_max.x) / eps_x));\n"
        "  weight = max(weight, y_max_weight * (1.0 + (floor_coord.y - cell_max.y) / eps_y));\n"
        "  weight *= 0.7;\n"

        "  gl_FragColor = floor_color * (1.0 - weight) + line_color * (weight);\n"
        "}\n";
    _context->makeCurrent(_window);
    _ortho_program.addShaderFromSourceCode(QOpenGLShader::Vertex,
                                           vsCode.c_str());
    _ortho_program.addShaderFromSourceCode(QOpenGLShader::Fragment,
                                           fsCode.c_str());
    _ortho_program.link();
    _context->doneCurrent();
  }
  void loadSquare() {
    _context->makeCurrent(_window);
    float points[12] = {
      0.0f, 0.0f, 0.0f,
      1.0f, 0.0f, 0.0f,
      1.0f, 1.0f, 0.0f,
      0.0f, 1.0f, 0.0f};
    glGenBuffers(1, &_buffer_square);
    glBindBuffer(GL_ARRAY_BUFFER, _buffer_square);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 12, (GLvoid*)points,
                 GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    unsigned int indices[6] = {
      0, 1, 2,
      0, 2, 3};
    glGenBuffers(1, &_buffer_square_indices);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _buffer_square_indices);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * 6,
                 (GLvoid*)indices, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    _context->doneCurrent();
  }
  void unloadSquare() {
    _context->makeCurrent(_window);
    glDeleteBuffers(1, &_buffer_square);
    glDeleteBuffers(1, &_buffer_square_indices);
    _context->doneCurrent();
  }
  float normalizeAngle(float angle) {
    // normalizes to interval [-pi,pi]
    angle = angle - floor(angle / (2.0f * PI)) * 2.0f * PI;
    if (angle > PI) angle -= 2.0f * PI;
    return angle;
  }
  float visibleDistance(float cell_size, float projected_cell_size,
                        const QtCamera& camera, float z_floor) {
    // cell size in world space and projected cell size in pixels
    float eye[3];
    camera.getCameraPosition(eye);
    float eye_floor_height = fabs(eye[2] - z_floor);
    return sqrt(cell_size * eye_floor_height * _window->height() /
                projected_cell_size / (2 * tan(45.0f / 2.0f / 180.0f * PI)));
  }
  bool computeHorizon(float& h_lo, float& h_hi, float cell_size,
                      const QtCamera& camera, float z_floor) {
    float projected_cell_size = 5.0f;
    float alpha = 45.0f / 180.0f * PI;
    float eye[3];
    camera.getCameraPosition(eye);
    float z_eye = eye[2];

    float eye_floor_height = z_eye - z_floor;
    float d_max =
        2.0f * visibleDistance(cell_size, projected_cell_size, camera, z_floor);
    if (fabs(eye_floor_height) > d_max) return false;
    float theta_max =
        asin(eye_floor_height /
             d_max);  // TODO: handle eye_floor_height > d_max case
    float theta = normalizeAngle(camera.getTheta());

    Sector floor_sector;
    if (eye_floor_height < 0.0f)
      floor_sector = Sector(-PI - theta_max, theta_max);
    else
      floor_sector = Sector(theta_max, PI - theta_max);
    Sector camera_sector(theta - 0.5f * alpha, theta + 0.5f * alpha);
    Sector horizon_sector = camera_sector.intersect(floor_sector);
    if (horizon_sector.empty()) return false;
    h_lo = tan(-horizon_sector.getEnd() + theta);
    h_hi = tan(-horizon_sector.getStart() + theta);
    return true;
  }
  void computeCellSize(float& cell_size, float& line_weight,
                       const QtCamera& camera, float z_floor) {
    float projected_cell_size = 150.0f;
    float alpha = 45.0f / 180.0f * PI;
    float eye[3];
    camera.getCameraPosition(eye);
    float lookat[3];
    camera.getLookAtPosition(lookat);
    float d = camera.getCameraDistance() + fabs(lookat[2] - z_floor);
    float d_0 =
        _window->height() / 2.0f / tan(alpha / 2.0f) / projected_cell_size;
    float x = log(d / d_0) / log(10.0f);
    float x_ = floor(x);
    line_weight = 1.0f - (x - x_);
    cell_size = pow(10.0f, x_);
  }
  void drawOrtho(const QtCamera& camera, float z_floor) {
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glEnable(GL_BLEND);

    float t = camera.getCameraDistance() * tan(0.5f * camera.getVerticalFOV());
    float r = camera.getAspectRatio() * t;
    z_floor = camera.getViewAxis() != QtCamera::ARBITRARY_AXIS ? 0.0f : z_floor;

    float delta_pixels = 1.0f;  // in pixels;
    float delta_image = delta_pixels / _window->height() /
                        _window->devicePixelRatio() * 2.0f * t;
    float eps_x, eps_y;
    {
      if (camera.getViewAxis() == QtCamera::ARBITRARY_AXIS) {
        float tan_theta = tan(camera.getTheta());
        float sin_phi = sin(camera.getPhi());
        float cos_phi = cos(camera.getPhi());
        float tan_theta_2 = tan_theta * tan_theta;
        float sin_phi_2 = sin_phi * sin_phi;
        float cos_phi_2 = cos_phi * cos_phi;
        eps_x = delta_image * sqrt(1.0f + cos_phi_2 / tan_theta_2);
        eps_y = delta_image * sqrt(1.0f + sin_phi_2 / tan_theta_2);
      } else {
        eps_x = eps_y = delta_image;
      }
    }

    computeCellSize(_cell_size, _line_weight, camera, z_floor);

    QVector4D line_color;
    {
      if (camera.getViewAxis() == QtCamera::ARBITRARY_AXIS) {
        float a = 0.1f;
        float b = 0.2f;
        float fade_weight = (std::min)(
            1.2f, (std::max)(0.0f, (float)(fabs(sin(camera.getTheta())) - a) /
                                       (b - a)));
        line_color = _grid_floor_color * (1.0f - fade_weight) +
                     _grid_line_color * fade_weight;
      } else {
        line_color = _grid_line_color;
      }
    }

    QVector3D eye, right, up, view;
    {
      if (camera.getViewAxis() == QtCamera::ARBITRARY_AXIS) {
        eye = camera.getCameraPosition();
        right = camera.getRightVector();
        up = camera.getUpVector();
        view = camera.getViewVector();
      } else {
        eye = camera.getCameraPosition();
        if (camera.getViewAxis() == QtCamera::X_AXIS) {
          eye = QVector3D(-eye.z(), eye.y(), 1.0f);
        } else if (camera.getViewAxis() == QtCamera::Y_AXIS) {
          eye = QVector3D(-eye.z(), eye.x(), 1.0f);
        } else {
          eye = QVector3D(-eye.y(), eye.x(), 1.0f);
        }
        right = QVector3D(0.0f, 1.0f, 0.0f);
        up = QVector3D(-1.0f, 0.0f, 0.0f);
        view = QVector3D(0.0f, 0.0f, 1.0f);
      }
    }

    _ortho_program.bind();
    _ortho_program.setUniformValue("eye", eye);
    _ortho_program.setUniformValue("right", right);
    _ortho_program.setUniformValue("up", up);
    _ortho_program.setUniformValue("view", view);
    _ortho_program.setUniformValue("t", t);
    _ortho_program.setUniformValue("r", r);
    _ortho_program.setUniformValue("z_floor", z_floor);
    _ortho_program.setUniformValue("eps_x", eps_x);
    _ortho_program.setUniformValue("eps_y", eps_y);
    _ortho_program.setUniformValue("cell_size", _cell_size);
    _ortho_program.setUniformValue("line_weight", _line_weight);
    _ortho_program.setUniformValue("line_color", line_color);
    _ortho_program.setUniformValue("floor_color", _grid_floor_color);

    glBindBuffer(GL_ARRAY_BUFFER, _buffer_square);
    _ortho_program.enableAttributeArray("position");
    _ortho_program.setAttributeArray("position", GL_FLOAT, 0, 3);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _buffer_square_indices);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, (GLvoid*)0);

    _ortho_program.disableAttributeArray("position");

    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
  }
  void drawPersp(const QtCamera& camera, float z_floor) {
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glEnable(GL_BLEND);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ZERO, GL_ONE);

    computeCellSize(_cell_size, _line_weight, camera, z_floor);
    float weighted_cell_size = _cell_size * pow(10, 1.0 - _line_weight);
    float max_dist_in_focus =
        visibleDistance(weighted_cell_size, 5.0f, camera, z_floor);
    float max_dist_visible = 2.0f * max_dist_in_focus;
    float h_lo, h_hi;
    computeHorizon(h_lo, h_hi, weighted_cell_size, camera, z_floor);
    float t = tan(45.0f / 2.0f / 180.0f * PI);
    float r = t * _window->width() / _window->height();
    float line_width =
        1.2f / _window->height() / _window->devicePixelRatio() * 2.0f * t;

    _persp_program.bind();
    _persp_program.setUniformValue("eye", camera.getCameraPosition());
    _persp_program.setUniformValue("right", camera.getRightVector());
    _persp_program.setUniformValue("up", camera.getUpVector());
    _persp_program.setUniformValue("view", camera.getViewVector());
    _persp_program.setUniformValue("height",
                                   camera.getCameraPosition().z() - z_floor);
    _persp_program.setUniformValue("cell_size", _cell_size);
    _persp_program.setUniformValue("line_weight", _line_weight);
    _persp_program.setUniformValue("line_color", _grid_line_color);
    _persp_program.setUniformValue("floor_color", _grid_floor_color);
    _persp_program.setUniformValue("max_dist_in_focus", max_dist_in_focus);
    _persp_program.setUniformValue("max_dist_visible", max_dist_visible);
    _persp_program.setUniformValue("h_lo", h_lo);
    _persp_program.setUniformValue("h_hi", h_hi);
    _persp_program.setUniformValue("t", t);
    _persp_program.setUniformValue("r", r);
    _persp_program.setUniformValue("line_width", line_width);

    _persp_program.enableAttributeArray("position");
    glBindBuffer(GL_ARRAY_BUFFER, _buffer_square);
    _persp_program.setAttributeArray("position", GL_FLOAT, 0, 3);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _buffer_square_indices);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, (GLvoid*)0);
    _persp_program.disableAttributeArray("position");

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_BLEND);
    glDepthMask(GL_TRUE);
    glEnable(GL_DEPTH_TEST);
  }

  QOpenGLShaderProgram _ortho_program;
  QOpenGLShaderProgram _persp_program;
  QOpenGLContext* _context;
  QWindow* _window;
  bool _visible;

  GLuint _buffer_square;
  GLuint _buffer_square_indices;

  // grid properties
  QVector4D _grid_line_color;
  QVector4D _grid_floor_color;
  float _grid_floor_z;

  float _cell_size;
  float _line_weight;
};

#endif  // __FLOORGRID_H__
