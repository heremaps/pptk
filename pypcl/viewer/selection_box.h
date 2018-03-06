#ifndef __SELECTIONBOX_H__
#define __SELECTIONBOX_H__
#include <QOpenGLContext>
#include <QOpenGLShaderProgram>
#include <QPointF>
#include <QRectF>
#include <QWindow>
#include "opengl_funcs.h"

class SelectionBox : protected OpenGLFuncs {
 public:
  enum SelectMode { ADD = 0, SUB = 1, NONE = 2 };

  SelectionBox(QWindow* window, QOpenGLContext* context)
      : _context(context), _window(window), _select_mode(NONE) {
    _context->makeCurrent(_window);
    initializeOpenGLFunctions();
    _context->doneCurrent();
    compileProgram();
  }

  void draw() {
    if (_select_mode == NONE) return;
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);

    GLuint buffer_square;
    glGenBuffers(1, &buffer_square);
    glBindBuffer(GL_ARRAY_BUFFER, buffer_square);
    float points[12] = {0.0f, 0.0f, 0.0f,
                        1.0f, 0.0f, 0.0f,
                        1.0f, 1.0f, 0.0f,
                        0.0f, 1.0f, 0.0f};
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 12, (GLvoid*)points,
                 GL_STATIC_DRAW);

    GLuint buffer_indices;
    glGenBuffers(1, &buffer_indices);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer_indices);
    unsigned int indices[5] = {0, 1, 2, 3, 0};
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * 5, indices,
                 GL_STATIC_DRAW);

    _program.bind();
    _program.setUniformValue("box_min", _box.topLeft());
    _program.setUniformValue("box_max", _box.bottomRight());
    _program.enableAttributeArray("position");
    _program.setAttributeArray("position", GL_FLOAT, 0, 3);
    glDrawElements(GL_LINE_STRIP, 5, GL_UNSIGNED_INT, (GLvoid*)0);
    _program.disableAttributeArray("position");
    glDeleteBuffers(1, &buffer_square);
    glDeleteBuffers(1, &buffer_indices);

    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
  }

  void click(QPointF p, SelectMode select_mode) {
    _select_mode = select_mode;
    _anchor = p;
    _box = QRectF(p, p);
  }

  void drag(QPointF p) {
    _box = QRectF(p, _anchor);
    _box = _box.normalized();
  }

  void release() {
    _select_mode = NONE;
    _box.setWidth(0.0f);
    _box.setHeight(0.0f);
  }

  bool active() const { return _select_mode != NONE; }

  bool empty() const { return _box.isEmpty(); }

  const QRectF& getBox() const { return _box; }

  SelectMode getType() const { return _select_mode; }

 private:
  void compileProgram() {
    std::string vsCode =
        "#version 110\n"
        "uniform vec2 box_min;\n"
        "uniform vec2 box_max;\n"
        "attribute vec3 position;\n"
        "void main() {\n"
        "  gl_Position = vec4(position.xy * (box_max - box_min) + box_min, 0, 1);\n"
        "}\n";
    std::string fsCode =
        "#version 110\n"
        "void main() {\n"
        "  gl_FragColor = vec4(1, 1, 0, 1);\n"
        "}\n";
    _context->makeCurrent(_window);
    _program.addShaderFromSourceCode(QOpenGLShader::Vertex, vsCode.c_str());
    _program.addShaderFromSourceCode(QOpenGLShader::Fragment, fsCode.c_str());
    _program.link();
    _context->doneCurrent();
  }

  QOpenGLContext* _context;
  QWindow* _window;
  QOpenGLShaderProgram _program;

  SelectMode _select_mode;
  QPointF _anchor;
  QRectF _box;
};

#endif  // __SELECTIONBOX_H__
