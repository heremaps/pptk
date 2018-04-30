#ifndef __BACKGROUND_H__
#define __BACKGROUND_H__
#include <QOpenGLContext>
#include <QOpenGLShaderProgram>
#include <QWindow>
#include "opengl_funcs.h"
class Background : protected OpenGLFuncs {
 public:
  Background(QWindow *window, QOpenGLContext *context)
      : _context(context),
        _window(window),
        _bg_color_top(0.0f, 0.0f, 0.0f, 1.0f),
        _bg_color_bottom(0.23f, 0.23f, 0.44f, 1.0f) {
    _context->makeCurrent(_window);
    initializeOpenGLFunctions();
    _context->doneCurrent();
    compileProgram();
  }
  void draw() {
    glDepthMask(GL_FALSE);
    glDisable(GL_DEPTH_TEST);


    float points[12] = {
        0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        1.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 0.0f};
    GLuint square;
    glGenBuffers(1, &square);
    glBindBuffer(GL_ARRAY_BUFFER, square);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 12, (GLvoid *)points,
                 GL_STATIC_DRAW);

    unsigned int indices[6] = {
        0, 1, 2,
        0, 2, 3};
    GLuint square_indices;
    glGenBuffers(1, &square_indices);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, square_indices);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * 6,
                 (GLvoid *)indices, GL_STATIC_DRAW);

    _program.bind();
    _program.setUniformValue("colorBottom", _bg_color_bottom);
    _program.setUniformValue("colorTop", _bg_color_top);
    _program.enableAttributeArray("position");
    _program.setAttributeArray("position", GL_FLOAT, 0, 3);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    _program.disableAttributeArray("position");
    glDeleteBuffers(1, &square);
    glDeleteBuffers(1, &square_indices);

    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
  }

  void setColorTop(QVector4D c) { _bg_color_top = c; }
  void setColorBottom(QVector4D c) { _bg_color_bottom = c; }
  QVector4D getColorTop() const { return _bg_color_top; }
  QVector4D getColorBottom() const { return _bg_color_bottom; }

 private:
  void compileProgram() {
    std::string vsCode =
        "#version 110\n"
        "\n"
        "attribute vec4 position;\n"
        "varying vec2 coordinate;\n"
        "void main() {\n"
        "  gl_Position = vec4(2.0*position.xy-1.0,0,1);\n"
        "  coordinate = position.xy;\n"
        "}\n";
    std::string fsCode =
        "#version 110\n"
        "\n"
        "uniform vec4 colorBottom;\n"
        "uniform vec4 colorTop;\n"
        "varying vec2 coordinate;\n"
        "void main() {\n"
        "  gl_FragColor = mix(colorBottom, colorTop, coordinate.y);\n"
        "}\n";

    _context->makeCurrent(_window);
    _program.addShaderFromSourceCode(QOpenGLShader::Vertex, vsCode.c_str());
    _program.addShaderFromSourceCode(QOpenGLShader::Fragment, fsCode.c_str());
    _program.link();
    _context->doneCurrent();
  }

  QOpenGLContext *_context;
  QWindow *_window;
  QOpenGLShaderProgram _program;
  QVector4D _bg_color_top;
  QVector4D _bg_color_bottom;
};

#endif  // __BACKGROUND_H__
