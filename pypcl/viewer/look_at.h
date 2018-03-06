#ifndef __LOOKAT_H__
#define __LOOKAT_H__
#include <QOpenGLContext>
#include <QOpenGLShaderProgram>
#include <QWindow>
#include "opengl_funcs.h"
#include "qt_camera.h"

class LookAt : protected OpenGLFuncs {
 public:
  LookAt(QWindow* window, QOpenGLContext* context)
      : _context(context), _window(window), _visible(true) {
    _context->makeCurrent(_window);
    initializeOpenGLFunctions();
    _context->doneCurrent();
    compileProgram();
  }
  void draw(const QtCamera& camera) {
    if (!_visible) return;

    QVector3D lookat = camera.getLookAtPosition();
    float d = 0.0625 * camera.getCameraDistance();
    vltools::Box3<float> lookatBox(lookat.x() - d, lookat.x() + d,
                                   lookat.y() - d, lookat.y() + d,
                                   lookat.z() - d, lookat.z() + d);

    _program.bind();
    _program.setUniformValue("mvp", camera.computeMVPMatrix(lookatBox));
    _program.setUniformValue("d", d);
    _program.setUniformValue("lookat", lookat);

    float positions[18] = {
      0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    float colors[18] = {
      1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
      0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f,
      0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f};

    GLuint buffer_positions;
    glGenBuffers(1, &buffer_positions);
    glBindBuffer(GL_ARRAY_BUFFER, buffer_positions);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 18, positions,
                 GL_STATIC_DRAW);
    _program.enableAttributeArray("position");
    _program.setAttributeArray("position", GL_FLOAT, 0, 3);

    GLuint buffer_colors;
    glGenBuffers(1, &buffer_colors);
    glBindBuffer(GL_ARRAY_BUFFER, buffer_colors);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 18, colors, GL_STATIC_DRAW);
    _program.enableAttributeArray("color");
    _program.setAttributeArray("color", GL_FLOAT, 0, 3);

    glLineWidth(2.0f);
    glDrawArrays(GL_LINES, 0, 6);

    _program.disableAttributeArray("position");
    _program.disableAttributeArray("color");
    glDeleteBuffers(1, &buffer_positions);
    glDeleteBuffers(1, &buffer_colors);
  }

  void setVisible(bool visible) { _visible = visible; }
  bool getVisible() const { return _visible; }

 private:
  void compileProgram() {
    std::string vsCode =
        "#version 110\n"
        "uniform float d;\n"
        "uniform vec3 lookat;\n"
        "uniform mat4 mvp;\n"
        "attribute vec3 position;\n"
        "attribute vec3 color;\n"
        "varying vec3 vcolor;\n"
        "void main() {\n"
        "  gl_Position = mvp * vec4(d * position + lookat, 1);\n"
        "  vcolor = color;\n"
        "}\n";
    std::string fsCode =
        "#version 110\n"
        "varying vec3 vcolor;\n"
        "void main() {\n"
        "  gl_FragColor = vec4(vcolor, 1);\n"
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
  bool _visible;
};

#endif  // __LOOKAT_H__
