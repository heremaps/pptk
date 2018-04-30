#ifndef __OPENGLFUNCS_H__
#define __OPENGLFUNCS_H__
#include <QOpenGLFunctions>

class OpenGLFuncs : public QOpenGLFunctions {
  // extends QOpenGLFunctions with some helper and error checking functions
 public:
  OpenGLFuncs() : QOpenGLFunctions() {}

  OpenGLFuncs(QOpenGLContext* context) : QOpenGLFunctions(context) {}

  GLint getBufferSize(GLuint bufferId) {
    GLint bufferSize = 0;
    glBindBuffer(GL_ARRAY_BUFFER, bufferId);
    glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &bufferSize);
    return bufferSize;
  }

  void checkError() {
    GLenum e = glGetError();
    switch (e) {
      case GL_INVALID_ENUM:
        qDebug() << "GLenum argument out of range";
        break;
      case GL_INVALID_VALUE:
        qDebug() << "Numeric argument out of range";
        break;
      case GL_INVALID_OPERATION:
        qDebug() << "Operation illegal in current state";
        break;
      case GL_STACK_OVERFLOW:
        qDebug() << "Command would cause a stack overflow";
        break;
      case GL_STACK_UNDERFLOW:
        qDebug() << "Command would cause a stack underflow";
        break;
      case GL_OUT_OF_MEMORY:
        qDebug() << "Not enough memory left to execute command";
        break;
      case GL_NO_ERROR:
        qDebug() << "No error";
    }
  }

  void printFramebufferStatus() {
    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    switch (status) {
      case GL_FRAMEBUFFER_COMPLETE:
        std::cout << "GL_FRAMEBUFFER_COMPLETE" << std::endl;
        break;
      case GL_FRAMEBUFFER_UNDEFINED:
        std::cout << "GL_FRAMEBUFFER_UNDEFINED" << std::endl;
        break;
      case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
        std::cout << "GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT" << std::endl;
        break;
      case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
        std::cout << "GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT"
                  << std::endl;
        break;
      case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
        std::cout << "GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER" << std::endl;
        break;
      case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
        std::cout << "GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER" << std::endl;
        break;
      case GL_FRAMEBUFFER_UNSUPPORTED:
        std::cout << "GL_FRAMEBUFFER_UNSUPPORTED" << std::endl;
        break;
      case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
        std::cout << "GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE" << std::endl;
        break;
      case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS:
        std::cout << "GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS" << std::endl;
        break;
    }
  }
};

#endif  // __OPENGLFUNCS_H__
