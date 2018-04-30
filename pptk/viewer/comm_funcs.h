#ifndef __COMMFUNCS_H__
#define __COMMFUNCS_H__
#include <QTcpSocket>
#include <vector>

namespace comm {
template <typename T>
struct TypeCode {
  static const unsigned char value = 0;
};
template <>
struct TypeCode<char> {
  static const unsigned char value = 1;
};
template <>
struct TypeCode<float> {
  static const unsigned char value = 2;
};
template <>
struct TypeCode<int> {
  static const unsigned char value = 3;
};
template <>
struct TypeCode<unsigned int> {
  static const unsigned char value = 4;
};

inline void receiveBytes(char* destination, const qint64 bytesExpected,
                         QTcpSocket* clientConnection) {
  // notes: read() can read just part of buffer
  // but waitForReadyRead() unblocks only on receiving *new* data
  // not-yet-read data in buffer is not considered new
  qint64 bytesReceived = 0;
  while (bytesReceived < bytesExpected) {
    qint64 received =
        clientConnection->read(destination, bytesExpected - bytesReceived);
    if (received == 0) clientConnection->waitForReadyRead(-1);
    if (received == -1) {
      qDebug() << "error during socket read()";
      exit(1);
    }
    bytesReceived += received;
    destination += received;
  }
}

inline void sendBytes(const char* source, const qint64 size,
                      QTcpSocket* clientConnection) {
  qint64 bytesLeft = size;
  const char* buf = source;
  while (bytesLeft > 0) {
    qint64 bytesSent = clientConnection->write(buf, bytesLeft);
    if (bytesSent == -1) {
      qDebug() << "error during socket write()";
      exit(1);
    }
    buf += bytesSent;
    bytesLeft -= bytesSent;
    clientConnection->waitForBytesWritten();
  }
}

template <typename T>
void sendScalar(const T value, QTcpSocket* clientConnection) {
  // send data type
  unsigned char dataType = TypeCode<T>::value;
  sendBytes((char*)&dataType, 1, clientConnection);

  // send number of dimensions
  quint64 numDims = 1;
  sendBytes((char*)&numDims, sizeof(quint64), clientConnection);

  // send dimensions
  quint64 numElts = 1;
  sendBytes((char*)&numElts, sizeof(quint64), clientConnection);

  // send array elements
  sendBytes((char*)&value, sizeof(T), clientConnection);
}

template <typename T>
void sendArray(const T* source, const quint64 size,
               QTcpSocket* clientConnection) {
  // send data type
  unsigned char dataType = TypeCode<T>::value;
  sendBytes((char*)&dataType, 1, clientConnection);

  // send number of dimensions
  quint64 numDims = 1;
  sendBytes((char*)&numDims, sizeof(quint64), clientConnection);

  // send dimensions
  sendBytes((char*)&size, sizeof(quint64), clientConnection);

  // send array elements
  sendBytes((char*)source, sizeof(T) * size, clientConnection);
}

inline void sendError(const char* msg, const quint64 size,
                      QTcpSocket* clientConnection) {
  // send data type
  unsigned char dataType = 0;
  sendBytes((char*)&dataType, 1, clientConnection);

  // send number of dimensions
  quint64 numDims = 1;
  sendBytes((char*)&numDims, sizeof(quint64), clientConnection);

  // send dimensions
  sendBytes((char*)&size, sizeof(quint64), clientConnection);

  // send array elements
  sendBytes((char*)msg, sizeof(char) * size, clientConnection);
}

template <typename T>
void sendMatrix(const T* source,  // in column major order
                const quint64 numRows, const quint64 numCols,
                QTcpSocket* clientConnection) {
  // send data type
  unsigned char dataType = TypeCode<T>::value;
  sendBytes((char*)&dataType, 1, clientConnection);

  // send number of dimensions
  quint64 numDims = 2;
  sendBytes((char*)&numDims, sizeof(quint64), clientConnection);

  // send dimensions
  quint64 dims[2] = {numRows, numCols};
  sendBytes((char*)&dims[0], 2 * sizeof(quint64), clientConnection);

  // send array elements
  sendBytes((char*)source, sizeof(T) * numRows * numCols, clientConnection);
}

template <typename T>
void sendMultiDimArray(const T* source, const std::vector<quint64>& dims,
                       QTcpSocket* clientConnection) {
  // send data type
  unsigned char dataType = TypeCode<T>::value;
  sendBytes((char*)&dataType, 1, clientConnection);

  // send number of dimensions
  quint64 numDims = dims.size();
  sendBytes((char*)&numDims, sizeof(quint64), clientConnection);

  // send dimensions
  quint64 numElts = 1;
  for (std::size_t i = 0; i < dims.size(); i++) {
    sendBytes((char*)&dims[i], sizeof(quint64), clientConnection);
    numElts *= dims[i];
  }

  // send array elements
  sendBytes((char*)source, sizeof(T) * numElts, clientConnection);
}
}
#endif  // __COMMFUNCS_H__
