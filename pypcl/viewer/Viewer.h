#ifndef _VIEWER_H_
#define _VIEWER_H_
#include <QCoreApplication>
#include <QWindow>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QVector3D>
#include <QMatrix4x4>
#include <QOpenGLContext>
#include <QOpenGLShaderProgram>
#include <QTcpServer>
#include <QTcpSocket>
#include <QTimer>
#include <QImage>
#include <QColor>
#include <QString>
#include <QtCore/qmath.h>
#include <iostream>
#include <fstream>
#include <limits>

#include "QtCamera.h"
#include "Background.h"
#include "FloorGrid.h"
#include "SelectionBox.h"
#include "PointCloud.h"
#include "LookAt.h"
#include "timer.h"
#include "CommFuncs.h"
#include "OpenGLFuncs.h"
#include "Text.h"
#include "CameraDolly.h"

//#define TEST_FINE_RENDER

class Viewer : public QWindow, protected OpenGLFuncs{
	Q_OBJECT
public:
	explicit Viewer(quint16 clientPort) : QWindow() {
		setSurfaceType(QSurface::OpenGLSurface);
		create();

		// create OpenGL context
		_context = new QOpenGLContext();
		QSurfaceFormat f = format();
		f.setDepthBufferSize(16);
		_context->setFormat(f);
		_context->create();

		// associate OpenGL functions with _context
		_context->makeCurrent(this);
		initializeOpenGLFunctions();
		_context->doneCurrent();

		// set font
		QFont font("Courier", 12);

		// initialize various viewer objects
		_background = new Background(this, _context);
		_floor_grid = new FloorGrid(this, _context);
		_look_at = new LookAt(this, _context);
		_points = new PointCloud(this, _context);
		_selection_box = new SelectionBox(this, _context);
		_text = new Text(this, _context, font);
		_dolly = new CameraDolly();

		// initalize various states
		_socket_waiting_on_enter_key = NULL;
		_timer_fine_render_delay = NULL;
		_fine_render_state = INACTIVE;
		_render_time = std::numeric_limits<double>::infinity();
		_show_text = true;

		// set up TCP server for receiving commands from Python terminal (client)
		_server = new QTcpServer();
		if (!_server->listen(QHostAddress::LocalHost,0)) {
			qDebug() << _server->errorString().toLocal8Bit().constData();
			exit(1);
		}
		connect(_server,SIGNAL(newConnection()),this,SLOT(reply()));
		qDebug() << "Viewer: TCP server set up on port " << _server->serverPort();

		quint16 serverPort = _server->serverPort();
		fwrite(&serverPort, sizeof(quint16), 1, stdout);
		fflush(stdout);

		// send TCP server port number back to Python terminal (client)
		QTcpSocket * socket = new QTcpSocket;
		socket->connectToHost("localhost", clientPort);
		/*
		qDebug() << "Viewer: Connecting back to client at port " << clientPort << "...";
		if (!socket->waitForConnected()) {
			std::cout << "Connection back to client failed" << std::endl;
			exit(1);
		}
		quint16 serverPort = _server->serverPort();
		comm::sendBytes((const char*)&serverPort, sizeof(quint16), socket);
		*/
		socket->disconnectFromHost();
	}
	~Viewer() {
		delete _background;
		delete _floor_grid;
		delete _look_at;
		delete _points;
		delete _selection_box;
		delete _text;
		delete _dolly;

		// deletion of context must occur after viewer objects
		delete _context;
		delete _server;
	}

protected:
	virtual void keyPressEvent(QKeyEvent * ev) {
		_dolly->stop();
		if (ev->key() == Qt::Key_5) {
			if (_camera.getProjectionMode() == QtCamera::PERSPECTIVE)
				_camera.setProjectionMode(QtCamera::ORTHOGRAPHIC);
			else
				_camera.setProjectionMode(QtCamera::PERSPECTIVE);
		} else if (ev->key() == Qt::Key_1) {
			_camera.setViewAxis(QtCamera::Y_AXIS);
		} else if (ev->key() == Qt::Key_3) {
			_camera.setViewAxis(QtCamera::X_AXIS);
		} else if (ev->key() == Qt::Key_7) {
			_camera.setViewAxis(QtCamera::Z_AXIS);
		} else if (ev->key() == Qt::Key_Left) {
			_camera.setViewAxis(QtCamera::ARBITRARY_AXIS);
			_camera.setPhi(_camera.getPhi() - 30.0f * _camera.getRotateRate());
		} else if (ev->key() == Qt::Key_Right) {
			_camera.setViewAxis(QtCamera::ARBITRARY_AXIS);
			_camera.setPhi(_camera.getPhi() + 30.0f * _camera.getRotateRate());
		} else if (ev->key() == Qt::Key_Down) {
			_camera.setViewAxis(QtCamera::ARBITRARY_AXIS);
			_camera.setTheta(_camera.getTheta() - 30.0f * _camera.getRotateRate());
		} else if (ev->key() == Qt::Key_Up) {
			_camera.setViewAxis(QtCamera::ARBITRARY_AXIS);
			_camera.setTheta(_camera.getTheta() + 30.0f * _camera.getRotateRate());
		} else if (ev->key() == Qt::Key_BracketLeft) {
			int next_idx = (int)_points->getCurrentAttributeIndex();
			next_idx = next_idx == 0 ? (int)_points->getNumAttributes() - 1 : next_idx - 1;
			_points->setCurrentAttributeIndex((std::size_t)next_idx);
		} else if (ev->key() == Qt::Key_BracketRight) {
			int next_idx = (int)_points->getCurrentAttributeIndex();
			next_idx = (next_idx + 1) % (int)_points->getNumAttributes();
			_points->setCurrentAttributeIndex((std::size_t)next_idx);
		} else if (ev->key() == Qt::Key_C) {
			_camera.setLookAtPosition(_points->computeSelectionCentroid());
			_camera.save();
			renderPoints();
			renderPointsFine();
		} else if ((ev->key() == Qt::Key_Enter || ev->key() == Qt::Key_Return) && _socket_waiting_on_enter_key) {
			const char * msg = "x";
			comm::sendBytes(msg, 1, _socket_waiting_on_enter_key);
			_socket_waiting_on_enter_key->disconnectFromHost();
			_socket_waiting_on_enter_key = NULL;
		} else {
			QWindow::keyPressEvent(ev);
			return;
		}
		renderPoints();
		renderPointsFine();
	}
	virtual void mouseDoubleClickEvent(QMouseEvent * ev) {
		Q_UNUSED(ev);
		_dolly->stop();
		// center on a point near cursor
		std::vector<unsigned int> indices;
		_points->queryNearPoint(indices, ev->windowPos(), _camera);
		if (indices.empty()) return;
		const std::vector<float> & ps = _points->getPositions();
		QVector3D p(ps[3 * indices[0] + 0], ps[3 * indices[0] + 1], ps[3 * indices[0] + 2]);
		_camera.setLookAtPosition(p);
		_camera.save();
		renderPoints();
		renderPointsFine();
	}
	virtual void mousePressEvent(QMouseEvent * ev) {
		_dolly->stop();
		if (ev->buttons() & Qt::LeftButton) {
			_pressPos = ev->windowPos();
			if (ev->modifiers() & Qt::ControlModifier) {
				if (ev->modifiers() & Qt::ShiftModifier)
					_selection_box->click(win2ndc(_pressPos), SelectionBox::SUB);
				else
					_selection_box->click(win2ndc(_pressPos), SelectionBox::ADD);
				renderPoints();
			}
		} else if (ev->buttons() & Qt::RightButton) {
			_points->clearSelected();
			renderPoints();
		} else {
			QWindow::mousePressEvent(ev);
		}
	}
	virtual void mouseMoveEvent(QMouseEvent * ev) {
		// note: +x right, +y down
		if (ev->buttons() & Qt::LeftButton) {
			if (_fine_render_state != INACTIVE)
				_fine_render_state = TERMINATE;
			if (_selection_box->active()) {
				_selection_box->drag(win2ndc(ev->windowPos()));
			} else {
				_camera.restore();
				if (ev->modifiers() == Qt::ShiftModifier)
					_camera.pan(QVector2D(ev->windowPos() - _pressPos) * QVector2D(2.0f / width(), 2.0f / height()));
				else if (ev->modifiers() == Qt::NoModifier)
					_camera.rotate(QVector2D(ev->windowPos() - _pressPos));
			}
			renderPoints();
		} else {
			QWindow::mouseMoveEvent(ev);
		}
	}
	virtual void mouseReleaseEvent(QMouseEvent * ev) {
		Q_UNUSED(ev);
		QPointF releasePos = ev->windowPos();
		bool mouse_moved = releasePos != _pressPos;
		if (_selection_box->active()) {
			if (_selection_box->empty()) {
				bool deselect = _selection_box->getType() == SelectionBox::SUB;
				_points->selectNearPoint(releasePos, _camera, deselect);
			} else {
				_points->selectInBox(*_selection_box, _camera);
			}
			_selection_box->release();
			renderPoints();
			renderPointsFine();
		} else {
			if (mouse_moved) {
				_camera.save();
				renderPoints();
				renderPointsFine();
			}
		}
	}
	virtual void wheelEvent(QWheelEvent * ev) {
		_dolly->stop();
		// note: angleDelta() is in units of 1/8 degree
		_camera.zoom(ev->angleDelta().y() / 120.0f);
		_camera.save();
		renderPoints();
		renderPointsFine(500);
	}
	virtual void exposeEvent(QExposeEvent * ev) {
		Q_UNUSED(ev);
		renderPoints();
		renderPointsFine(1000);
	}
	virtual void resizeEvent(QResizeEvent * ev) {
		Q_UNUSED(ev);
		_camera.setAspectRatio((float)width() / height());
		_context->makeCurrent(this);
		glViewport(0, 0, width(), height());
		_context->doneCurrent();
	}

private slots:
	void reply() {
		QTcpSocket * clientConnection = _server->nextPendingConnection();
		connect(clientConnection,SIGNAL(disconnected()),clientConnection,SLOT(deleteLater()));

		// read first byte of incoming message
		char msgType;
		while (clientConnection->read(&msgType, 1) != 1);

		// switch on message type
		switch(msgType) {
			case 1:	// load points
			{
				qint64 bytesReceived;
				qint64 expectedBytes;
				// receive point count (next 4 bytes)
				qint32 numPoints;
				bytesReceived = 0;
				while (bytesReceived < 4) {
					bytesReceived += clientConnection->read(
						(char*)&numPoints + bytesReceived, 4 - bytesReceived);
				}
				qDebug() << "Viewer: expecting" << numPoints << "points";

				// receive position vectors
				// (next 3 x numPoints x sizeof(float) bytes)
				std::vector<float> positions(3 * numPoints);
				expectedBytes = (qint64)positions.size() * sizeof(float);
				bytesReceived = 0;
				while (bytesReceived < expectedBytes) {
					qint64 received = clientConnection->read(
						(char*)&positions[0] + bytesReceived,
						expectedBytes - bytesReceived);
					if (received == -1) {
						qDebug() << "error during socket read()";
						exit(1);
					}
					bytesReceived += received;
					clientConnection->waitForReadyRead();
				}
				qDebug() << "Viewer: received positions," << bytesReceived << "bytes";

				_points->loadPoints(positions);
				_camera = QtCamera(_points->getBox());
				_camera.setAspectRatio((float)width() / height());
				_floor_grid->setFloorLevel(_points->getFloor());
				renderPoints();
				renderPointsFine();
				break;
			}
			case 2:	// clear points
			{
				_points->clearPoints();
				renderPoints();
				renderPointsFine();
				break;
			}
			case 3:	// reset view to fit all
			{
				_camera = QtCamera(_points->getBox());
				_camera.setAspectRatio((float)width() / height());
				renderPoints();
				renderPointsFine();
				break;
			}
			case 4:	// set viewer property
			{
				// receive length of property name string
				quint64 stringLength;
				comm::receiveBytes((char*)&stringLength,
					sizeof(quint64), clientConnection);

				// receive property name string
				std::string propertyName(stringLength,'x');
				comm::receiveBytes((char*)&propertyName[0],
					(qint64)stringLength, clientConnection);

				//	receive length of payload
				quint64 payloadLength;
				comm::receiveBytes((char*)&payloadLength,
					sizeof(quint64), clientConnection);

				// receive payload
				std::vector<char> payload(payloadLength, 0);
				comm::receiveBytes(&payload[0],
					(qint64)payloadLength, clientConnection);

				// set viewer properties accordingly
				// ignore set requests with unexpected payload lengths
				if (!strcmp(propertyName.c_str(), "point_size")) {
					if (payloadLength != sizeof(GLfloat)) break;
					GLfloat point_size = *(GLfloat*)&payload[0];
					_points->setPointSize(point_size);
					glPointSize(point_size);
				} else if (!strcmp(propertyName.c_str(), "bg_color")) {
					if (payloadLength != 4 * sizeof(float)) break;
					float * rgba = (GLfloat*)&payload[0];
					QVector4D bg_color(rgba[0], rgba[1], rgba[2], rgba[3]);
					_background->setColorTop(bg_color);
					_background->setColorBottom(bg_color);
				} else if (!strcmp(propertyName.c_str(), "bg_color_top")) {
					if (payloadLength != 4 * sizeof(float)) break;
					float * rgba = (float*)&payload[0];
					QVector4D bg_color_top(rgba[0], rgba[1], rgba[2], rgba[3]);
					_background->setColorTop(bg_color_top);
				} else if (!strcmp(propertyName.c_str(), "bg_color_bottom")) {
					if (payloadLength != 4 * sizeof(float)) break;
					float * rgba = (float*)&payload[0];
					QVector4D bg_color_bottom(rgba[0], rgba[1], rgba[2], rgba[3]);
					_background->setColorBottom(bg_color_bottom);
				} else if (!strcmp(propertyName.c_str(), "show_grid")) {
					if (payloadLength != sizeof(bool)) break;
					bool visible = *(bool*)&payload[0];
					_floor_grid->setVisible(visible);
				} else if (!strcmp(propertyName.c_str(), "show_info")) {
					if (payloadLength != sizeof(bool)) break;
					_show_text = *(bool*)&payload[0];
				} else if (!strcmp(propertyName.c_str(), "show_axis")) {
					if (payloadLength != sizeof(bool)) break;
					bool visible = *(bool*)&payload[0];
					_look_at->setVisible(visible);
				} else if (!strcmp(propertyName.c_str(), "floor_level")) {
					if (payloadLength != sizeof(float)) break;
					float floor_level = *(float*)&payload[0];
					_floor_grid->setFloorLevel(floor_level);
				} else if (!strcmp(propertyName.c_str(), "floor_color")) {
					if (payloadLength != 4 * sizeof(float)) break;
					float * rgba = (float*)&payload[0];
					QVector4D floor_color(rgba[0], rgba[1], rgba[2], rgba[3]);
					_floor_grid->setFloorColor(floor_color);
				} else if (!strcmp(propertyName.c_str(), "floor_grid_color")) {
					if (payloadLength != 4 * sizeof(float)) break;
					float * rgba = (float*)&payload[0];
					QVector4D floor_grid_color(rgba[0], rgba[1], rgba[2], rgba[3]);
					_floor_grid->setLineColor(floor_grid_color);
				} else if (!strcmp(propertyName.c_str(), "lookat")) {
					if (payloadLength != 3 * sizeof(float)) break;
					float * xyz = (float*)&payload[0];
					QVector3D lookat(xyz[0], xyz[1], xyz[2]);
					_camera.setLookAtPosition(lookat);
				} else if (!strcmp(propertyName.c_str(), "phi")) {
					if (payloadLength != sizeof(float)) break;
					float phi = *(float*)&payload[0];
					_camera.setPhi(phi);
				} else if (!strcmp(propertyName.c_str(), "theta")) {
					if (payloadLength != sizeof(float)) break;
					float theta = *(float*)&payload[0];
					_camera.setTheta(theta);
				} else if (!strcmp(propertyName.c_str(), "r")) {
					if (payloadLength != sizeof(float)) break;
					float r = *(float*)&payload[0];
					_camera.setCameraDistance(qMax(0.1f, r));
				} else if (!strcmp(propertyName.c_str(), "selected")) {
					quint64 num_selected = payloadLength / sizeof(unsigned int);
					if (payloadLength != num_selected * sizeof(unsigned int)) break;
					unsigned int * ptr = (unsigned int*)&payload[0];
					std::vector<unsigned int> selected;
					selected.reserve(num_selected);
					for (quint64 i = 0; i < num_selected; i++)
						if (ptr[i] < _points->getNumPoints())
							selected.push_back(ptr[i]);	// silently drop out of range indices
					_points->setSelected(selected);
				} else if (!strcmp(propertyName.c_str(), "color_map")) {
					quint64 num_colors = payloadLength / sizeof(float) / 4;
					if (payloadLength != num_colors * sizeof(float) * 4) break;
					float * ptr = (float*)&payload[0];
					std::vector<float> color_map(ptr, ptr + num_colors * 4);
					_points->setColorMap(color_map);
				} else if (!strcmp(propertyName.c_str(), "color_map_scale")) {
					if (payloadLength != sizeof(float) * 2) break;
					float * v = (float*)&payload[0];
					_points->setColorMapScale(v[0], v[1]);
				} else if (!strcmp(propertyName.c_str(), "curr_attribute_id")) {
					if (payloadLength != sizeof(unsigned int)) break;
					_points->setCurrentAttributeIndex(*(unsigned int*)&payload[0]);
				} else {
					// unrecognized property name, do nothing
					// todo: consider doing something
				}
				renderPoints();
				renderPointsFine();
				break;
			}
			case 5:	// get viewer property
			{
				// receive length of property name string
				quint64 stringLength;
				comm::receiveBytes((char*)&stringLength, sizeof(quint64), clientConnection);

				// receive property name string
				std::string propertyName(stringLength, 'x');
				comm::receiveBytes((char*)&propertyName[0], stringLength, clientConnection);

				// send property
				if (!strcmp(propertyName.c_str(), "selected")) {
					std::vector<unsigned int> selected_ids;
					selected_ids.reserve(_points->getNumSelected());
					_points->getSelected(selected_ids);
					comm::sendArray<unsigned int>(&selected_ids[0], selected_ids.size(), clientConnection);
				} else if (!strcmp(propertyName.c_str(), "eye")) {
					float eye[3];
					_camera.getCameraPosition(eye);
					comm::sendArray<float>(&eye[0], 3, clientConnection);
				} else if (!strcmp(propertyName.c_str(), "lookat")) {
					float lookat[3];
					_camera.getLookAtPosition(lookat);
					comm::sendArray<float>(&lookat[0], 3, clientConnection);
				} else if (!strcmp(propertyName.c_str(), "view")) {
					float view[3];
					_camera.getViewVector(view);
					comm::sendArray<float>(&view[0], 3, clientConnection);
				} else if (!strcmp(propertyName.c_str(), "right")) {
					float right[3];
					_camera.getRightVector(right);
					comm::sendArray<float>(&right[0], 3, clientConnection);
				} else if (!strcmp(propertyName.c_str(), "up")) {
					float up[3];
					_camera.getUpVector(up);
					comm::sendArray<float>(&up[0], 3, clientConnection);
				} else if (!strcmp(propertyName.c_str(), "phi")) {
					comm::sendScalar<float>(_camera.getPhi(), clientConnection);
				} else if (!strcmp(propertyName.c_str(), "theta")) {
					comm::sendScalar<float>(_camera.getTheta(), clientConnection);
				} else if (!strcmp(propertyName.c_str(), "r")) {
					comm::sendScalar<float>(_camera.getCameraDistance(), clientConnection);
				} else if (!strcmp(propertyName.c_str(), "mvp")) {
					QMatrix4x4 mvp = _camera.computeMVPMatrix(_points->getBox());
					comm::sendMatrix<float>((float*)mvp.data(), 4, 4, clientConnection);
				} else if (!strcmp(propertyName.c_str(), "num_points")) {
					comm::sendScalar<unsigned int>((unsigned int)_points->getNumPoints(), clientConnection);
				} else if (!strcmp(propertyName.c_str(), "num_attributes")) {
					comm::sendScalar<unsigned int>((unsigned int)_points->getNumAttributes(), clientConnection);
				} else if (!strcmp(propertyName.c_str(), "curr_attribute_id")) {
					comm::sendScalar<unsigned int>((unsigned int)_points->getCurrentAttributeIndex(), clientConnection);
				} else {
					std::string msg = "Unrecognized property name \"" + propertyName + "\"";
					comm::sendError(&msg[0], msg.length(), clientConnection);
				}
				break;
			}
			case 6:	// print screen
			{
				// receive length of property name string
				quint64 stringLength;
				comm::receiveBytes((char*)&stringLength, sizeof(quint64), clientConnection);

				// receive property name string
				std::string filename(stringLength, 'x');
				comm::receiveBytes((char*)&filename[0], stringLength, clientConnection);
				printScreen(filename);
				break;
			}
			case 7: // wait for enter
			{
				// save current connection socket and return
				_socket_waiting_on_enter_key = clientConnection;
				return;
			}
			case 8: // load camera path animation
			{
				// receive number of poses (1 int)
				qint32 numPoses;
				comm::receiveBytes((char*)&numPoses, sizeof(qint32), clientConnection);

				// receive poses (6n floats)
				std::vector<float> poses(6 * numPoses);
				comm::receiveBytes((char*)&poses[0], 6 * numPoses * sizeof(float), clientConnection);

				// receive number of time stamps (1 int)
				qint32 numTimeStamps;
				comm::receiveBytes((char*)&numTimeStamps, sizeof(qint32), clientConnection);

				// receive time stamps (n floats)
				std::vector<float> ts(numTimeStamps);
				comm::receiveBytes((char*)&ts[0], numTimeStamps * sizeof(float), clientConnection);

				// receive interpolation code (1 byte)
				quint8 interp;
				comm::receiveBytes((char*)&interp, sizeof(quint8), clientConnection);

				// reorganize poses into vector of CameraPoses
				std::vector<CameraPose> cam_poses(poses.size() / 6);
				for (int i = 0; i < (int)poses.size() / 6; i++) {
					cam_poses[i].setLookAt(QVector3D(poses[6 * i], poses[6 * i + 1], poses[6 * i + 2]));
					cam_poses[i].setPhi(poses[6 * i + 3]);
					cam_poses[i].setTheta(poses[6 * i + 4]);
					cam_poses[i].setD(poses[6 * i + 5]);
				}

				// create new camera dolly
				delete _dolly;
				_dolly = new CameraDolly(ts, cam_poses, (CameraDolly::InterpolationType)interp);

				break;
			}
			case 9: // playback camera path animation
			{
				// receive playback time range (2 float)
				float tmin, tmax;
				comm::receiveBytes((char*)&tmin, sizeof(float), clientConnection);
				comm::receiveBytes((char*)&tmax, sizeof(float), clientConnection);

				// receive repeat flag (1 bool)
				bool repeat;
				comm::receiveBytes((char*)&repeat, sizeof(bool), clientConnection);

				// start playback
				_dolly->setStartTime(tmin);
				_dolly->setEndTime(tmax);
				_dolly->setRepeat(repeat);
				_dolly->start();
				playCameraAnimation();

				break;
			}
			case 10: // set per point attributes
			{
				//	receive length of payload
				quint64 payloadLength;
				comm::receiveBytes((char*)&payloadLength,
					sizeof(quint64), clientConnection);

				// receive payload
				std::vector<char> payload(payloadLength, 0);
				comm::receiveBytes(&payload[0],
					(qint64)payloadLength, clientConnection);

				_points->loadAttributes(payload);
				renderPoints();
				renderPointsFine();
				break;
			}
			default :	// unrecognized message type
				break;
				// do nothing
		}
		clientConnection->write("1234");
		clientConnection->disconnectFromHost();
	}
	void drawRefinedPointsDelayed() {
		if (_fine_render_state == INACTIVE)
			QTimer::singleShot(0, this, SLOT(drawRefinedPoints()));
		_fine_render_state = INITIALIZE;
		delete _timer_fine_render_delay;
		_timer_fine_render_delay = NULL;
	}
	void drawRefinedPoints() {
		switch (_fine_render_state) {
			case INACTIVE:
			{
				#ifdef TEST_FINE_RENDER
				qDebug() << "should not be here" << std::endl;
				#endif
				break;
			}
			case INITIALIZE:
			{
				#ifdef TEST_FINE_RENDER
				qDebug() << "fine render: initialize" << std::endl;
				_max_chunk_size = 1;
				_chunk_offset = 0;
				_refined_indices.clear();
				for (int i = 0; i < 10; i++)
					_refined_indices.push_back(i);
				#else
				// get points at finest LOD, for the current image resolution
				_points->queryLOD(_refined_indices, _camera, 1.0f);

				_max_chunk_size = 50000;
				_chunk_offset = 0;

				// draw background and grid
				_context->makeCurrent(this);
				glClearColor(0.0f,0.0f,0.0f,1.0f);
				glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
				_background->draw();
				_floor_grid->draw(_camera);
				_context->doneCurrent();
				#endif
				_fine_render_state = CHUNK;
				QCoreApplication::processEvents();
				QTimer::singleShot(0, this, SLOT(drawRefinedPoints()));
				break;
			}
			case CHUNK:
			{
				std::size_t chunk_size = _refined_indices.size() - _chunk_offset;
				if (chunk_size > _max_chunk_size)
					chunk_size = _max_chunk_size;
				#ifdef TEST_FINE_RENDER
				double t = vltools::getTime();
				dummyCalculation(1000000);
				t = vltools::getTime() - t;
				qDebug() << "fine render: processed " << _chunk_offset << ", " << chunk_size << ", " << t << std::endl;
				#else
				_context->makeCurrent(this);
				if (chunk_size > 0)
					_points->draw(&_refined_indices[_chunk_offset], (unsigned int)chunk_size, _camera, _selection_box);
				_context->doneCurrent();
				#endif
				_chunk_offset += chunk_size;
				if (_chunk_offset == _refined_indices.size())
					_fine_render_state = FINALIZE;
				else
					_fine_render_state = CHUNK;
				QCoreApplication::processEvents();
				QTimer::singleShot(0, this, SLOT(drawRefinedPoints()));
				break;
			}
			case FINALIZE:
			{
				#ifdef TEST_FINE_RENDER
				qDebug() << "fine render: finalized" << std::endl;
				#else
				_context->makeCurrent(this);
				_look_at->draw(_camera);
				displayInfo();
				if (this->isExposed())
					_context->swapBuffers(this);
				_context->doneCurrent();
				#endif
				_fine_render_state = INACTIVE;
				break;
			}
			case TERMINATE:
			{
				#ifdef TEST_FINE_RENDER
				qDebug() << "fine render: terminate" << std::endl;
				#endif
				_fine_render_state = INACTIVE;
				break;
			}
		}
	}
	void playCameraAnimation() {
		if (_dolly->done()) {
			renderPointsFine();
			return;
		}
		CameraPose pose = _dolly->getPose();
		_camera.setLookAtPosition(pose.lookAt());
		_camera.setPhi(pose.phi());
		_camera.setTheta(pose.theta());
		_camera.setCameraDistance(qMax(0.1f, pose.d()));
		_camera.save();
		renderPoints();
		QTimer::singleShot(15, this, SLOT(playCameraAnimation()));
	}

private:
	void dummyCalculation(int n) {
		_dummy_accumulator /= (float)n;
		for (int i = 0; i < n; i++)
			_dummy_accumulator += sqrtf(pow(6.9f,log((float)i)));
	}
	void printScreen(std::string filename)
	{
		_context->makeCurrent(this);
		int w = width() * this->devicePixelRatio();
		int h = height() * this->devicePixelRatio();
		GLubyte * pixels = new GLubyte[4 * w * h];
		glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glReadBuffer(GL_FRONT);	// otherwise will read back buffer
		glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
		QImage image(w, h, QImage::Format_ARGB32);
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				int index = j * 4 + (h - i - 1) * w * 4;
				QColor color(
						pixels[index + 0],
						pixels[index + 1],
						pixels[index + 2],
						pixels[index + 3]);
				image.setPixel(j, i, color.rgba());
			}
		}
		// expect absolute filename path
		QString qstr_filename = QString::fromStdString(filename);
		image.save(qstr_filename);
		delete[] pixels;
		_context->doneCurrent();
	}
	void displayInfo()
	{
		if (!_show_text) return;

		// assumes _context is current
		float pad = 4.0f;
		float cursor_x = pad;
		float cursor_y = pad;

		// display look-at coordinates
		QVector3D p = _camera.getLookAtPosition();
		QString lookat_text;
		lookat_text.sprintf("Look-at position: x = %.3f, y = %.3f, z = %.3f", p.x(), p.y(), p.z());
		cursor_y += _text->renderText(cursor_x, cursor_y, lookat_text).height();

		// display # points loaded
		QString numpoints_text;
		numpoints_text.sprintf("%d points loaded", (int)_points->getNumPoints());
		cursor_y += _text->renderText(cursor_x, cursor_y, numpoints_text).height();

		// display # points selected
		std::size_t num_selected = _points->getNumSelected();
		QString selected_text;
		if (num_selected == 1) {
			unsigned int selected_id = _points->getSelectedIds()[0];
			const float * pos = &_points->getPositions()[3 * selected_id];
			cursor_y += _text->renderText(cursor_x, cursor_y, selected_text).height();
			selected_text.sprintf("   z = %.3f", pos[2]);
			cursor_y += _text->renderText(cursor_x, cursor_y, selected_text).height();
			selected_text.sprintf("   y = %.3f", pos[1]);
			cursor_y += _text->renderText(cursor_x, cursor_y, selected_text).height();
			selected_text.sprintf("   x = %.3f", pos[0]);
			cursor_y += _text->renderText(cursor_x, cursor_y, selected_text).height();

			// display attribute value
			const PointAttributes & attr = _points->getAttributes();
			quint64 attr_dim = _points->getAttributes().dim((int)_points->getCurrentAttributeIndex());
			if (attr_dim == 1) {
				selected_text.sprintf("   v = %.3f", attr(selected_id, 0));
				cursor_y += _text->renderText(cursor_x, cursor_y, selected_text).height();
			} else if (attr_dim == 4) {
				selected_text.sprintf("   a = %.3f", attr(selected_id, 3));
				cursor_y += _text->renderText(cursor_x, cursor_y, selected_text).height();
				selected_text.sprintf("   b = %.3f", attr(selected_id, 2 ));
				cursor_y += _text->renderText(cursor_x, cursor_y, selected_text).height();
				selected_text.sprintf("   g = %.3f", attr(selected_id, 1));
				cursor_y += _text->renderText(cursor_x, cursor_y, selected_text).height();
				selected_text.sprintf("   r = %.3f", attr(selected_id, 0));
				cursor_y += _text->renderText(cursor_x, cursor_y, selected_text).height();
			}
			selected_text.sprintf("Selected point:");
			_text->renderText(cursor_x, cursor_y, selected_text);
		} else if (num_selected > 1) {
			selected_text.sprintf("%d points selected", (int)num_selected);
			_text->renderText(cursor_x, cursor_y, selected_text);
		}

		// display grid scale
		cursor_x = pad;
		cursor_y = height() - pad - _text->computeTextSize("log of grid size: ").height();
		QVector4D grid_line_color = _floor_grid->getLineColor();
		int log_grid_size = qRound(qLn(_floor_grid->getCellSize()) / qLn(10.0f));
		cursor_x += _text->renderText(cursor_x, cursor_y, "log of grid size: ").width();
		cursor_x += _text->renderText(cursor_x, cursor_y, QString::number(log_grid_size), qPow(_floor_grid->getLineWeight(), 0.25f) * grid_line_color).width();
		cursor_x += _text->renderText(cursor_x, cursor_y, " | ").width();
		_text->renderText(cursor_x, cursor_y, QString::number(log_grid_size + 1), grid_line_color).width();

		// display current attribute id
		QString attr_text;
		int curr_attr = (int)_points->getCurrentAttributeIndex();
		int num_attr = (int)_points->getNumAttributes();
		attr_text.sprintf("Attribute %d of %d", curr_attr + 1, num_attr);
		cursor_x = pad;
		cursor_y -= _text->computeTextSize(attr_text).height();
		_text->renderText(cursor_x, cursor_y, attr_text).height();

		// display port number
		QString port_text;
		port_text.sprintf("port %d", _server->serverPort());
		QSizeF port_text_size = _text->computeTextSize(port_text);
		cursor_x = width() - pad - port_text_size.width();
		cursor_y = height() - pad - port_text_size.height();
		_text->renderText(cursor_x, cursor_y, port_text);

		// display fps
		QString fps_text;
		fps_text.sprintf("%.1f fps", 1.0f / _render_time);
		QSizeF fps_text_size = _text->computeTextSize(fps_text);
		cursor_x = width() - pad - fps_text_size.width();
		cursor_y -= fps_text_size.height();
		_text->renderText(cursor_x, cursor_y, fps_text);
	}

	void renderPointsFine(int msec = 0) {
		// assumes timer points to valid memory if it is non NULL
		delete _timer_fine_render_delay;
		_timer_fine_render_delay = new QTimer(this);
		connect(_timer_fine_render_delay, SIGNAL(timeout()), this, SLOT(drawRefinedPointsDelayed()));
		_timer_fine_render_delay->start(msec);
		if (_fine_render_state != INACTIVE)
			_fine_render_state = TERMINATE;
	}
	void renderPoints() {
		if (!_context->makeCurrent(this))
			return;
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glEnable(GL_DEPTH_TEST);
		glEnable(GL_POINT_SPRITE);
		_render_time = vltools::getTime();
		_background->draw();
		_floor_grid->draw(_camera);
		_points->draw(_camera, _selection_box);
		_look_at->draw(_camera);
		_selection_box->draw();
		_render_time = vltools::getTime() - _render_time;
		displayInfo();
		if (this->isExposed())
			_context->swapBuffers(this);
		_context->doneCurrent();

	}
	QPointF win2ndc(QPointF p) {
		QVector2D v = QVector2D(p) * QVector2D(2.0f / width(), -2.0f / height()) + QVector2D(-1.0f, 1.0f);
		return QPointF(v.x(), v.y());
	}

	QTcpServer * _server;
	QPointF _pressPos;
	QOpenGLContext * _context;

	QtCamera _camera;
	FloorGrid * _floor_grid;
	SelectionBox * _selection_box;
	PointCloud * _points;
	Background * _background;
	LookAt * _look_at;
	Text * _text;
	CameraDolly * _dolly;

	float _dummy_accumulator;
	enum FineRenderState {INACTIVE, INITIALIZE, CHUNK, FINALIZE, TERMINATE};
	FineRenderState _fine_render_state;
	QTimer * _timer_fine_render_delay;
	std::size_t _chunk_offset;
	std::size_t _max_chunk_size;
	std::vector<unsigned int> _refined_indices;

	QTcpSocket * _socket_waiting_on_enter_key;
	double _render_time;
	bool _show_text;
};

#endif
