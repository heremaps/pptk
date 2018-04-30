#ifndef QMOUSECONTROLLEDCAMERA_H
#define QMOUSECONTROLLEDCAMERA_H
#include "camera.h"
#include <QVector2D>
#include <QVector3D>
#include <QMatrix4x4>
#include <math.h>
#include "box3.h"
class QtCamera : public Camera
{
	// adapter class that utilizes Qt features
	// (Parent class Camera is Qt-independent)
public:
	enum ProjectionMode {PERSPECTIVE = 0, ORTHOGRAPHIC = 1};
	enum ViewAxis {ARBITRARY_AXIS, X_AXIS, Y_AXIS, Z_AXIS};

	QtCamera() : Camera(),
		_vfov(pi()/4.0f), _aspect_ratio(1.0f), _projection_mode(PERSPECTIVE), _view_axis(ARBITRARY_AXIS) {}
	QtCamera(float xmin, float xmax, float ymin, float ymax, float zmin, float zmax) :
		Camera(xmin, xmax, ymin, ymax, zmin, zmax),
		_vfov(pi()/4.0f), _aspect_ratio(1.0f), _projection_mode(PERSPECTIVE), _view_axis(ARBITRARY_AXIS) {}
	QtCamera(const QVector3D & lo, const QVector3D & hi) :
		Camera(lo.x(), hi.x(), lo.y(), hi.y(), lo.z(), hi.z()),
		_vfov(pi()/4.0f), _aspect_ratio(1.0f), _projection_mode(PERSPECTIVE), _view_axis(ARBITRARY_AXIS) {}
	QtCamera(const vltools::Box3<float> & box) :
		Camera(box.x(0), box.x(1), box.y(0), box.y(1), box.z(0), box.z(1)),
		_vfov(pi()/4.0f), _aspect_ratio(1.0f), _projection_mode(PERSPECTIVE), _view_axis(ARBITRARY_AXIS) {}

	QVector3D getCameraPosition() const {
		float p[3];
		Camera::getCameraPosition(p);
		return QVector3D(p[0], p[1], p[2]);
	}
	QVector3D getLookAtPosition() const {
		float p[3];
		Camera::getLookAtPosition(p);
		return QVector3D(p[0], p[1], p[2]);
	}
	QVector3D getRightVector() const {
		float v[3];
		Camera::getRightVector(v);
		return QVector3D(v[0], v[1], v[2]);
	}
	QVector3D getUpVector() const {
		float v[3];
		Camera::getUpVector(v);
		return QVector3D(v[0], v[1], v[2]);
	}
	QVector3D getViewVector() const {
		float v[3];
		Camera::getViewVector(v);
		return QVector3D(v[0], v[1], v[2]);
	}
	void setLookAtPosition(const QVector3D & p) {
		float buf[3] = {p.x(), p.y(), p.z()};
		Camera::setLookAtPosition(buf);
	}

	using Camera::computeRightVector;
	using Camera::computeUpVector;
	using Camera::computeViewVector;
	using Camera::getCameraDistance;
	using Camera::getCameraPosition;
	using Camera::getLookAtPosition;
	using Camera::getPanRate;
	using Camera::getPhi;
	using Camera::getRightVector;
	using Camera::getRotateRate;
	using Camera::getTheta;
	using Camera::getUpVector;
	using Camera::getViewVector;
	using Camera::getZoomRate;
	using Camera::pan;
	using Camera::restore;
	using Camera::rotate;
	using Camera::save;
	using Camera::setCameraDistance;
	using Camera::setLookAtPosition;
	using Camera::setPanRate;
	using Camera::setPhi;
	using Camera::setRotateRate;
	using Camera::setTheta;
	using Camera::setZoomRate;
	using Camera::zoom;

	void pan(QVector2D delta) {
		// delta in ndc scale
		if (delta.x() == 0.0f && delta.y() == 0.0f)
			return;
		float h = getCameraDistance() * tan (0.5f * _vfov);
		float w = _aspect_ratio * h;
		delta *= QVector2D(w, h) / getPanRate();
		Camera::pan(delta.x(), delta.y());
	}
	void rotate(QVector2D delta) {
		// delta in screen space pixel scale
		if (delta.x() == 0.0f && delta.y() == 0.0f)
			return;
		if (_view_axis != ARBITRARY_AXIS)
			_view_axis = ARBITRARY_AXIS;
		Camera::rotate(delta.x(), delta.y());
	}
	float getTop() const
	{
		float t = tan(0.5f * _vfov);
		if (_projection_mode == ORTHOGRAPHIC)
			t *= getCameraDistance();
		return t;
	}
	float getRight() const
	{
		return _aspect_ratio * getTop();
	}
	float getAspectRatio() const {return _aspect_ratio;}
	float getVerticalFOV() const {return _vfov;}
	ProjectionMode getProjectionMode() const {return _projection_mode;}
	ViewAxis getViewAxis() const {return _view_axis;}

	void setAspectRatio(float aspect_ratio) {_aspect_ratio = aspect_ratio;}
	void setVerticalFOV(float vfov) {_vfov = vfov;}
	void setProjectionMode(ProjectionMode mode) {_projection_mode = mode;}
	void setViewAxis(ViewAxis axis)
	{
		if (axis == X_AXIS) {
			setPhi(0.0f);
			setTheta(0.0f);
		} else if (axis == Y_AXIS) {
			setPhi(-0.5f * pi());
			setTheta(0.0f);
		} else if (axis == Z_AXIS) {
			setPhi(-0.5f * pi());
			setTheta(0.5f * pi());
		}
		_view_axis = axis;
	}

	QMatrix4x4 computeMVPMatrix(const vltools::Box3<float> & box) const {
		QMatrix4x4 matrix;
		matrix.setToIdentity();
		float d_near, d_far; computeNearFar(d_near, d_far, box);
		if (_projection_mode == PERSPECTIVE) {
			matrix.perspective(_vfov / pi() * 180.0f, _aspect_ratio, std::max(0.1f, 0.8f * d_near), 1.2f * d_far);
		} else {
			float t = getCameraDistance() * tan(0.5f * _vfov);
			float r = _aspect_ratio * t;
			matrix.ortho(-r, r, -t, t, 0.8f * d_near, 1.2f * d_far);
		}
		matrix.lookAt(getCameraPosition(), getLookAtPosition(), getUpVector());
		return matrix;
	}

private:
	void computeNearFar(float & d_near, float & d_far, const vltools::Box3<float> & box) const {
		d_near = std::numeric_limits<float>::max();
		d_far = -std::numeric_limits<float>::max();

		QVector3D view = getViewVector();
		QVector3D eye = getCameraPosition();
		for (std::size_t i = 0; i < 2; i++) {
			for (std::size_t j = 0; j < 2; j++) {
				for (std::size_t k = 0; k < 2; k++) {
					QVector3D corner(box.x(i), box.y(j), box.z(k));
					float t = QVector3D::dotProduct(corner - eye, -view);
					d_near = std::min(d_near, t);
					d_far = std::max(d_far, t);
				}
			}
		}
	}
	static float pi() {return atan2(0.0, -1.0);}
	float _vfov;	// vertical fov in radians
	float _aspect_ratio;	// width / height
	ProjectionMode _projection_mode;
	ViewAxis _view_axis;
};

#endif // QMOUSECONTROLLEDCAMERA_H
