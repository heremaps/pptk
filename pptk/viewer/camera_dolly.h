#ifndef __CAMERADOLLY_H__
#define __CAMERADOLLY_H__
#include <QVector3D>
#include <QtGlobal>
#include <vector>
#include "splines.h"
#include "timer.h"

class CameraPose {
 public:
  CameraPose() : _look_at(), _phi(0.0f), _theta(0.0f), _d(1.0f) {}
  CameraPose(const QVector3D& p, float phi, float theta, float d)
      : _look_at(p), _phi(phi), _theta(theta), _d(d) {}

  // getter functions
  const QVector3D& lookAt() const { return _look_at; }
  float phi() const { return _phi; }
  float theta() const { return _theta; }
  float d() const { return _d; }

  // setter functions
  void setLookAt(const QVector3D& p) { _look_at = p; }
  void setPhi(float phi) { _phi = phi; }
  void setTheta(float theta) { _theta = theta; }
  void setD(float d) { _d = d; }

 private:
  QVector3D _look_at;
  float _phi;
  float _theta;
  float _d;
};

struct CameraPosesSOA {
  CameraPosesSOA(std::vector<CameraPose>& poses) {
    x.resize(poses.size());
    y.resize(poses.size());
    z.resize(poses.size());
    phi.resize(poses.size());
    theta.resize(poses.size());
    d.resize(poses.size());
    for (int i = 0; i < (int)poses.size(); i++) {
      x[i] = poses[i].lookAt().x();
      y[i] = poses[i].lookAt().y();
      z[i] = poses[i].lookAt().z();
      phi[i] = poses[i].phi();
      theta[i] = poses[i].theta();
      d[i] = poses[i].d();
    }
  }
  std::vector<float> x;
  std::vector<float> y;
  std::vector<float> z;
  std::vector<float> phi;
  std::vector<float> theta;
  std::vector<float> d;
};

class CameraDolly {
 public:
  enum InterpolationType { CONSTANT, LINEAR, CUBIC_NATURAL, CUBIC_PERIODIC };

  CameraDolly() : _interp_type(LINEAR), _repeat(false), _active(false) {
    check_and_init();
    compute_splines();
  }

  /*! \brief CameraDolly
   *  Assumes key poses sorted in increasing time
   *  Assumes ts.size() == poses.size() and ts.size() > 0
   *  Assumes ts consists of unique time instances
   *  TODO: consider introducing a CameraPose class
   *        i.e. for consolidating lookAt, phi, theta, d parameters
   */
  CameraDolly(const std::vector<float>& ts,
              const std::vector<CameraPose>& poses,
              InterpolationType interp = LINEAR, bool repeat = false)
      : _ts(ts),
        _poses(poses),
        _interp_type(interp),
        _repeat(repeat),
        _active(false) {
    check_and_init();
    compute_splines();
  }

  ~CameraDolly() {
    delete _look_at_x;
    delete _look_at_y;
    delete _look_at_z;
    delete _phi;
    delete _theta;
    delete _d;
  }

  // actions
  void start() {
    _active = true;
    _timer = vltools::getTime();
    _current_time =
        _start_time;  // this shouldn't be necessary, but just in case
  }
  void stop() { _active = false; }

  // dolly states
  void getTimeAndPose(float& t, CameraPose& p) {
    step();
    t = _current_time;
    p = getPose(t);
  }

  float getTime() {
    step();
    return _current_time;
  }
  CameraPose getPose() {
    step();
    return getPose(_current_time);
  }
  CameraPose getPose(float t) const {
    CameraPose pose;
    QVector3D look_at(_look_at_x->eval(t), _look_at_y->eval(t),
                      _look_at_z->eval(t));
    pose.setLookAt(look_at);
    pose.setPhi(_phi->eval(t));
    pose.setTheta(_theta->eval(t));
    pose.setD(_d->eval(t));
    return pose;
  }
  bool done() const { return !_active; }

  // getters
  const std::vector<float>& ts() const { return _ts; }
  const std::vector<CameraPose>& poses() const { return _poses; }
  float startTime() const { return _start_time; }
  float endTime() const { return _end_time; }

  // setters
  void setStartTime(float t) {
    _start_time = qMin(qMax(t, _ts.front()), _end_time);
  }
  void setEndTime(float t) {
    _end_time = qMin(qMax(t, _start_time), _ts.back());
  }
  void setInterpType(InterpolationType type) {
    if (type != _interp_type) {
      delete _look_at_x;
      delete _look_at_y;
      delete _look_at_z;
      delete _phi;
      delete _theta;
      delete _d;
      _interp_type = type;
      compute_splines();
    }
  }
  void setRepeat(bool b) { _repeat = b; }

 private:
  void check_and_init() {
    if (_ts.size() == _poses.size() && !_ts.empty()) {
      _start_time = _ts.front();
      _end_time = _ts.back();
      _current_time = _start_time;
    } else {
      _ts.clear();
      _ts.push_back(0.0);
      _poses.clear();
      _poses.push_back(CameraPose());
      _start_time = 0.0;
      _end_time = 0.0;
      _current_time = _start_time;
    }
  }

  void step() {
    // modifies _current_time and _active
    if (_active) {
      float elapsed = (float)(vltools::getTime() - _timer);
      if (_repeat)
        _current_time = fmod(elapsed, _end_time - _start_time) + _start_time;
      else {
        _current_time = elapsed + _start_time;
        if (_current_time >= _end_time) {
          _current_time = _end_time;
          _active = false;
        }
      }
    }
  }

  void compute_splines() {
    if (_interp_type == LINEAR) {
      interpolate_linear();
    } else if (_interp_type == CUBIC_NATURAL) {
      interpolate_cubic(CubicSpline<float>::NATURAL);
    } else if (_interp_type == CUBIC_PERIODIC) {
      interpolate_cubic(CubicSpline<float>::PERIODIC);
    } else {
      // _interp_type == CONSTANT
      interpolate_const();
    }
  }

  void interpolate_const() {
    CameraPosesSOA posesSOA(_poses);
    _look_at_x = new ConstantSpline<float>(_ts, posesSOA.x);
    _look_at_y = new ConstantSpline<float>(_ts, posesSOA.y);
    _look_at_z = new ConstantSpline<float>(_ts, posesSOA.z);
    _phi = new ConstantSpline<float>(_ts, posesSOA.phi);
    _theta = new ConstantSpline<float>(_ts, posesSOA.theta);
    _d = new ConstantSpline<float>(_ts, posesSOA.d);
  }

  void interpolate_linear() {
    CameraPosesSOA posesSOA(_poses);
    _look_at_x = new LinearSpline<float>(_ts, posesSOA.x);
    _look_at_y = new LinearSpline<float>(_ts, posesSOA.y);
    _look_at_z = new LinearSpline<float>(_ts, posesSOA.z);
    _phi = new LinearSpline<float>(_ts, posesSOA.phi);
    _theta = new LinearSpline<float>(_ts, posesSOA.theta);
    _d = new LinearSpline<float>(_ts, posesSOA.d);
  }

  void interpolate_cubic(CubicSpline<float>::BoundaryBehavior b) {
    CameraPosesSOA posesSOA(_poses);
    _look_at_x = new CubicSpline<float>(_ts, posesSOA.x, b);
    _look_at_y = new CubicSpline<float>(_ts, posesSOA.y, b);
    _look_at_z = new CubicSpline<float>(_ts, posesSOA.z, b);
    _phi = new CubicSpline<float>(_ts, posesSOA.phi, b);
    _theta = new CubicSpline<float>(_ts, posesSOA.theta, b);
    _d = new CubicSpline<float>(_ts, posesSOA.d, b);
  }

  std::vector<float> _ts;
  std::vector<CameraPose> _poses;

  Spline<float>* _look_at_x;
  Spline<float>* _look_at_y;
  Spline<float>* _look_at_z;
  Spline<float>* _phi;
  Spline<float>* _theta;
  Spline<float>* _d;

  float _start_time;
  float _end_time;
  float _current_time;
  double _timer;

  InterpolationType _interp_type;
  bool _repeat;
  bool _active;
};

#endif  // __CAMERADOLLY_H__
