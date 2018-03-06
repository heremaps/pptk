#ifndef __SPLINES_H__
#define __SPLINES_H__
#include "Eigen/Dense"
#include "Eigen/Sparse"

template <typename T>
class Spline {
 public:
  Spline(const std::vector<T>& ts, const std::vector<T>& vs)
      : _ts(ts), _vs(vs) {
    checkAndInit();
  }

  virtual ~Spline() = 0;

  virtual T eval(T t) const = 0;

  const std::vector<T>& ts() const { return _ts; }
  const std::vector<T>& vs() const { return _vs; }

 protected:
  static bool checkTs(std::vector<T>& ts) {
    // returns true if ts is:
    // 1. non-empty
    // 2. sorted in increasing order
    // 3. unique
    for (std::size_t i = 1; i < ts.size(); i++)
      if (ts[i] <= ts[i - 1]) return false;
    return !ts.empty();
  }

  void checkAndInit() {
    // checks that _ts and _vs are valid.
    // if not, make this spline into a constant zero function
    if (!checkTs(_ts) || _ts.size() != _vs.size()) {
      _ts.clear();
      _ts.push_back(0.0f);
      _vs.clear();
      _vs.push_back(0.0f);
    }
  }

  std::vector<T> _ts;
  std::vector<T> _vs;
};

template <typename T>
inline Spline<T>::~Spline() {}

template <typename T>
class ConstantSpline : public Spline<T> {
  using Spline<T>::_ts;
  using Spline<T>::_vs;

 public:
  ConstantSpline(const std::vector<T>& ts, const std::vector<T>& vs)
      : Spline<T>(ts, vs) {}

  ~ConstantSpline() {}

  T eval(T t) const {
    int idx = std::upper_bound(_ts.begin(), _ts.end(), t) - _ts.begin();
    if (idx == 0)
      return _vs.front();
    else if (idx == (int)_ts.size())
      return _vs.back();
    else
      // at this point, we know ts.size() > 1 and  0 < idx < _ts.size()
      return _vs[idx - 1];
  }
};

template <typename T>
class LinearSpline : public Spline<T> {
  using Spline<T>::_ts;
  using Spline<T>::_vs;

 public:
  LinearSpline(const std::vector<T>& ts, const std::vector<T>& vs)
      : Spline<T>(ts, vs) {}

  ~LinearSpline() {}

  T eval(T t) const {
    int idx = std::upper_bound(_ts.begin(), _ts.end(), t) - _ts.begin();
    if (idx == 0)
      return _vs.front();
    else if (idx == (int)_ts.size())
      return _vs.back();
    else {
      // at this point, we know ts.size() > 1 and  0 < idx < _ts.size()
      float dt = (t - _ts[idx - 1]) / (_ts[idx] - _ts[idx - 1]);
      return (1.0f - dt) * _vs[idx - 1] + dt * _vs[idx];
    }
  }
};

template <typename T>
class CubicSpline : public Spline<T> {
  using Spline<T>::_ts;
  using Spline<T>::_vs;

 public:
  typedef Eigen::Triplet<float> Triplet;
  typedef Eigen::SparseMatrix<float, Eigen::RowMajor> SpMat;
  enum BoundaryBehavior { NATURAL, PERIODIC };
  CubicSpline(const std::vector<float>& ts, const std::vector<float>& vs,
              BoundaryBehavior boundaryBehavior = NATURAL)
      : Spline<T>(ts, vs), _boundary_behavior(boundaryBehavior) {
    calculateCoefficients();
  }

  ~CubicSpline() {}

  T eval(T t) const {
    int idx = std::upper_bound(_ts.begin(), _ts.end(), t) - _ts.begin();
    if (idx == 0)
      return _vs.front();
    else if (idx == (int)_ts.size())
      return _vs.back();
    else {
      // at this point, we know ts.size() > 1 and  0 < idx < _ts.size()
      float dt = (t - _ts[idx - 1]) / (_ts[idx] - _ts[idx - 1]);
      float c[4] = {_vs[idx - 1],
                    _coeffs_1[idx - 1],
                    _coeffs_2[idx - 1],
                    _vs[idx]};
      float a[4] = {1.0f, 3.0f, 3.0f, 1.0f};
      float v = 0.0f;
      for (int i = 0; i < 4; i++)
        v += c[i] * a[i] * powf(1.0f - dt,
                                3.0f - (float)i) * powf(dt, (float)i);
      return v;
    }
  }

 private:
  void setupLinearSystem(SpMat& A, Eigen::VectorXf& b) {
    // assumes _ts.size() > 1
    int num_intervals = (int)_ts.size() - 1;
    int num_knots = num_intervals - 1;
    int n = 2 * num_intervals;

    // precompute interval lengths
    std::vector<float> delta_inv(num_intervals);
    std::vector<float> delta_inv_2(num_intervals);
    for (int i = 0; i < num_intervals; i++) {
      float temp = _ts[i + 1] - _ts[i];
      delta_inv[i] = 1.0f / temp;
      delta_inv_2[i] = delta_inv[i] * delta_inv[i];
    }

    // calculate non-zero entries of A and b
    b.resize(n);
    std::vector<Triplet> triplets;
    for (int i = 0; i < num_knots; i++) {
      // first derivative continuity
      triplets.push_back(Triplet(2 * i, 2 * i + 1, -3.0f * delta_inv[i]));
      triplets.push_back(Triplet(2 * i, 2 * (i + 1), -3.0f * delta_inv[i + 1]));
      b(2 * i) = -3.0f * _vs[i + 1] * delta_inv[i + 1] +
                 -3.0f * _vs[i + 1] * delta_inv[i];

      // second derivative continuity
      triplets.push_back(Triplet(2 * i + 1, 2 * i, 6.0f * delta_inv_2[i]));
      triplets.push_back(
          Triplet(2 * i + 1, 2 * i + 1, -12.0f * delta_inv_2[i]));
      triplets.push_back(
          Triplet(2 * i + 1, 2 * (i + 1), 12.0f * delta_inv_2[i + 1]));
      triplets.push_back(
          Triplet(2 * i + 1, 2 * (i + 1) + 1, -6.0f * delta_inv_2[i + 1]));
      b(2 * i + 1) = 6.0f * _vs[i + 1] * delta_inv_2[i + 1] -
                     6.0f * _vs[i + 1] * delta_inv_2[i];
    }
    if (_boundary_behavior == PERIODIC) {
      // first derivative continuity
      triplets.push_back(Triplet(n - 2, 0, 3.0f * delta_inv.front()));
      triplets.push_back(Triplet(n - 2, n - 1, 3.0f * delta_inv.back()));
      b(n - 2) = 3.0f * _vs.back() * delta_inv.back() +
                 3.0f * _vs.front() * delta_inv.front();

      // second derivative continuity
      triplets.push_back(Triplet(n - 1, 0, -12.0f * delta_inv_2.front()));
      triplets.push_back(Triplet(n - 1, 1, 6.0f * delta_inv_2.front()));
      triplets.push_back(Triplet(n - 1, n - 2, -6.0f * delta_inv_2.back()));
      triplets.push_back(Triplet(n - 1, n - 1, 12.0f * delta_inv_2.back()));
      b(n - 1) = 6.0f * _vs.back() * delta_inv_2.back() -
                 6.0f * _vs.front() * delta_inv_2.front();
    } else if (_boundary_behavior == NATURAL) {
      triplets.push_back(Triplet(n - 2, 0, -12.0f * delta_inv_2.front()));
      triplets.push_back(Triplet(n - 2, 1, 6.0f * delta_inv_2.front()));
      b(n - 2) = -6.0f * _vs.front() * delta_inv_2.front();

      triplets.push_back(Triplet(n - 1, n - 2, 6.0f * delta_inv_2.back()));
      triplets.push_back(Triplet(n - 1, n - 1, -12.0f * delta_inv_2.back()));
      b(n - 1) = -6.0f * _vs.back() * delta_inv_2.back();
    }
    A.resize(n, n);
    A.setFromTriplets(triplets.begin(), triplets.end());
  }

  void calculateCoefficients() {
    int num_intervals = (int)_ts.size() - 1;
    if (num_intervals > 0) {
      SpMat A;
      Eigen::VectorXf b, x;
      setupLinearSystem(A, b);
      // what's wrong with the following?
      // Eigen::SparseLU<SpMat, Eigen::COLAMDOrdering<int> > solver;
      // solver.analyzePattern(A);
      // solver.factorize(A);
      // x = solver.solve(b);
      x = Eigen::MatrixXf(A).colPivHouseholderQr().solve(b);
      _coeffs_1.resize(num_intervals);
      _coeffs_2.resize(num_intervals);
      for (int i = 0; i < num_intervals; i++) {
        _coeffs_1[i] = x(2 * i);
        _coeffs_2[i] = x(2 * i + 1);
      }
    }
  }

  std::vector<float> _coeffs_1;
  std::vector<float> _coeffs_2;
  BoundaryBehavior _boundary_behavior;
};

#endif  // __SPLINES_H__
