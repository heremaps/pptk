#ifndef __PROGRESS_BAR_H__
#define __PROGRESS_BAR_H__

#include <algorithm>
#include <sstream>
#include <string>

class ProgressWheel {
 public:
  ProgressWheel() : c_("\\|/-"), counter_(0) {}
  void tick() { counter_ = (counter_ + 1) % 4; }
  char get_char() const { return c_[counter_]; }

 private:
  const std::string c_;
  int counter_;
};

template <typename T>
class ProgressBar {
 public:
  ProgressBar(T num_iters) : len_(20), curr_iter_(0), num_iters_(num_iters) {}
  ProgressBar(T num_iters, int len)
      : len_(len), curr_iter_(0), num_iters_(num_iters) {}

  void update(T curr_iter) {
    wheel_.tick();
    curr_iter_ = curr_iter;
  }

  std::string get_string() const {
    float p = (float)curr_iter_ / num_iters_;
    if (num_iters_ == 0) p = 1.0f;
    int n = std::max(0, std::min(len_, (int)(p * len_)));
    if (curr_iter_ == num_iters_) n = len_;
    std::stringstream ss;
    ss << (int)(p * 100);
    return "[" + std::string(n, '=')
               + std::string(len_ - n, '-')
               + wheel_.get_char() + "] " + ss.str() + "%";
  }

 private:
  ProgressWheel wheel_;
  int len_;  // progress bar length (# char)
  T curr_iter_;
  T num_iters_;
};

#endif  // __PROGRESS_BAR_H__