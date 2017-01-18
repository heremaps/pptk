#ifndef __RESULTSET_H__
#define __RESULTSET_H__
#include <vector>
// following code copied from flann result_set.h, mod cosmetic changes

namespace pointkd {

template <typename DistanceType>
struct DistanceIndex {
	DistanceIndex(DistanceType dist, size_t index) :
		dist_(dist), index_(index) {}
	bool operator<(const DistanceIndex& dist_index) const {
		return (dist_ < dist_index.dist_) ||
			((dist_ == dist_index.dist_) && index_ < dist_index.index_);
	}
	DistanceType dist_;
	size_t index_;
};

template <typename DistanceType>
class ResultSet {
public:
	virtual void clear() = 0;
	virtual DistanceType worstDist() const = 0; 
	virtual void copy(DistanceType * dists, int * indices) = 0;
	virtual void addPoint(DistanceType dist, std::size_t index) = 0;
	virtual bool full() const = 0;
};

template <typename DistanceType>
class ResultSetLinearInsert : public ResultSet<DistanceType> {
public:
	typedef DistanceIndex<DistanceType> DistIndex;
	ResultSetLinearInsert(
		std::size_t capacity, DistanceType initialMaxDist = 
			std::numeric_limits<DistanceType>::max()): 
		capacity_(capacity), initialMaxDist_(initialMaxDist) {
		dist_index_.resize(capacity_,
			DistIndex(initialMaxDist_,-1));
    	clear();
	}
	bool full() const {
		return count_ == capacity_;
	}
	void clear() {
		worst_distance_ = initialMaxDist_;
		dist_index_[capacity_-1].dist_ = worst_distance_;
		count_ = 0;
	}
	DistanceType worstDist() const {
		return worst_distance_;
	}
	void copy(DistanceType * dists, int * indices) {
		for (std::size_t i = 0; i < count_; i++) {
			dists[i] = dist_index_[i].dist_;
			indices[i] = (int)dist_index_[i].index_;
		}
	}
	void addPoint(DistanceType dist, size_t index) {
		if (dist>=worst_distance_) return;

		if (count_ < capacity_) ++count_;
		size_t i;
		for (i=count_-1; i>0; --i) {
			if (dist_index_[i-1].dist_>dist)
			{
				dist_index_[i] = dist_index_[i-1];
			}
			else break;
		}
		dist_index_[i].dist_ = dist;
		dist_index_[i].index_ = index;
		worst_distance_ = dist_index_[capacity_-1].dist_;
	}
private:
	std::vector<DistIndex> dist_index_;
	std::size_t count_;
	DistanceType worst_distance_;
	std::size_t capacity_;
	DistanceType initialMaxDist_;
};

template <typename DistanceType>
class ResultSetHeap : public ResultSet<DistanceType> {
public:
	typedef DistanceIndex<DistanceType> DistIndex;
	ResultSetHeap(
		std::size_t capacity, DistanceType initialMaxDist =
			std::numeric_limits<DistanceType>::max()): 
		capacity_(capacity), initialMaxDist_(initialMaxDist) {
		dist_index_.reserve(capacity_);
		clear();
	}
	bool full() const {
		return dist_index_.size() == capacity_;
	}
	void clear() {
		dist_index_.clear();
		worst_dist_ = initialMaxDist_;
		is_full_ = false;
	}
	DistanceType worstDist() const {
		return worst_dist_;
	}
	void copy(DistanceType * dists, int * indices) {
		std::sort(dist_index_.begin(), dist_index_.end());
		for (std::size_t i = 0; i < dist_index_.size(); i++) {
			dists[i] = dist_index_[i].dist_;
			indices[i] = (int)dist_index_[i].index_;
		}
	}
	void addPoint(DistanceType dist, std::size_t index) {
		if (dist>=worst_dist_) return;

    	if (dist_index_.size()==capacity_) {
    		// if result set if filled to capacity, remove farthest element
    		std::pop_heap(dist_index_.begin(), dist_index_.end());
        	dist_index_.pop_back();
    	}

    	// add new element
    	dist_index_.push_back(DistIndex(dist,index));
    	if (is_full_) { // when is_full_==true, we have a heap
    		std::push_heap(dist_index_.begin(), dist_index_.end());
    	}

    	if (dist_index_.size()==capacity_) {
    		if (!is_full_) {
    			std::make_heap(dist_index_.begin(), dist_index_.end());
            	is_full_ = true;
    		}
    		// we replaced the farthest element, update worst distance
        	worst_dist_ = dist_index_[0].dist_;
		}
	}
private:
	std::size_t capacity_;
	DistanceType initialMaxDist_;
	DistanceType worst_dist_;
	std::vector<DistIndex> dist_index_;
	bool is_full_;
};

}	// namespace pointkd

#endif	// #ifndef __RESULTSET_H__
