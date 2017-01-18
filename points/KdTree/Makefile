PYTHON_ROOT = /usr
TBB_ROOT = /home/victlu/Sources/tbb42_20140601oss
INC = Code UsefulHeaders \
	$(PYTHON_ROOT)/include/python2.7 \
	$(PYTHON_ROOT)/lib/python2.7/dist-packages/numpy/core/include/numpy \
	$(TBB_ROOT)/include
LIB = $(PYTHON_ROOT)/lib/python2.7/config-x86_64-linux-gnu \
	$(TBB_ROOT)/lib/intel64/gcc4.4
CXX = g++
CXX_OPTIONS = -O2 -fPIC

INC_OPTIONS = $(patsubst %,-I%,$(INC))
LIB_OPTIONS = $(patsubst %,-L%,$(LIB))

all: kdtree.so

kdtree.so: kdtree_wrapper.cpp
	$(CXX) $(CXX_OPTIONS) -o $@ $^ \
		$(INC_OPTIONS) $(LIB_OPTIONS) -lpython2.7 -shared -ltbbmalloc -ltbb

clean:
	rm -f kdtree.so
