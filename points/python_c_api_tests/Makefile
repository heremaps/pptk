PYTHON_ROOT = /usr
INC = 	$(PYTHON_ROOT)/include/python2.7 \
	$(PYTHON_ROOT)/lib/python2.7/dist-packages/numpy/core/include/numpy \
   /usr/include/eigen3/
#	/home/victlu/Sources/eigen-3.2.1

LIB = $(PYTHON_ROOT)/lib/python2.7/config-x86_64-linux-gnu
CXX = g++
CXX_OPTIONS = -O2 -fPIC -fopenmp

INC_OPTIONS = $(patsubst %,-I%,$(INC))
LIB_OPTIONS = $(patsubst %,-L%,$(LIB))

all: foo.so vfuncs.so

foo.so: foo.cpp
	$(CXX) $(CXX_OPTIONS) -o $@ $^ \
		$(INC_OPTIONS) $(LIB_OPTIONS) -lpython2.7 -shared

vfuncs.so: vfuncs.cpp
	$(CXX) $(CXX_OPTIONS) -o $@ $^ \
		$(INC_OPTIONS) $(LIB_OPTIONS) -lpython2.7 -shared \

clean:
	rm -rf foo.so vfuncs.so
