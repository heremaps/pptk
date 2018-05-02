# pptk - Point Processing Toolkit

Copyright (c) 2011-2017 HERE Europe B.V.

pptk is a Python package for facilitating point cloud processing in Python.

## License

Unless otherwise noted in `LICENSE` files for specific files or directories,
the [LICENSE](LICENSE) in the root applies to all content in this repository.

## Install

One can either install pptk directly from PyPI

```
>> pip install pptk
```

or from the .whl file that results from [building pptk from source](#build).

```
>> pip install <.whl file>
```

## Build

We provide CMake scripts for automating most of the build process, but ask the
user to manually prepare [dependencies](#requirements) and record their paths
in the following CMake cache variables.

* `Numpy_INCLUDE_DIR`
* `PYTHON_INCLUDE_DIR`
* `PYTHON_LIBRARY`
* `Eigen_INCLUDE_DIR`
* `TBB_INCLUDE_DIR`
* `TBB_tbb_LIBRARY`
* `TBB_tbb_RUNTIME`
* `TBB_tbbmalloc_LIBRARY`
* `TBB_tbbmalloc_RUNTIME`
* `Qt5_DIR`

To set these variables, either use one of CMake's GUIs (ccmake or cmake-gui),
or provide an initial CMakeCache.txt in the target build folder
(for examples of initial cache files, see the CMakeCache.<platform>.txt files)

##### Requirements

Listed are versions of libraries used to develop pptk, though earlier versions
of these libraries may also work.

* [QT](https://www.qt.io/) 5.4
* [TBB](https://www.threadingbuildingblocks.org/) 4.3
* [Eigen](http://eigen.tuxfamily.org) 3.2.9
* [Python](https://www.python.org/) 2.7+ or 3.6+
* [Numpy](http://www.numpy.org/) 1.13

##### Windows

1. Create an empty build folder

```
>> mkdir <build_folder>
```

2. Create an initial CMakeCache.txt under <build_folder>, specifying in it
library paths on the build platform (see CMakeFiles.win.txt for an example)

3. Type the following...

```
>> cd <build_folder>
>> cmake -G "NMake Makefiles" <source_folder>
>> nmake
>> python setup.py bdist_wheel
>> pip install dist\<.whl file>
```

##### Linux

Similar to building on Windows.

##### Mac

Similar to building on Windows.