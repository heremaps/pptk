cmake_minimum_required(VERSION 3.1)
project(pypcl)

set(CMAKE_MODULE_PATH
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_CURRENT_SOURCE_DIR})

find_package(PythonLibs 2.7 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Numpy REQUIRED)
find_package(TBB REQUIRED)
find_package(Eigen REQUIRED)
find_package(Qt5 CONFIG REQUIRED COMPONENTS Widgets Network OpenGL Core)
find_package(OpenMP)

# get root Qt5 folder (i.e. contains bin, lib, plugins, etc.)
get_target_property(Qt5_DIR Qt5::qmake LOCATION)
get_filename_component(Qt5_DIR ${Qt5_DIR} DIRECTORY)
get_filename_component(Qt5_DIR ${Qt5_DIR} DIRECTORY)

# localize all dependencies (.dll, .so, or .dylib) in the following folder
set(PYPCL_LIBS_DIR ${PROJECT_BINARY_DIR}/pypcl/libs)

# use the following variable to store a list .dll paths required by targets
# built in pypcl; this is useful only for building on Windows platform
get_filename_component(TBB_RUNTIME_DIR ${TBB_tbb_RUNTIME} DIRECTORY)
set(PYPCL_DLL_DIRS
  ${TBB_RUNTIME_DIR}
  ${Qt5_DIR}/bin
  CACHE INTERNAL "Additional folder paths for finding .dll's")

include(UsefulMacros)

# following lines specifies the CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS variable
set(CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS_SKIP TRUE)
set(CMAKE_INSTALL_OPENMP_LIBRARIES TRUE)  # requires cmake 3.1
include(InstallRequiredSystemLibraries)

copy_file(setup.py)
add_subdirectory(pypcl)
