cmake_minimum_required(VERSION 3.1)
project(pypcl)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_CURRENT_SOURCE_DIR})

find_package(PythonLibs 2.7 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Numpy REQUIRED)
find_package(TBB REQUIRED)
find_package(Eigen REQUIRED)
find_package(Qt5 CONFIG REQUIRED COMPONENTS Widgets Network OpenGL Core)
find_package(OpenMP)

include(UsefulMacros)

# following lines specifies the CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS variable
set(CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS_SKIP TRUE)
set(CMAKE_INSTALL_OPENMP_LIBRARIES TRUE)  # requires cmake 3.1
include(InstallRequiredSystemLibraries)

copy_file(setup.py)
add_subdirectory(pypcl)
