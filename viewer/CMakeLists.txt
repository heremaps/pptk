cmake_minimum_required(VERSION 2.8.11)
project(viewer)

set(EIGEN_INC "" CACHE PATH "Eigen include folder")
set(QMAKE_BIN "" CACHE FILEPATH "qmake executable")

if (EIGEN_INC STREQUAL "")
	message(FATAL_ERROR "EIGEN_INC not set")
endif()
if (QMAKE_BIN STREQUAL "")
	message(FATAL ERROR "QMAKE_BIN not set")
endif()

set(QMAKE_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/qmake-build)
file(MAKE_DIRECTORY ${QMAKE_BUILD_DIR})
add_custom_target(viewer
	ALL ${QMAKE_BIN} ${CMAKE_CURRENT_SOURCE_DIR} "EIGEN_INC = ${EIGEN_INC}"
	COMMAND nmake install
	WORKING_DIRECTORY ${QMAKE_BUILD_DIR})