cmake_minimum_required(VERSION 2.8.11)
project(viewer)

set(EIGEN_INC "EIGEN_INC-NOTFOUND" CACHE PATH "Eigen include folder")
find_program(QMAKE_BIN qmake DOC "qmake executable")

if (EIGEN_INC STREQUAL "EIGEN_INC-NOTFOUND")
	message(FATAL_ERROR "EIGEN_INC not set")
endif()
if (QMAKE_BIN STREQUAL "QMAKE_BIN-NOTFOUND")
	message(FATAL_ERROR "QMAKE_BIN not set")
endif()
if (WIN32 AND MSVC)
	set(MAKE_CMD nmake)
else()
	set(MAKE_CMD make)
endif()

set(QMAKE_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/qmake-build)
file(MAKE_DIRECTORY ${QMAKE_BUILD_DIR})
add_custom_target(viewer
	ALL ${QMAKE_BIN} ${CMAKE_CURRENT_SOURCE_DIR} "EIGEN_INC = ${EIGEN_INC}"
	COMMAND ${MAKE_CMD} install
	WORKING_DIRECTORY ${QMAKE_BUILD_DIR})
