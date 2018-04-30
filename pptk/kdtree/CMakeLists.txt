# # Uncomment the following lines to run as standalone cmake script
# cmake_minimum_required(VERSION 3.0)
# project(kdtree)

# set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake" ${CMAKE_CURRENT_SOURCE_DIR})
# find_package(PythonLibs 2.7 REQUIRED)
# find_package(Numpy REQUIRED)
# find_package(TBB REQUIRED)
# include(UsefulMacros)

add_library(kdtree SHARED kdtree_wrapper.cpp)
set_target_python_module_name(kdtree)
if (WIN32)
    target_compile_options(kdtree PRIVATE /bigobj)
    target_link_libraries(kdtree ${PYTHON_LIBRARY})
elseif(APPLE)
  set_target_properties(kdtree PROPERTIES
    LINK_FLAGS "-undefined dynamic_lookup")
elseif(UNIX)
  set_target_properties(kdtree PROPERTIES
    LINK_FLAGS "-static-libstdc++")
endif()
target_compile_definitions(kdtree PRIVATE -DUSE_TBB)
target_include_directories(kdtree PRIVATE
  ../include  # for python_util.h
  src         # for k-d tree source code
  ${TBB_INCLUDE_DIR}
  ${PYTHON_INCLUDE_DIR}
  ${Numpy_INCLUDE_DIR} )
target_link_libraries(kdtree
  ${TBB_tbb_LIBRARY}
  ${TBB_tbbmalloc_LIBRARY})
copy_target(kdtree)
copy_target_dependencies(kdtree)
copy_file(__init__.py)

