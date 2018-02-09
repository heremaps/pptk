copy_file(${Qt5_DIR}/plugins/platforms/libqcocoa.dylib _target_name)
set(_target_file ${CMAKE_CURRENT_BINARY_DIR}/libqcocoa.dylib)
add_custom_command(TARGET ${_target_name} POST_BUILD
  COMMAND ${CMAKE_COMMAND} -P
    ${PROJECT_SOURCE_DIR}/cmake/CopyAppleDependencies.cmake 
    ${_target_file} ${PYPCL_LIBS_DIR})

