if (WIN32)
  copy_file(${Qt5_DIR}/plugins/platforms/qwindows.dll _target_name)
  add_custom_command(TARGET ${_target_name} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -P
      ${PROJECT_SOURCE_DIR}/cmake/CopyWindowsDependencies.cmake
      ${CMAKE_CURRENT_BINARY_DIR}/qwindows.dll
      ${PYPCL_LIBS_DIR} "${PYPCL_DLL_DIRS}")
elseif (APPLE)
  copy_file(${Qt5_DIR}/plugins/platforms/libqcocoa.dylib _target_name)
  add_custom_command(TARGET ${_target_name} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -P
      ${PROJECT_SOURCE_DIR}/cmake/CopyAppleDependencies.cmake
      ${CMAKE_CURRENT_BINARY_DIR}/libqcocoa.dylib
      ${PYPCL_LIBS_DIR})
elseif (UNIX)
  copy_file(${Qt5_DIR}/plugins/platforms/libqxcb.so)
endif()

