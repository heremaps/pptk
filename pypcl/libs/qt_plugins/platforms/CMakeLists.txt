if (WIN32)
  copy_file(${Qt5_DIR}/plugins/platforms/qwindows.dll)
elseif (APPLE)
  include(${CMAKE_CURRENT_SOURCE_DIR}/mac.cmake)
elseif (UNIX)
  copy_file(${Qt5_DIR}/plugins/platforms/libqxcb.so)
endif()

