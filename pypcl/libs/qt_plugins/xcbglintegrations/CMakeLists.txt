if(UNIX AND NOT APPLE)
  copy_file_with_dependencies(
    ${Qt5_PLUGINS_DIR}/xcbglintegrations/libqxcb-egl-integration.so)
  copy_file_with_dependencies(
    ${Qt5_PLUGINS_DIR}/xcbglintegrations/libqxcb-glx-integration.so)
endif()
