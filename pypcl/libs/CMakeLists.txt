# install system runtime libraries
if (WIN32)
  # for Windows, this copies msvcr*.dll, msvcp*.dll, msvcomp*.dll
  foreach(f ${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS})
    copy_file(${f})
  endforeach()
elseif (APPLE)
elseif (UNIX)
  # assume Linux system already has libstdc++.so.6 and libgcc_s.so.1
  copy_file(/usr/lib64/libgomp.so.1)
endif()

# copy Qt runtime libraries
macro(copy_import_target x)
  get_target_property(temp ${x} LOCATION)
  copy_file(${temp})
  unset(temp)
endmacro()
if (UNIX AND NOT APPLE)
  copy_import_target(Qt5::Network)
  copy_import_target(Qt5::Core)
  copy_import_target(Qt5::Gui)
  copy_import_target(Qt5::OpenGL)
  copy_import_target(Qt5::Widgets)
endif()

add_subdirectory(qt_plugins)
