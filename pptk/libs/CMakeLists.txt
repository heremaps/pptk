function(copy_system_file x)
  # creates local copy of x in CMAKE_CURRENT_BINARY_DIR
  # optional second argument is a list of search paths for x
  if(ARGC EQUAL 2)
    set(y ${ARGV1})
  endif()
  find_file(temp ${x} ${y})
  copy_file(${temp} _target)
  if(UNIX AND NOT APPLE)
    add_custom_command(TARGET ${_target} POST_BUILD COMMAND
      ${PPTK_PATCHELF} --set-rpath \\\$\$ORIGIN
      ${CMAKE_CURRENT_BINARY_DIR}/${x})
  endif()
  unset(temp CACHE)
endfunction()

function(copy_import_target x)
  get_target_property(_file_path ${x} LOCATION)
  get_filename_component(_file_name ${_file_path} NAME)
  copy_file(${_file_path} _target)
  if(UNIX AND NOT APPLE)
    add_custom_command(TARGET ${_target} POST_BUILD COMMAND
      ${PPTK_PATCHELF} --set-rpath \\\$\$ORIGIN
      ${CMAKE_CURRENT_BINARY_DIR}/${_file_name})
  endif()
endfunction()

# install system runtime libraries
if (WIN32)
  # for Windows, this copies msvcr*.dll, msvcp*.dll, msvcomp*.dll
  foreach(f ${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS})
    copy_file(${f})
  endforeach()
elseif (APPLE)
elseif (UNIX)
  # create local copies of system libraries that are required by pptk targets
  # but not provided by a bare bone manylinux1 platform, as specified in PEP 513
  if (CMAKE_LIBRARY_ARCHITECTURE)
    set(_paths
      /lib/${CMAKE_LIBRARY_ARCHITECTURE}
      /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE})
  endif()
  copy_system_file(libgomp.so.1 "${_paths}")
  copy_system_file(libz.so.1 "${_paths}")
endif()

add_subdirectory(qt_plugins)
