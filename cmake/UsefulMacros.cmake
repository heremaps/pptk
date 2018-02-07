
macro(set_target_python_module_name target)
  set_target_properties(${target} PROPERTIES PREFIX "")
  if (WIN32)
    set_target_properties(${target} PROPERTIES SUFFIX ".pyd")
  elseif(APPLE)
    set_target_properties(${target} PROPERTIES SUFFIX ".so")
  endif (WIN32)
endmacro()

macro(set_target_rpath target path)
  if (APPLE)
    set_target_properties(${target} PROPERTIES INSTALL_RPATH "@loader_path/${path}" BUILD_WITH_INSTALL_RPATH TRUE)
  elseif (UNIX)
    set_target_properties(${target} PROPERTIES INSTALL_RPATH "$ORIGIN/${path}" BUILD_WITH_INSTALL_RPATH TRUE)
  endif(APPLE)
endmacro()

macro(copy_target x)
  set(src $<TARGET_FILE:${x}>)
  set(dst ${CMAKE_CURRENT_BINARY_DIR})
  add_custom_command(
    TARGET ${x}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${src} ${dst}
    COMMENT "Copying ${src} to ${dst}")
  unset(src)
  unset(dst)
endmacro()

macro(current_source_dir x)
  string(CONCAT ${x} "^" ${PROJECT_SOURCE_DIR} "/?")
  string(REGEX REPLACE ${${x}} "" ${x} ${CMAKE_CURRENT_SOURCE_DIR})
endmacro()

function(copy_file x)
  # x should be a file path, and should not be a variable
  # i.e. copy_file(${var}), not copy_file(var)
  get_filename_component(name ${x} NAME)
  file(RELATIVE_PATH temp ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
  if (NOT (temp STREQUAL ""))
    string(REGEX REPLACE "(/|\\\\)" "." temp "${temp}")
    string(CONCAT name "${temp}" "." "${name}")
  endif()
  if (NOT IS_ABSOLUTE ${x})
    set(src ${CMAKE_CURRENT_SOURCE_DIR}/${x})
  else()
    set(src ${x})
  endif()
  set(dst ${CMAKE_CURRENT_BINARY_DIR})
  add_custom_target(${name} ALL
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${src} ${dst}
    COMMENT "Copying ${src} to ${dst}")
endfunction()
