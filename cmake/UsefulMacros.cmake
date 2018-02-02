
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

macro(copy_file x)
  # x should be specified relative to current source directory
  get_filename_component(y ${x} NAME)
  set(src ${CMAKE_CURRENT_SOURCE_DIR}/${x})
  set(dst ${CMAKE_CURRENT_BINARY_DIR})
  add_custom_target(${y} ALL
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${src} ${dst}
    COMMENT "Copying ${src} to ${dst}")
  unset(y)
  unset(src)
  unset(dst)
endmacro()