
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

function(copy_target_dependencies x)
  if(ARGC GREATER 1)
    set(_target_file ${ARGV1})
    get_filename_component(_target_file_name ${_target_file} NAME)
  else()
    set(_target_file $<TARGET_FILE:${x}>)
    set(_target_file_name $<TARGET_FILE_NAME:${x}>)
  endif()
  if(WIN32)
    add_custom_command(TARGET ${x} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -P
        ${PROJECT_SOURCE_DIR}/cmake/CopyWindowsDependencies.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/${_target_file_name}
        ${PPTK_LIBS_DIR} "${PPTK_DLL_DIRS}")
  elseif(APPLE)
    add_custom_command(TARGET ${x} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -P
        ${PROJECT_SOURCE_DIR}/cmake/CopyAppleDependencies.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/${_target_file_name} ${PPTK_LIBS_DIR})
  elseif(UNIX)
    add_custom_command(TARGET ${x} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -P
        ${PROJECT_SOURCE_DIR}/cmake/CopyLinuxDependencies.cmake
        ${_target_file}
        ${CMAKE_CURRENT_BINARY_DIR}/${_target_file_name}
        ${PPTK_LIBS_DIR} ${PPTK_PATCHELF})
  endif()
endfunction()

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
  if (ARGC EQUAL 2)
    set(${ARGV1} ${name} PARENT_SCOPE)
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

function(copy_file_with_dependencies x)
  copy_file(${x} _target_name)
  copy_target_dependencies(${_target_name} ${x})
endfunction()
