# usage: cmake -P CopyWindowsDependencies.cmake
#   <target file path> <copy folder path> <.dll folder paths>
# assumes full existing paths
# .dll folder paths are semicolon-separated

include(GetPrerequisites)

set(_target_file ${CMAKE_ARGV3})
set(_copy_folder ${CMAKE_ARGV4})
set(_dll_paths ${CMAKE_ARGV5})
get_prerequisites(${_target_file} _prereqs 1 1 "" "${_dll_paths}")
foreach(p ${_prereqs})
  if(NOT (p MATCHES "python[0-9]*.dll"))
    gp_resolve_item("" ${p} "" "${_dll_paths}" src)
    set(dst ${_copy_folder})
    execute_process(COMMAND
      ${CMAKE_COMMAND} -E copy_if_different ${src} ${dst})
  endif()
endforeach()