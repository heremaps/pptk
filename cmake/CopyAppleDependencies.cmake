# usage: cmake -P CopyAppleDependencies.cmake 
#   <target file path> <copy folder path>
# paths assumed to be existing full paths

include(BundleUtilities)

find_program(_install_name_tool "install_name_tool")
set(_target ${CMAKE_ARGV3})
set(_copy_folder ${CMAKE_ARGV4})
set(_paths "/usr/bin")
get_item_rpaths(${_target} _rpaths)
get_prerequisites(${_target} _prereqs 1 1 "" "${_paths}" "${_rpaths}")

# delete existing rpaths in _target and
# add relative path to _copy_folder as new rpath
foreach(p ${_rpaths})
  execute_process(COMMAND ${_install_name_tool} -delete_rpath ${p} ${_target})
endforeach()
get_filename_component(_target_folder ${_target} DIRECTORY)
file(RELATIVE_PATH _new_rpath ${_target_folder} ${_copy_folder})
execute_process(COMMAND
  ${_install_name_tool} -add_rpath "@loader_path/${_new_rpath}" ${_target})

# copy _target's dependencies to _copy_folder
foreach(p ${_prereqs})
  get_filename_component(y ${p} NAME)
  set(dst ${_copy_folder}/${y})
  gp_resolve_item(${_target} ${p} "" "${_paths}" src "${_rpaths}")
  execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different ${src} ${dst})
  execute_process(COMMAND ${_install_name_tool} -id @rpath/${y} ${dst})
  if (IS_ABSOLUTE ${p})
    execute_process(COMMAND ${_install_name_tool} -change ${p} @rpath/${y} ${_target})
  endif()
  get_prerequisites(${dst} _prereqs_of_prereqs 1 0 "" "")
  foreach (pp ${_prereqs_of_prereqs})
    get_filename_component(yy ${pp} NAME)
    if (IS_ABSOLUTE ${pp})
      execute_process(COMMAND ${_install_name_tool} -change ${pp} @rpath/${yy} ${dst})
    endif()
  endforeach()
endforeach()
