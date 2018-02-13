# usage: cmake -P CopyLinuxDependencies.cmake
#   <target file (orig)> <target file (copy)> <libs folder> <patchelf path>
# assumes full existing paths

include(GetPrerequisites)

set(_target_file_orig ${CMAKE_ARGV3})
set(_target_file_copy ${CMAKE_ARGV4})
set(_libs_folder ${CMAKE_ARGV5})
set(_patchelf_cmd ${CMAKE_ARGV6})

get_prerequisites(${_target_file_orig} _prereqs 1 1 "" "")
foreach(p ${_prereqs})
  gp_resolve_item("" ${p} "" "" src)
  get_filename_component(x ${src} NAME)
  set(dst ${_libs_folder}/${x})
  execute_process(COMMAND
    ${CMAKE_COMMAND} -E copy_if_different ${src} ${dst})
  execute_process(COMMAND
    ${_patchelf_cmd} --set-rpath \$ORIGIN ${dst})
endforeach()

get_filename_component(y ${_target_file_copy} DIRECTORY)
file(RELATIVE_PATH _libs_folder_r ${y} ${_libs_folder})
execute_process(COMMAND
  ${_patchelf_cmd} --set-rpath \$ORIGIN/${_libs_folder_r} ${_target_file_copy})
