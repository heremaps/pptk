# for Windows, this copies msvcr*.dll, msvcp*.dll, msvcomp*.dll
# TODO: find out what this includes for Linux and Mac
foreach(f ${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS})
  copy_file(${f})
endforeach()

# copy TBB runtime libraries
copy_file(${TBB_tbb_RUNTIME})
copy_file(${TBB_tbbmalloc_RUNTIME})
