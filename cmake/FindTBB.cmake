include(FindPackageHandleStandardArgs)

set(TBB_INCLUDE_DIR "TBB_INCLUDE_DIR-NOTFOUND" CACHE PATH "Path to directory containing TBB header files")
set(TBB_tbb_LIBRARY "TBB_tbb_LIBRARY-NOTFOUND" CACHE FILEPATH "Path to tbb link library (i.e. tbb.lib)")
set(TBB_tbbmalloc_LIBRARY "TBB_tbbmalloc_LIBRARY-NOTFOUND" CACHE FILEPATH "Path to tbbmalloc link library (i.e. tbbmalloc.lib)")
set(TBB_tbb_RUNTIME "TBB_tbb_RUNTIME-NOTFOUND" CACHE FILEPATH "Path to tbb runtime library (i.e. tbb.dll)")
set(TBB_tbbmalloc_RUNTIME "TBB_tbbmalloc_RUNTIME-NOTFOUND" CACHE FILEPATH "Path to tbbmalloc runtime library (i.e. tbbmalloc.dll)")

find_package_handle_standard_args(TBB
  REQUIRED_VARS
    TBB_INCLUDE_DIR
    TBB_tbb_LIBRARY
    TBB_tbbmalloc_LIBRARY
    TBB_tbb_RUNTIME
    TBB_tbbmalloc_RUNTIME
)
