include(FindPackageHandleStandardArgs)

set(Numpy_INCLUDE_DIR "Numpy_INCLUDE_DIR-NOTFOUND" CACHE PATH "Path of folder containing arrayobject.h")

find_package_handle_standard_args(Numpy REQUIRED_VARS Numpy_INCLUDE_DIR)
