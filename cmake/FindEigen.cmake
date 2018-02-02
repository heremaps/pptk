include(FindPackageHandleStandardArgs)

set(Eigen_INCLUDE_DIR "Eigen_INCLUDE_DIR-NOTFOUND" CACHE PATH "Path containing Eigen folder")

find_package_handle_standard_args(Eigen REQUIRED_VARS Eigen_INCLUDE_DIR)
