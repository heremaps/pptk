#ifndef __PYPCL_DLL_EXPORT_H__
#define __PYPCL_DLL_EXPORT_H__

#if defined(_WIN32) || defined(_WIN64)
  #ifdef __cplusplus
    #define PYPCL_DLL_EXPORT extern "C" __declspec(dllexport)
  #else
    #define PYPCL_DLL_EXPORT __declspec(dllexport)
  #endif
#else
  #define PYPCL_DLL_EXPORT
#endif

#endif // __PYPCL_DLL_EXPORT_H__
