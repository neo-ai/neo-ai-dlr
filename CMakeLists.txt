cmake_minimum_required (VERSION 3.6)
include(cmake/Utils.cmake)
include(3rdparty/tvm/cmake/util/FindCUDA.cmake)

set_default_configuration_release()
msvc_use_static_runtime()

# Option for Android on Arm --- has to come before project() function
option(ANDROID_BUILD "Build for Android target" OFF)

if(ANDROID_BUILD)
    set(ANDROID_SYSROOT "${NDK_ROOT}/sysroot")
    set(CMAKE_ANDROID_API 21)
    set(ANDROID_PLATFORM android-21)
    set(ANDROID_DEPRECATED_HEADERS ON)
    include_directories(${ANDROID_SYSROOT}/usr/include/arm-linux-androideabi)
endif(ANDROID_BUILD)

project(dlr)
# Options
option(USE_OPENCL  "Build with OpenCL" OFF)
option(USE_CUDA  "Build with CUDA" OFF)
option(USE_CUDNN "Build with CUDNN" OFF)
option(USE_TENSORRT "Build with Tensor RT" OFF)


# Use RPATH on Mac OS X as flexible mechanism for locating dependencies
# See https://blog.kitware.com/upcoming-in-cmake-2-8-12-osx-rpath-support/
set(CMAKE_MACOSX_RPATH TRUE)

set(python-build "bp3-python setup.py build")

# Set BUILD_SHARED_LIBS as option. By default, build shared libraries;
# User can set this to OFF to build static libraries instead.
option(BUILD_SHARED_LIBS "Build shared library" ON)
option(TEST_COVERAGE "C++ test coverage" OFF)

# Compiler flags
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funroll-loops")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")
if(TEST_COVERAGE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -fprofile-arcs -ftest-coverage")
endif()
set(TVM_RUNTIME_LINKER_LIBS "")

set(TVM_SRC "${PROJECT_SOURCE_DIR}/3rdparty/tvm")
set(TREELITE_SRC "${PROJECT_SOURCE_DIR}/3rdparty/treelite")
set(DMLC_CORE_SRC "${TVM_SRC}/3rdparty/dmlc-core")
set(DLPACK_SRC "${TVM_SRC}/3rdparty/dlpack")

include_directories("${TVM_SRC}/include")
include_directories("${TVM_SRC}/src/runtime")
include_directories("${DLPACK_SRC}/include")
include_directories("${DMLC_CORE_SRC}/include")
include_directories("${TREELITE_SRC}/include")
include_directories("${TREELITE_SRC}/runtime/native/include")
include_directories("${PROJECT_SOURCE_DIR}/include")

FILE(GLOB_RECURSE DLR_SRC
    "src/*.cc"
    ${TVM_SRC}/src/runtime/dso_module.cc
    ${TVM_SRC}/src/runtime/cpu_device_api.cc
    ${TVM_SRC}/src/contrib/sort/sort.cc
)
FILE(GLOB_RECURSE DLR_INC 
    "src/*.h"
    "include/*.h"
    "${TVM_SRC}/src/*.h" 
    "${TVM_SRC}/include/*.h"
    "${TREELITE_SRC}/src/*.h" 
    "${TREELITE_SRC}/include/*.h"
)

if(USE_OPENCL)
    message("USING OpenCL")
    if(USE_OPENCL STREQUAL "ON")
      find_package(OpenCL QUIET)
      if(NOT OpenCL_FOUND)
        message(FATAL_ERROR "OpenCL not found, please specify OpenCL location with -DUSE_OPENCL=/path/to/OpenCL")
      endif(NOT OpenCL_FOUND)
    else(USE_OPENCL STREQUAL "ON")
      set(OpenCL_TOOLKIT_ROOT_DIR ${USE_OPENCL})
      message(STATUS "Custom OPENCL_PATH=" ${OpenCL_TOOLKIT_ROOT_DIR})
      set(OpenCL_INCLUDE_DIRS ${OpenCL_TOOLKIT_ROOT_DIR}/include)
      set(OpenCL_LIBRARIES ${USE_OPENCL}/lib/libOpenCL.so)
    endif(USE_OPENCL STREQUAL "ON")
    include_directories(${OpenCL_INCLUDE_DIRS})
    list(APPEND TVM_RUNTIME_LINKER_LIBS ${OpenCL_LIBRARIES})
    file(GLOB RUNTIME_OPENCL_SRCS ${TVM_SRC}/src/runtime/opencl/*.cc)
    list(APPEND DLR_SRC ${RUNTIME_OPENCL_SRCS})
    set(USE_OPENCL OFF)
endif(USE_OPENCL)

if(USE_CUDA)
    message("USING CUDA")
    find_cuda(${USE_CUDA})
    if(NOT CUDA_FOUND)
        message(FATAL_ERROR "CUDA not found, please specify CUDA location with -DUSE_CUDA=/path/to/cuda/")
    endif(NOT CUDA_FOUND)
    if (NOT USE_CUDA STREQUAL "ON")
      set(CUDA_TOOLKIT_ROOT_DIR ${USE_CUDA})
    endif(NOT USE_CUDA STREQUAL "ON")
    message(STATUS "Custom CUDA_PATH=" ${CUDA_TOOLKIT_ROOT_DIR})
    set(CUDA_INCLUDE_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/include)
    include_directories(${CUDA_INCLUDE_DIRS})
    find_library(_CUDA_CUDA_LIBRARY cuda
      PATHS ${CUDA_TOOLKIT_ROOT_DIR}
      PATH_SUFFIXES lib lib64 lib64/stubs)
    message(STATUS "CUDA_CUDA_LIBRARY: " ${_CUDA_CUDA_LIBRARY})
    if(_CUDA_CUDA_LIBRARY)
      set(CUDA_CUDA_LIBRARY ${_CUDA_CUDA_LIBRARY})
    endif()
    find_library(CUDA_CUDART_LIBRARY cudart
      PATHS ${CUDA_TOOLKIT_ROOT_DIR}
      PATH_SUFFIXES lib lib64)
    message(STATUS "CUDA_CUDART_LIBRARY: " ${CUDA_CUDART_LIBRARY})
    list(APPEND TVM_RUNTIME_LINKER_LIBS ${CUDA_CUDART_LIBRARY})
    list(APPEND TVM_RUNTIME_LINKER_LIBS ${CUDA_CUDA_LIBRARY})
    list(APPEND TVM_RUNTIME_LINKER_LIBS ${CUDA_NVRTC_LIBRARY})
    file(GLOB RUNTIME_CUDA_SRCS ${TVM_SRC}/src/runtime/cuda/*.cc)
    list(APPEND DLR_SRC ${RUNTIME_CUDA_SRCS})
    set(USE_CUDA OFF)
endif()
if(USE_CUDNN)
    message("USING CUDNN")
    set(USE_CUDNN ${USE_CUDA})
    set(CUDNN_TOOLKIT_ROOT_DIR ${USE_CUDNN})
    message(STATUS "Custom CUDNN_PATH=" ${CUDNN_TOOLKIT_ROOT_DIR})
    find_library(CUDA_CUDNN_LIBRARY cudnn
      PATH ${CUDNN_TOOLKIT_ROOT_DIR}
      PATH_SUFFIXES lib lib64)
    if (CUDA_CUDNN_LIBRARY MATCHES "NOTFOUND")
      set(CUDA_CUDNN_LIBRARY ${USE_CUDNN}/lib64/libcudnn.so)
    endif()
    message(STATUS "CUDA_CUDNN_LIBRARY: " ${CUDA_CUDNN_LIBRARY})
    list(APPEND TVM_RUNTIME_LINKER_LIBS ${CUDA_CUDNN_LIBRARY})
    file(GLOB CONTRIB_CUDNN_SRCS ${TVM_SRC}/src/contrib/cudnn/*.cc)
    list(APPEND RUNTIME_SRCS ${CONTRIB_CUDNN_SRCS})
    set(USE_CUDNN_OFF)
endif()
if(USE_TENSORRT)
    message("USING TENSORRT")
    set(TENSORRT_ROOT_DIR ${USE_TENSORRT})
    message(STATUS "Custom TensorRT path: " ${TENSORRT_ROOT_DIR})
    set(TENSORRT_INCLUDE_DIR ${TENSORRT_ROOT_DIR}/include)
    set(TENSORRT_LIB_DIR ${TENSORRT_ROOT_DIR}/lib)
    include_directories(${TENSORRT_INCLUDE_DIR})
    file(GLOB TENSORRT_SRCS ${TVM_SRC}/src/contrib/subgraph/*.cc)
    list(APPEND DLR_SRC ${TENSORRT_SRCS})
    find_library(TENSORRT_NVINFER_LIBRARY nvinfer
      PATH ${TENSORRT_LIB_DIR}
      PATH_SUFFIXES lib lib64)
    if (TENSORRT_NVINFER_LIBRARY MATCHES "NOTFOUND")
      set(TENSORRT_NVINFER_LIBRARY ${USE_TENSORRT}/lib/libnvinfer.so)
    endif()
    message(STATUS "TENSORRT_NVINFER_LIBRARY: " ${TENSORRT_NVINFER_LIBRARY})
    list(APPEND TVM_RUNTIME_LINKER_LIBS ${TENSORRT_NVINFER_LIBRARY})
    add_definitions(-DTVM_GRAPH_RUNTIME_TENSORRT)
    set(USE_TENSORRT OFF)
endif()

set(MAIN_EXEC "")
FILE(GLOB MAIN_SRC src/*.cc)

add_subdirectory(${TVM_SRC} EXCLUDE_FROM_ALL)
add_subdirectory(${TREELITE_SRC} EXCLUDE_FROM_ALL)
add_library(objdlr OBJECT ${DLR_SRC})

#shared_library
find_package(Threads)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
add_library(dlr SHARED $<TARGET_OBJECTS:objdlr>)
set_output_directory(dlr ${PROJECT_SOURCE_DIR}/lib)
set_target_properties(dlr PROPERTIES LINKER_LANGUAGE CXX)
message(STATUS "TVM_RUNTIME_LINKER_LIBS: " ${TVM_RUNTIME_LINKER_LIBS})
target_link_libraries(dlr treelite_runtime_static tvm_runtime_static ${TVM_RUNTIME_LINKER_LIBS})

set(OPREFIX object_)
add_custom_target(combined_lib
	COMMAND mkdir -p ${OPREFIX}tvm_runtime || true && cd ${OPREFIX}tvm_runtime &&  ar -x ${TVM_RUNTIME}
	COMMAND mkdir -p ${OPREFIX}treelite_runtime || true && cd ${OPREFIX}treelite_runtime &&  ar -x ${TREELITE_RUNTIME}
	COMMAND g++  ${OPREFIX}*/*.o -shared -o ${CMAKE_CURRENT_SOURCE_DIR}/libcombined.so
	COMMAND rm -rf ${OPREFIX}*
	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )

# Tests
set(TEST_EXECS "")
file(GLOB TEST_SRCS test/cpp/*.cc)
find_library(GTEST_LIB NAMES libgtest.a gtest)

if(GTEST_LIB)
  foreach(__srcpath ${TEST_SRCS})
    get_filename_component(__srcname ${__srcpath} NAME)
    string(REPLACE ".cc" "" __execname ${__srcname})
    add_executable(${__execname} ${__srcpath})
    list(APPEND TEST_EXECS ${__execname})
    target_link_libraries(${__execname}
      dlr ${GTEST_LIB} pthread)
    set_target_properties(${__execname} PROPERTIES EXCLUDE_FROM_ALL 1)
    set_target_properties(${__execname} PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD 1)
  endforeach()
  add_custom_target(dlrtest DEPENDS ${TEST_EXECS})
endif()

# Group sources
#auto_source_group("${SOURCES}")
