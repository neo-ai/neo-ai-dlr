cmake_minimum_required (VERSION 3.1)
project(dlr)
#include(cmake/Utils.cmake)

# Use RPATH on Mac OS X as flexible mechanism for locating dependencies
# See https://blog.kitware.com/upcoming-in-cmake-2-8-12-osx-rpath-support/
set(CMAKE_MACOSX_RPATH TRUE)

set(python-build "bp3-python setup.py build")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build/lib")

# Set BUILD_SHARED_LIBS as option. By default, build shared libraries;
# User can set this to OFF to build static libraries instead.
option(BUILD_SHARED_LIBS "Build shared library" ON)
option(TEST_COVERAGE "C++ test coverage" OFF)

#set_default_configuration_release()

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
set(TREELITE_SRC "${PROJECT_SOURCE_DIR}/3rdparty/neo-ai-treelite")
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

set(USE_CUDA OFF)
set(USE_CUDNN OFF)
set(USE_TENSORRT OFF)

if(USE_CUDA STREQUAL "ON")
    message("USING CUDA")
    set(USE_CUDA "/usr/local/cuda-9.0")
    set(CUDA_TOOLKIT_ROOT_DIR ${USE_CUDA})
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
endif()
if(USE_CUDNN STREQUAL "ON")
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
endif()
if(USE_TENSORRT STREQUAL "ON")
    message("USING TENSORRT")
    set(USE_TENSORRT "/home/ubuntu/TensorRT-4.0.1.6")
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
    #set_source_files_properties(src/runtime/graph/graph_runtime.cc PROPERTIES COMPILE_DEFINITIONS "TVM_GRAPH_RUNTIME_TENSORRT")
    #set_property(GLOBAL PROPERTY COMPILE_DEFINITIONS "TVM_GRAPH_RUNTIME_TENSORRT")
    #set_target_properties(dlr PROPERTIES COMPILE_DEFINITIONS "TVM_GRAPH_RUNTIME_TENSORRT")
    add_definitions(-DTVM_GRAPH_RUNTIME_TENSORRT)
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
