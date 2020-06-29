cmake_minimum_required (VERSION 3.6)
include(cmake/Utils.cmake)
include(3rdparty/tvm/cmake/util/FindCUDA.cmake)

# Option for Android on Arm --- has to come before project() function
option(ANDROID_BUILD "Build for Android target" OFF)
option(AAR_BUILD "Build Android Archive (AAR)" OFF)

if(ANDROID_BUILD)
    set(ANDROID_SYSROOT "${NDK_ROOT}/sysroot")
    if(NOT ANDROID_PLATFORM AND NOT ANDROID_NATIVE_API_LEVEL)
        set(ANDROID_PLATFORM "android-21")
        message(STATUS "ANDROID_PLATFORM not set. Defaulting to android-21")
    endif()
endif(ANDROID_BUILD)

project(dlr)

# The following lines should be after project()
set_default_configuration_release()
msvc_use_static_runtime()
message(STATUS "CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE})
set(CMAKE_LOCAL "${PROJECT_SOURCE_DIR}/cmake")

# CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES stuff should go after project() function
if(ANDROID_BUILD)
    # Disable debugging info for Release build by setting -g level to 0. It will reduce libdlr.so size by a factor of 3.
    # NDK Issue https://github.com/android/ndk/issues/243
    if (CMAKE_BUILD_TYPE STREQUAL "Release")
        string(REPLACE "-g " "-g0 " CMAKE_C_FLAGS ${CMAKE_C_FLAGS})
        string(REPLACE "-g " "-g0 " CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
        string(REPLACE "-g " "-g0 " CMAKE_ASM_FLAGS ${CMAKE_ASM_FLAGS})
    endif()
  # Add ARCH specific header folder to CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES
    if (ANDROID_ABI STREQUAL "x86_64")
        list(APPEND CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES ${ANDROID_SYSROOT}/usr/include/x86_64-linux-android)
    elseif (ANDROID_ABI STREQUAL "x86")
        list(APPEND CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES ${ANDROID_SYSROOT}/usr/include/i686-linux-android)
    elseif (ANDROID_ABI STREQUAL "arm64-v8a")
        list(APPEND CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES ${ANDROID_SYSROOT}/usr/include/aarch64-linux-android)
    else() # Default to armv7a which matches NDK toolchain.cmake behavior
        list(APPEND CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES ${ANDROID_SYSROOT}/usr/include/arm-linux-androideabi)
    endif()
    message(STATUS "CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES: ${CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES}")
    message(STATUS "Android CMAKE_C_FLAGS: " ${CMAKE_C_FLAGS})
    message(STATUS "Android CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS})
    message(STATUS "Android CMAKE_ASM_FLAGS: " ${CMAKE_ASM_FLAGS})
endif(ANDROID_BUILD)

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
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funroll-loops")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")
if(TEST_COVERAGE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -fprofile-arcs -ftest-coverage")
endif()
set(DLR_LINKER_LIBS "")
if (MSVC)
    add_definitions(-DTVM_EXPORTS)
endif()

set(TVM_SRC "${PROJECT_SOURCE_DIR}/3rdparty/tvm")
set(JSON_SRC "${PROJECT_SOURCE_DIR}/3rdparty/json")
set(LIBNPY_SRC "${PROJECT_SOURCE_DIR}/3rdparty/libnpy")
set(TREELITE_SRC "${PROJECT_SOURCE_DIR}/3rdparty/treelite")
set(DMLC_CORE_SRC "${TVM_SRC}/3rdparty/dmlc-core")
set(DLPACK_SRC "${TVM_SRC}/3rdparty/dlpack")
set(FMT_SRC "${TREELITE_SRC}/3rdparty/fmt")

include_directories("${TVM_SRC}/include")
include_directories("${TVM_SRC}/src/runtime")
include_directories("${DLPACK_SRC}/include")
include_directories("${DMLC_CORE_SRC}/include")
include_directories("${FMT_SRC}/include")
include_directories("${TREELITE_SRC}/include")
include_directories("${TREELITE_SRC}/runtime/native/include")
include_directories("${PROJECT_SOURCE_DIR}/include")
include_directories("${JSON_SRC}")

# Add only top level *.cc files (non-RECURSE)
FILE(GLOB DLR_SRC
    "src/*.cc"
    ${TVM_SRC}/src/runtime/dso_library.cc
    ${TVM_SRC}/src/runtime/cpu_device_api.cc
    ${TVM_SRC}/src/runtime/contrib/sort/sort.cc
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
    list(APPEND DLR_LINKER_LIBS ${OpenCL_LIBRARIES})
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
    list(APPEND DLR_LINKER_LIBS ${CUDA_CUDART_LIBRARY})
    list(APPEND DLR_LINKER_LIBS ${CUDA_CUDA_LIBRARY})
    list(APPEND DLR_LINKER_LIBS ${CUDA_NVRTC_LIBRARY})
    file(GLOB RUNTIME_CUDA_SRCS ${TVM_SRC}/src/runtime/cuda/*.cc)

    # Thrust is used to improve NMS performance.
    message(STATUS "Build with Thrust support")
    cmake_minimum_required(VERSION 3.13) # to compile CUDA code
    set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc)
    if(NOT DEFINED CMAKE_CUDA_STANDARD)
      set(CMAKE_CUDA_STANDARD 11)
      set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    endif()
    enable_language(CUDA)
    file(GLOB CONTRIB_THRUST_SRC ${TVM_SRC}/src/runtime/contrib/thrust/*.cu)
    list(APPEND RUNTIME_CUDA_SRCS ${CONTRIB_THRUST_SRC})

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
    list(APPEND DLR_LINKER_LIBS ${CUDA_CUDNN_LIBRARY})
    file(GLOB CONTRIB_CUDNN_SRCS ${TVM_SRC}/src/contrib/cudnn/*.cc)
    list(APPEND RUNTIME_SRCS ${CONTRIB_CUDNN_SRCS})
    set(USE_CUDNN_OFF)
endif()
if(USE_TENSORRT)
    message("USING TENSORRT")
    if(IS_DIRECTORY ${USE_TENSORRT})
        set(TENSORRT_ROOT_DIR ${USE_TENSORRT})
        message(STATUS "Custom TensorRT path: " ${TENSORRT_ROOT_DIR})
    endif()
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS ${TENSORRT_ROOT_DIR} PATH_SUFFIXES include)
    find_library(TENSORRT_LIB_DIR nvinfer HINTS ${TENSORRT_ROOT_DIR} PATH_SUFFIXES lib)
    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(TENSORRT DEFAULT_MSG TENSORRT_INCLUDE_DIR TENSORRT_LIB_DIR)
    if(NOT TENSORRT_FOUND)
        message(ERROR "Could not find TensorRT.")
    endif()
    message(STATUS "TENSORRT_LIB_DIR: " ${TENSORRT_LIB_DIR})
    include_directories(${TENSORRT_INCLUDE_DIR})
    list(APPEND DLR_LINKER_LIBS ${TENSORRT_LIB_DIR})

    # NNVM TRT Integration
    file(GLOB TENSORRT_SRCS ${TVM_SRC}/src/contrib/subgraph/*.cc)
    list(APPEND DLR_SRC ${TENSORRT_SRCS})
    add_definitions(-DTVM_GRAPH_RUNTIME_TENSORRT)

    # TRT Relay sources
    file(GLOB TENSORRT_RELAY_SRCS ${TVM_SRC}/src/runtime/contrib/tensorrt/*.cc)
    list(APPEND DLR_SRC ${TENSORRT_RELAY_SRCS})
    # Additional TVM compiler sources
    include_directories("${TVM_SRC}/3rdparty/rang/include")
    include_directories("${TVM_SRC}/3rdparty/compiler-rt")
    include_directories("${TVM_SRC}/topi/include")
    list(APPEND DLR_SRC
         # Ops
         ${TVM_SRC}/src/relay/op/nn/nn.cc
         ${TVM_SRC}/src/relay/op/nn/pooling.cc
         ${TVM_SRC}/src/relay/op/nn/convolution.cc
         ${TVM_SRC}/src/relay/op/nn/pad.cc
         ${TVM_SRC}/src/relay/op/tensor/binary.cc
         ${TVM_SRC}/src/relay/op/tensor/unary.cc
         ${TVM_SRC}/src/relay/op/tensor/transform.cc
         ${TVM_SRC}/src/relay/op/tensor/reduce.cc
         ${TVM_SRC}/src/relay/op/image/resize.cc
         ${TVM_SRC}/src/node/container.cc
         ${TVM_SRC}/src/ir/attrs.cc

         ${TVM_SRC}/src/relay/ir/expr_functor.cc
         ${TVM_SRC}/src/relay/ir/pattern_functor.cc
         ${TVM_SRC}/src/relay/op/type_relations.cc
         ${TVM_SRC}/src/tir/ir/op.cc
         ${TVM_SRC}/src/arith/int_set.cc
         ${TVM_SRC}/src/tir/ir/expr.cc
         ${TVM_SRC}/src/te/tensor.cc
         ${TVM_SRC}/src/relay/ir/expr.cc
         ${TVM_SRC}/src/ir/expr.cc
         ${TVM_SRC}/src/relay/ir/function.cc
         ${TVM_SRC}/src/arith/canonical_simplify.cc
         ${TVM_SRC}/src/arith/rewrite_simplify.cc
         ${TVM_SRC}/src/arith/ir_mutator_with_analyzer.cc
         ${TVM_SRC}/src/tir/ir/stmt_functor.cc
         ${TVM_SRC}/src/tir/ir/expr_functor.cc
         ${TVM_SRC}/src/tir/ir/stmt.cc
         ${TVM_SRC}/src/tir/analysis/deep_equal.cc
         ${TVM_SRC}/src/relay/ir/adt.cc
         ${TVM_SRC}/src/arith/const_int_bound.cc
         ${TVM_SRC}/src/tir/ir/data_layout.cc
         ${TVM_SRC}/src/tir/pass/simple_passes.cc
         ${TVM_SRC}/src/ir/type.cc
         ${TVM_SRC}/src/ir/tensor_type.cc
         ${TVM_SRC}/src/ir/op.cc
         ${TVM_SRC}/src/arith/analyzer.cc
         ${TVM_SRC}/src/arith/modular_set.cc
         ${TVM_SRC}/src/arith/stmt_simplify.cc
         ${TVM_SRC}/src/node/repr_printer.cc
         ${TVM_SRC}/src/ir/env_func.cc
         ${TVM_SRC}/src/relay/analysis/util.cc
         ${TVM_SRC}/src/ir/type_functor.cc
         ${TVM_SRC}/src/ir/module.cc
         ${TVM_SRC}/src/relay/transforms/type_infer.cc
         ${TVM_SRC}/src/ir/error.cc
         ${TVM_SRC}/src/relay/analysis/type_solver.cc
         ${TVM_SRC}/src/node/reflection.cc
         ${TVM_SRC}/src/relay/ir/base.cc
         ${TVM_SRC}/src/relay/analysis/kind_check.cc
         ${TVM_SRC}/src/printer/relay_text_printer.cc
         ${TVM_SRC}/src/printer/doc.cc
         ${TVM_SRC}/src/relay/analysis/dependency_graph.cc
         ${TVM_SRC}/src/node/structural_equal.cc
         ${TVM_SRC}/src/relay/analysis/match_exhaustion.cc
         ${TVM_SRC}/src/node/serialization.cc
         ${TVM_SRC}/src/relay/transforms/eta_expand.cc
         ${TVM_SRC}/src/ir/transform.cc
         ${TVM_SRC}/src/ir/type_relation.cc
         ${TVM_SRC}/src/relay/transforms/de_duplicate.cc
         ${TVM_SRC}/src/te/operation/compute_op.cc
         ${TVM_SRC}/src/te/schedule/message_passing.cc
         ${TVM_SRC}/src/te/operation/op_util.cc
         ${TVM_SRC}/src/tir/pass/ir_util.cc
         ${TVM_SRC}/src/te/operation/cross_thread_reduction.cc
         ${TVM_SRC}/src/te/operation/tensorize.cc
         ${TVM_SRC}/src/tir/pass/arg_binder.cc
         ${TVM_SRC}/src/relay/ir/transform.cc
         ${TVM_SRC}/src/relay/analysis/well_formed.cc
         )
    set(USE_TENSORRT OFF)
endif()
if(WITH_TENSORFLOW_LITE_LIB)
    if(NOT WITH_TENSORFLOW_LITE_LIB MATCHES ".*libtensorflow-lite\.a$")
        message(FATAL_ERROR "WITH_TENSORFLOW_LITE_LIB should point to static library libtensorflow-lite.a")
    endif()
    message("Adding libtensorflow-lite.a to DLR shared library")

    # Download Tensorflow and FlatBuffers source code
    set(TENSORFLOW_VER "1.15.2")
    set(TENSORFLOW_SHA1 "b9e71bf287a186ceb9d859588671d890c45e51a7")
    set(FLATBUFFERS_VER "1.11.0")
    set(FLATBUFFERS_SHA1 "2b2633902c57c6980b3a41b44d8ad933f214d71d")
    set(TENSORFLOW_URL "https://github.com/tensorflow/tensorflow/archive/v${TENSORFLOW_VER}.tar.gz")
    set(TENSORFLOW_TGZ "/tmp/tensorflow-${TENSORFLOW_VER}.tar.gz")
    set(TENSORFLOW_SRC "${PROJECT_SOURCE_DIR}/3rdparty/tensorflow-${TENSORFLOW_VER}")
    set(FLATBUFFERS_URL "https://github.com/google/flatbuffers/archive/v${FLATBUFFERS_VER}.tar.gz")
    set(FLATBUFFERS_TGZ "/tmp/flatbuffers-${FLATBUFFERS_VER}.tar.gz")
    set(FLATBUFFERS_SRC "${PROJECT_SOURCE_DIR}/3rdparty/flatbuffers-${FLATBUFFERS_VER}")

    download_file(${TENSORFLOW_URL} ${TENSORFLOW_TGZ} SHA1 ${TENSORFLOW_SHA1})
    download_file(${FLATBUFFERS_URL} ${FLATBUFFERS_TGZ} SHA1 ${FLATBUFFERS_SHA1})

    # Extract Tensorflow and FlatBuffers source code
    message(STATUS "Extracting " ${TENSORFLOW_TGZ} " ...")
    execute_process(
        COMMAND rm -rf ${PROJECT_SOURCE_DIR}/3rdparty/tensorflow*
        COMMAND tar -C ${PROJECT_SOURCE_DIR}/3rdparty -zxf ${TENSORFLOW_TGZ}
        RESULT_VARIABLE tf_ret
    )
    if(NOT tf_ret EQUAL "0")
        message(FATAL_ERROR "Failed to extract " ${TENSORFLOW_TGZ})
    endif()
    message(STATUS "Extracting " ${FLATBUFFERS_TGZ} " ...")
    execute_process(
        COMMAND rm -rf ${PROJECT_SOURCE_DIR}/3rdparty/flatbuffers*
        COMMAND tar -C ${PROJECT_SOURCE_DIR}/3rdparty -zxf ${FLATBUFFERS_TGZ}
        RESULT_VARIABLE fb_ret
    )
    if(NOT fb_ret EQUAL "0")
        message(FATAL_ERROR "Failed to extract " ${FLATBUFFERS_TGZ})
    endif()

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DDLR_TFLITE")
    include_directories("${TENSORFLOW_SRC}")
    include_directories("${FLATBUFFERS_SRC}/include")
    list(APPEND DLR_SRC "src/dlr_tflite/dlr_tflite.cc")

    if(ANDROID_BUILD) # Android build needs additional library liblog
        list(APPEND DLR_LINKER_LIBS -Wl,--whole-archive ${WITH_TENSORFLOW_LITE_LIB} -Wl,--no-whole-archive -llog)
    elseif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin") # Mac OS linker is different from Linux.
        # Use -force_load instead of --whole-archive
        list(APPEND DLR_LINKER_LIBS -Wl,-force_load,${WITH_TENSORFLOW_LITE_LIB})
    else() # Regulal Linux Build
        list(APPEND DLR_LINKER_LIBS -Wl,--whole-archive ${WITH_TENSORFLOW_LITE_LIB} -Wl,--no-whole-archive -ldl)
    endif()
endif()
if(WITH_TENSORFLOW_LIB)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDLR_TENSORFLOW")
    include_directories("${WITH_TENSORFLOW_LIB}/include")
    list(APPEND DLR_SRC "src/dlr_tensorflow/dlr_tensorflow.cc")
    list(APPEND DLR_LINKER_LIBS -L${WITH_TENSORFLOW_LIB}/lib -ltensorflow_framework -ltensorflow)
endif()
if(WITH_HEXAGON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDLR_HEXAGON")
    list(APPEND DLR_SRC "src/dlr_hexagon/dlr_hexagon.cc")
endif()
if(AAR_BUILD)
    list(APPEND DLR_SRC "src/jni/dlr_jni.cc")
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
set_output_directory(dlr ${CMAKE_BINARY_DIR}/lib)
set_target_properties(dlr PROPERTIES LINKER_LANGUAGE CXX)
message(STATUS "DLR_LINKER_LIBS: " ${DLR_LINKER_LIBS})
if(ANDROID_BUILD)
  target_link_libraries(dlr treelite_runtime_static tvm_runtime_static ${DLR_LINKER_LIBS})
else(ANDROID_BUILD)
  target_link_libraries(dlr treelite_runtime_static tvm_runtime_static ${DLR_LINKER_LIBS} -lpthread)
endif()

add_library(dlr_static STATIC $<TARGET_OBJECTS:objdlr>)
set_output_directory(dlr_static ${CMAKE_BINARY_DIR}/lib)

set(OPREFIX object_)
add_custom_target(combined_lib
	COMMAND mkdir -p ${OPREFIX}tvm_runtime || true && cd ${OPREFIX}tvm_runtime &&  ar -x ${TVM_RUNTIME}
	COMMAND mkdir -p ${OPREFIX}treelite_runtime || true && cd ${OPREFIX}treelite_runtime &&  ar -x ${TREELITE_RUNTIME}
	COMMAND g++  ${OPREFIX}*/*.o -shared -o ${CMAKE_CURRENT_SOURCE_DIR}/libcombined.so
	COMMAND rm -rf ${OPREFIX}*
	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )

# Demos
set(DEMO_EXECS "")
file(GLOB DEMO_SRCS demo/cpp/*.cc)

foreach(__srcpath ${DEMO_SRCS})
  get_filename_component(__srcname ${__srcpath} NAME)
  string(REPLACE ".cc" "" __execname ${__srcname})
  list(APPEND DEMO_EXECS ${__execname})
  add_executable(${__execname} ${__srcpath} $<TARGET_OBJECTS:objdlr>)
  target_include_directories(${__execname} PUBLIC ${LIBNPY_SRC})
  if (ANDROID_BUILD)
    target_link_libraries(${__execname} PRIVATE treelite_runtime_static tvm_runtime_static ${DLR_LINKER_LIBS} -fuse-ld=gold)
  else (ANDROID_BUILD)
    target_link_libraries(${__execname} PRIVATE treelite_runtime_static tvm_runtime_static ${DLR_LINKER_LIBS} -lpthread)     
  endif ()
  set_output_directory(${__execname} ${CMAKE_BINARY_DIR}/bin)
  set_target_properties(${__execname} PROPERTIES EXCLUDE_FROM_ALL 1)
  set_target_properties(${__execname} PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD 1)
endforeach()
add_custom_target(demo DEPENDS ${DEMO_EXECS})

# Tests
if(NOT(AAR_BUILD))
  include(cmake/googletest.cmake)
  fetch_googletest(
    ${PROJECT_SOURCE_DIR}/cmake
    ${PROJECT_BINARY_DIR}/googletest
    )

  enable_testing()

  file(GLOB TEST_SRCS tests/cpp/*.cc)
  if(WITH_TENSORFLOW_LITE_LIB)
    file(GLOB TFLITE_TEST_SRCS tests/cpp/dlr_tflite/*.cc)
    list(APPEND TEST_SRCS ${TFLITE_TEST_SRCS})
  endif()
  if(WITH_TENSORFLOW_LIB)
    file(GLOB TENSORFLOW_TEST_SRCS tests/cpp/dlr_tensorflow/*.cc)
    list(APPEND TEST_SRCS ${TENSORFLOW_TEST_SRCS})
  endif()
  if(WITH_HEXAGON)
    file(GLOB HEXAGON_TEST_SRCS tests/cpp/dlr_hexagon/*.cc)
    list(APPEND TEST_SRCS ${HEXAGON_TEST_SRCS})
  endif()
  foreach(__srcpath ${TEST_SRCS})
    get_filename_component(__srcname ${__srcpath} NAME)
    string(REPLACE ".cc" "" __execname ${__srcname})
    add_executable(${__execname} ${__srcpath})
    target_link_libraries(${__execname} dlr gtest_main)
    set_output_directory(${__execname} ${CMAKE_BINARY_DIR})
    add_test(NAME ${__execname} COMMAND ${__execname})
    message(STATUS "Added Test: " ${__execname})
  endforeach()

  set(CAT_IMAGE ${CMAKE_CURRENT_BINARY_DIR}/cat224-3.txt)
  if(NOT EXISTS ${CAT_IMAGE})
  download_file(
    https://neo-ai-dlr-test-artifacts.s3-us-west-2.amazonaws.com/tflite-models/cat224-3.txt
    ${CAT_IMAGE}
    SHA1
    e35e82f3371bed37caa7ecece417f50876414077
  )
  endif()
  
  # Download compiled model for unit tests
  set(RESNET_MODEL ${CMAKE_CURRENT_BINARY_DIR}/resnet_v1_5_50)
  if(NOT IS_DIRECTORY ${RESNET_MODEL})
    file(MAKE_DIRECTORY ${RESNET_MODEL})
    download_file(
      https://neo-ai-dlr-test-artifacts.s3-us-west-2.amazonaws.com/compiled-models/resnet_v1.5_50-ml_c4.tar.gz
      /tmp/resnet_v1.5_50-ml_c4.tar.gz
      SHA1
      447c22239e63882a2bc754db550131756373d4df
    )
    # this is OS-agnostic
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar -xf /tmp/resnet_v1.5_50-ml_c4.tar.gz
                    WORKING_DIRECTORY ${RESNET_MODEL})
    file(REMOVE /tmp/resnet_v1.5_50-ml_c4.tar.gz)
  endif()

  set(XGBOOST_TEST_MODEL ${CMAKE_CURRENT_BINARY_DIR}/xgboost_test)
  if(NOT IS_DIRECTORY ${XGBOOST_TEST_MODEL})
    file(MAKE_DIRECTORY ${XGBOOST_TEST_MODEL})
    download_file(
      https://neo-ai-dlr-test-artifacts.s3-us-west-2.amazonaws.com/compiled-models/xgboost_test.tar.gz
      /tmp/xgboost_test.tar.gz
      SHA1
      4c8ac5f20db6c7c6d674dd2ac9d9e14ce887281e
    )
    # this is OS-agnostic
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar -xf /tmp/xgboost_test.tar.gz
                    WORKING_DIRECTORY ${XGBOOST_TEST_MODEL})
    file(REMOVE /tmp/xgboost_test.tar.gz)
  endif()


  if(WITH_TENSORFLOW_LITE_LIB)
      # Download Test TFLite model
      file(MAKE_DIRECTORY mobilenet_v2_0.75_224)
      download_file(
        https://neo-ai-dlr-test-artifacts.s3-us-west-2.amazonaws.com/tflite-models/mobilenet_v2_0.75_224.tflite
        ./mobilenet_v2_0.75_224/mobilenet_v2_0.75_224.tflite
        SHA1
        c653a09facaf2dfb5e3910030b3f74ad04259e30
      )
      download_file(
        https://neo-ai-dlr-test-artifacts.s3-us-west-2.amazonaws.com/tflite-models/mobilenet_v1_0.75_224_quant.tflite
        ./mobilenet_v1_0.75_224_quant/mobilenet_v1_0.75_224_quant.tflite
        SHA1
        0f5e9dae3a41c5b198595bcb5ba50b936ee61527
      )
  endif() # WITH_TENSORFLOW_LITE_LIB
  if(WITH_TENSORFLOW_LIB)
      # Download Test Tensorflow model
      file(MAKE_DIRECTORY mobilenet_v1_1.0_224)
      download_file(
        https://neo-ai-dlr-test-artifacts.s3-us-west-2.amazonaws.com/tf-models/mobilenet_v1_1.0_224_frozen.pb
        ./mobilenet_v1_1.0_224/mobilenet_v1_1.0_224_frozen.pb
        SHA1
        4df8525bcf2ec96296098ba7fec0fff1abe32fce
      )
  endif() # WITH_TENSORFLOW_LIB
  if(WITH_HEXAGON)
      # Download Test Hexagon model for Android 64 aarch64
      file(MAKE_DIRECTORY dlr_hexagon_model)
      download_file(
        https://neo-ai-dlr-test-artifacts.s3-us-west-2.amazonaws.com/hexagon-models/android_ReleaseG_aarch64/mobilenet_v1_0.75_224_quant_hexagon_model.so
        ./dlr_hexagon_model/mobilenet_v1_0.75_224_quant_hexagon_model.so
        SHA1
        989d6f1613e948e432a31d5b5741bff7f9a9bacb
      )
      # Download Hexagon NNLib for Hexagon V65
      download_file(
        https://neo-ai-dlr-test-artifacts.s3-us-west-2.amazonaws.com/hexagon-models/hexagon_ReleaseG_dynamic_toolv83_v65/libhexagon_nn_skel.so
        ./dlr_hexagon_model/libhexagon_nn_skel.so
        SHA1
        6746c34f54aad3df24d9fc5f632ebd9dfc64ed69
      )
  endif() # WITH_HEXAGON
endif()
# Group sources
#auto_source_group("${SOURCES}")
