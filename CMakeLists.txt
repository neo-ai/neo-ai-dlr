cmake_minimum_required (VERSION 3.6)
include(cmake/Utils.cmake)
include(3rdparty/tvm/cmake/utils/Utils.cmake)
include(3rdparty/tvm/cmake/utils/FindCUDA.cmake)

# Option for Android on Arm --- has to come before project() function
option(ANDROID_BUILD "Build for Android target" OFF)
option(AAR_BUILD "Build Android Archive (AAR)" OFF)
option(DLR_BUILD_TESTS "Build DLR tests" ON)
option(DLR_STATIC_LIBSTDCPP "Use -static-libstdc++" OFF)

if(ANDROID_BUILD)
    set(ANDROID_SYSROOT "${NDK_ROOT}/sysroot")
    if(NOT ANDROID_PLATFORM AND NOT ANDROID_NATIVE_API_LEVEL)
        set(ANDROID_PLATFORM "android-21")
        message(STATUS "ANDROID_PLATFORM not set. Defaulting to android-21")
    endif()
endif(ANDROID_BUILD)

project(dlr VERSION 1.9.0 LANGUAGES C CXX)

message(STATUS "dlr version: ${dlr_VERSION}")

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
option(ENABLE_DATATRANSFORM "Enable datatransform for Sagemaker-scikit-learn-extension models" OFF)


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
# --exclude-libs is not available on Windows and macOS. As such, Windows and 
# Mac do not support the creation of multiple DLRModel instances (in Python) in 
# case model folders have their own libdlr.so.
if (WIN32 OR APPLE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
else()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--exclude-libs,ALL")
endif()
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
set(DLFCN_WIN32 "${PROJECT_SOURCE_DIR}/3rdparty/dlfcn-win32")
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
    ${TVM_SRC}/src/runtime/metadata_module.cc
    ${TVM_SRC}/src/runtime/contrib/sort/sort.cc
)

if(NOT(ENABLE_DATATRANSFORM))
  list(REMOVE_ITEM DLR_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/dlr_data_transform.cc)
endif()

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
      set(CMAKE_CUDA_STANDARD 14)
      set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
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

    # TRT runtime sources
    file(GLOB RUNTIME_TENSORRT_SRCS ${TVM_SRC}/src/runtime/contrib/tensorrt/*.cc)
    set_source_files_properties(${RUNTIME_TENSORRT_SRCS} PROPERTIES COMPILE_FLAGS "-Wno-deprecated-declarations")
    list(APPEND DLR_SRC ${RUNTIME_TENSORRT_SRCS})

    # Set defines
    add_definitions(-DTVM_GRAPH_RUNTIME_TENSORRT)

    set(USE_TENSORRT OFF)
endif()
if(WITH_HEXAGON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDLR_HEXAGON")
    list(APPEND DLR_SRC "src/dlr_hexagon/dlr_hexagon.cc")
endif()

if(ENABLE_DATATRANSFORM)
    add_definitions(-DENABLE_DATATRANSFORM) 
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
target_link_libraries(dlr PRIVATE treelite_runtime_static tvm_runtime_static)
target_link_libraries(dlr PUBLIC ${DLR_LINKER_LIBS})
if(DLR_STATIC_LIBSTDCPP)
  message(STATUS "Using -static-libstdc++")
  target_link_libraries(dlr PRIVATE -static-libstdc++)
endif()
if(NOT(ANDROID_BUILD))
  target_link_libraries(dlr PUBLIC pthread)
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

target_include_directories(dlr INTERFACE $<INSTALL_INTERFACE:include>)
install(TARGETS dlr
        DESTINATION lib
        EXPORT dlrTargets)
install(FILES include/dlr.h DESTINATION include)
install(EXPORT dlrTargets
  FILE dlrTargets.cmake
  DESTINATION lib/cmake/dlr
)

include(CMakePackageConfigHelpers)
# generate the config file that is includes the exports
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/dlrConfig.cmake"
  INSTALL_DESTINATION "lib/cmake/dlr"
  NO_SET_AND_CHECK_MACRO
  NO_CHECK_REQUIRED_COMPONENTS_MACRO
  )
# generate the version file for the config file
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/dlrConfigVersion.cmake"
  VERSION "${dlr_VERSION}"
  COMPATIBILITY AnyNewerVersion
)

# install the configuration file
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/dlrConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/dlrConfigVersion.cmake
  DESTINATION lib/cmake/dlr
)

# Tests
if(DLR_BUILD_TESTS AND NOT(AAR_BUILD))
  include(cmake/googletest.cmake)
  if (WIN32)
    include_directories(${DLFCN_WIN32}/src)
    add_subdirectory(${DLFCN_WIN32})
  endif()

  fetch_googletest(
    ${PROJECT_SOURCE_DIR}/cmake
    ${PROJECT_BINARY_DIR}/googletest
  )

  enable_testing()
  file(GLOB TEST_SRCS tests/cpp/*.cc)

  if(NOT(ENABLE_DATATRANSFORM))
    list(REMOVE_ITEM TEST_SRCS 
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp/dlr_data_transform_test.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp/dlr_pipeline_skl_xgb_test.cc)
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

  file(GLOB TEST_DLSYM_SRCS tests/cpp/dlsym/*.cc)
  foreach(__srcpath ${TEST_DLSYM_SRCS})
    get_filename_component(__srcname ${__srcpath} NAME)
    string(REPLACE ".cc" "" __execname ${__srcname})
    add_executable(${__execname} ${__srcpath})
    target_link_libraries(${__execname} gtest_main dl) 
    set_output_directory(${__execname} ${CMAKE_BINARY_DIR})
    add_test(NAME ${__execname} COMMAND ${__execname})
    message(STATUS "Added dlsym Test: " ${__execname})
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
  
  set(STREET_IMAGE ${CMAKE_CURRENT_BINARY_DIR}/street_small.npy)
  if(NOT EXISTS ${STREET_IMAGE})
  download_file(
    https://neo-ai-dlr-test-artifacts.s3-us-west-2.amazonaws.com/test-data/street_small.npy
    ${STREET_IMAGE}
    SHA1
    206d1a12646a5fe43d0877d38bd137c70a5adc3b
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

  set(RELAYVM_MODEL ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03-LINUX_X86_64.tar.gz)
  set(RELAYVM_MODEL_DIR ${CMAKE_CURRENT_BINARY_DIR}/ssd_mobilenet_v1)
  if(NOT IS_DIRECTORY ${RELAYVM_MODEL_DIR})
    file(MAKE_DIRECTORY ${RELAYVM_MODEL_DIR})
    download_file(
      https://neo-ai-dlr-test-artifacts.s3-us-west-2.amazonaws.com/compiled-models/release-1.5.0/${RELAYVM_MODEL}
      /tmp/${RELAYVM_MODEL}
      SHA1
      49ddd9e815c6cc14ef0e9a594c8c2d0d129e5e91
    )
    # this is OS-agnostic
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar -xf /tmp/${RELAYVM_MODEL}
                    WORKING_DIRECTORY ${RELAYVM_MODEL_DIR})
    file(REMOVE /tmp/${RELAYVM_MODEL})
  endif()

  set(AUTOML_MODEL automl-ml_m4.tar.gz)
  set(AUTOML_MODEL_DIR ${CMAKE_CURRENT_BINARY_DIR}/automl)
  if(NOT IS_DIRECTORY ${AUTOML_MODEL_DIR})
    file(MAKE_DIRECTORY ${AUTOML_MODEL_DIR})
    download_file(
      https://neo-ai-dlr-test-artifacts.s3-us-west-2.amazonaws.com/compiled-models/release-1.5.0/${AUTOML_MODEL}
      /tmp/${AUTOML_MODEL}
      SHA1
      a17df0b2d980051fb39a72dc61265b97ea0e4639
    )
    # this is OS-agnostic
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar -xf /tmp/${AUTOML_MODEL}
                    WORKING_DIRECTORY ${AUTOML_MODEL_DIR})
    file(REMOVE /tmp/${AUTOML_MODEL})
  endif()

  set(AUTOML_STATIC_MODEL automl_static-ml_m4.tar.gz)
  set(AUTOML_STATIC_MODEL_DIR ${CMAKE_CURRENT_BINARY_DIR}/automl_static)
  if(NOT IS_DIRECTORY ${AUTOML_STATIC_MODEL_DIR})
    file(MAKE_DIRECTORY ${AUTOML_STATIC_MODEL_DIR})
    download_file(
      https://neo-ai-dlr-test-artifacts.s3-us-west-2.amazonaws.com/compiled-models/release-1.9.0/${AUTOML_STATIC_MODEL}
      /tmp/${AUTOML_STATIC_MODEL}
      SHA1
      d776a30bcaeceff99844958fc88085972f5aec24
    )
    # this is OS-agnostic
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar -xf /tmp/${AUTOML_STATIC_MODEL}
                    WORKING_DIRECTORY ${AUTOML_STATIC_MODEL_DIR})
    file(REMOVE /tmp/${AUTOML_STATIC_MODEL})
  endif()

  set(PIPELINE_MODEL1 ${CMAKE_CURRENT_BINARY_DIR}/pipeline_model1)
  if(NOT IS_DIRECTORY ${PIPELINE_MODEL1})
    file(MAKE_DIRECTORY ${PIPELINE_MODEL1})
    download_file(
      https://neo-ai-dlr-test-artifacts.s3-us-west-2.amazonaws.com/compiled-models/pipeline_model1-LINUX_X86_64.tar.gz
      /tmp/pipeline_model1-LINUX_X86_64.tar.gz
      SHA1
      b605cc76bb38b6462a1bd6881955cb5ae2de3218
    )
    # this is OS-agnostic
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar -xf /tmp/pipeline_model1-LINUX_X86_64.tar.gz
                    WORKING_DIRECTORY ${PIPELINE_MODEL1})
    file(REMOVE /tmp/pipeline_model1-LINUX_X86_64.tar.gz)
  endif()

  set(PIPELINE_MODEL2 ${CMAKE_CURRENT_BINARY_DIR}/pipeline_model2)
  if(NOT IS_DIRECTORY ${PIPELINE_MODEL2})
    file(MAKE_DIRECTORY ${PIPELINE_MODEL2})
    download_file(
      https://neo-ai-dlr-test-artifacts.s3-us-west-2.amazonaws.com/compiled-models/release-1.5.0/pipeline_model2-LINUX_X86_64.tar.gz
      /tmp/pipeline_model2-LINUX_X86_64.tar.gz
      SHA1
      4c480e66f243db9b0fdb5528babb606091eabaad
    )
    # this is OS-agnostic
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar -xf /tmp/pipeline_model2-LINUX_X86_64.tar.gz
                    WORKING_DIRECTORY ${PIPELINE_MODEL2})
    file(REMOVE /tmp/pipeline_model2-LINUX_X86_64.tar.gz)
  endif()

  set(INVERSELABEL_MODEL inverselabel-ml_m4.tar.gz)
  set(INVERSELABEL_MODEL_DIR ${CMAKE_CURRENT_BINARY_DIR}/inverselabel)
  if(NOT IS_DIRECTORY ${INVERSELABEL_MODEL_DIR})
    file(MAKE_DIRECTORY ${INVERSELABEL_MODEL_DIR})
    download_file(
      https://neo-ai-dlr-test-artifacts.s3-us-west-2.amazonaws.com/compiled-models/release-1.5.0/${INVERSELABEL_MODEL}
      /tmp/${INVERSELABEL_MODEL}
      SHA1
      055f9606c97d781f0468ad839ff42b3d41474315
    )
    # this is OS-agnostic
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar -xf /tmp/${INVERSELABEL_MODEL}
                    WORKING_DIRECTORY ${INVERSELABEL_MODEL_DIR})
    file(REMOVE /tmp/${INVERSELABEL_MODEL})
  endif()

  set(INPUTORDER_MODEL input_order-ml_m4.tar.gz)
  set(INPUTORDER_MODEL_DIR ${CMAKE_CURRENT_BINARY_DIR}/input_order)
  if(NOT IS_DIRECTORY ${INPUTORDER_MODEL_DIR})
    file(MAKE_DIRECTORY ${INPUTORDER_MODEL_DIR})
    download_file(
      https://neo-ai-dlr-test-artifacts.s3-us-west-2.amazonaws.com/compiled-models/release-1.9.0/${INPUTORDER_MODEL}
      /tmp/${INPUTORDER_MODEL}
      SHA1
      e677130d4e47f0a5298f0280aa36a28ae9bb5ef1
    )
    # this is OS-agnostic
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar -xf /tmp/${INPUTORDER_MODEL}
                    WORKING_DIRECTORY ${INPUTORDER_MODEL_DIR})
    file(REMOVE /tmp/${INPUTORDER_MODEL})
  endif()

  set(INVERSELABEL_STATIC_MODEL inverselabel_static-ml_m4.tar.gz)
  set(INVERSELABEL_STATIC_MODEL_DIR ${CMAKE_CURRENT_BINARY_DIR}/inverselabel_static)
  if(NOT IS_DIRECTORY ${INVERSELABEL_STATIC_MODEL_DIR})
    file(MAKE_DIRECTORY ${INVERSELABEL_STATIC_MODEL_DIR})
    download_file(
      https://neo-ai-dlr-test-artifacts.s3-us-west-2.amazonaws.com/compiled-models/release-1.9.0/${INVERSELABEL_STATIC_MODEL}
      /tmp/${INVERSELABEL_STATIC_MODEL}
      SHA1
      bc4b0d958a18579362c4255741fdb4b86ab92aad
    )
    # this is OS-agnostic
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar -xf /tmp/${INVERSELABEL_STATIC_MODEL}
                    WORKING_DIRECTORY ${INVERSELABEL_STATIC_MODEL_DIR})
    file(REMOVE /tmp/${INVERSELABEL_STATIC_MODEL})
  endif()

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
