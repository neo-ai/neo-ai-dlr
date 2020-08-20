# CMakeLists.txt
cmake_minimum_required(VERSION 3.6)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)

project(tutorial)

# for debugging only
if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "-g")
endif()

# build json 
find_package(nlohmann_json 3.2.0 REQUIRED)
find_package(AWSSDK REQUIRED COMPONENTS s3 iam sagemaker)

if (WIN32)
    Message(STATUS "AWSSDK_LIB_DIR ${AWSSDK_LIB_DIR}")
    Message(STATUS "AWSSDK_BIN_DIR ${AWSSDK_BIN_DIR}")
    Message(STATUS "AWSSDK_INCLUDE_DIR ${AWSSDK_INCLUDE_DIR}")
    Message(STATUS "AWSSDK_PLATFORM_DEPS ${AWSSDK_PLATFORM_DEPS}")
    Message(STATUS "All AWSSDK TARGETS: ${AWSSDK_TARGETS}")
    Message(STATUS "All linked libs: ${AWSSDK_LINK_LIBRARIES}")
endif()

set(JSON_SRC "${DLR_HOME}/3rdparty/json")
set(LIBNPY_SRC "${DLR_HOME}/3rdparty/libnpy")

add_executable(tutorial tutorial.cc)

target_include_directories(tutorial PUBLIC ${JSON_SRC})
target_include_directories(tutorial PUBLIC ${LIBNPY_SRC})
target_include_directories(tutorial PUBLIC
    $<BUILD_INTERFACE:${DLR_HOME}/include>
    $<INSTALL_INTERFACE:include>)

target_compile_features(tutorial PRIVATE cxx_std_11)

# add dlr
add_library(dlr SHARED IMPORTED)

if (WIN32)
    target_link_libraries(tutorial PRIVATE ${AWSSDK_LINK_LIBRARIES} ${DLR_HOME}/build/lib/dlr.lib)
else ()
    set_target_properties(dlr PROPERTIES IMPORTED_LOCATION ${DLR_HOME}/build/lib/libdlr.so)
    target_link_libraries(tutorial PRIVATE ${AWSSDK_LINK_LIBRARIES} dlr)
endif()

