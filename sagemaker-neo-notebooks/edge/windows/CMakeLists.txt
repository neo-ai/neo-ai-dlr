# export CC=/usr/bin/gcc
# export CXX=/usr/bin/g++


# CMakeLists.txt
cmake_minimum_required(VERSION 3.2)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)

project(tutorial)

set(CMAKE_CXX_FLAGS "-g")

# set(XTENSOR_USE_XSIMD 1)
# set(XTENSOR_USE_TBB 1)


# build json 
find_package(nlohmann_json 3.2.0 REQUIRED)
find_package(AWSSDK REQUIRED COMPONENTS s3 iam sagemaker)
find_package(xtensor REQUIRED)

add_executable(tutorial windows.cc)

target_include_directories(tutorial PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

target_include_directories(tutorial PUBLIC ${xtensor_INCLUDE_DIRS})


target_compile_features(tutorial PRIVATE cxx_std_11)

# add dlr
add_library(dlr SHARED IMPORTED)
# set_target_properties(dlr PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/lib/libdlr.so)
set_target_properties(dlr PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/lib/libdlr.dylib)



target_link_libraries(tutorial PRIVATE nlohmann_json::nlohmann_json ${AWSSDK_LINK_LIBRARIES} dlr)
target_link_libraries(tutorial PUBLIC xtensor)

# work on linux 
# target_link_libraries(tutorial PRIVATE ${AWSSDK_DIR} nlohmann_json::nlohmann_json)
# target_link_libraries(tutorial ${AWSSDK_LINK_LIBRARIES})

# export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH;/Users/wuchih/Documents/rde-workspace/aws-sdk/skd_build
# /Users/wuchih/Documents/rde-workspace/aws-sdk/aws-sdk-cpp
# cmake -DAWSSDK_DIR=/Users/wuchih/Documents/rde-workspace/aws-sdk/skd_build ..

# cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/aws-sdk
# export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH;/opt/aws-sdk
# cmake -DAWSSDK_DIR=/opt/aws-sdk ..


# https://github.com/apivovarov/lambda-dlr-tf-c-api