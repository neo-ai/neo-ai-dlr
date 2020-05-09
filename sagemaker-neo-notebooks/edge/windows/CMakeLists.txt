# CMakeLists.txt
cmake_minimum_required(VERSION 3.2)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)

project(tutorial)


# build json 
find_package(nlohmann_json 3.2.0 REQUIRED)
find_package(AWSSDK REQUIRED COMPONENTS s3 iam sagemaker)


add_executable(tutorial windows.cc)
target_compile_features(tutorial PRIVATE cxx_std_11)


target_link_libraries(tutorial PRIVATE nlohmann_json::nlohmann_json ${AWSSDK_LINK_LIBRARIES})
# target_link_libraries(tutorial ${AWSSDK_LINK_LIBRARIES})





# export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH;/Users/wuchih/Documents/rde-workspace/aws-sdk/skd_build
# /Users/wuchih/Documents/rde-workspace/aws-sdk/aws-sdk-cpp
# cmake -DAWSSDK_DIR=/Users/wuchih/Documents/rde-workspace/aws-sdk/skd_build ..