
# C++ Tutorial

## Prerequisite

### Environment Setup

1. Install AWS C++ SDK https://docs.aws.amazon.com/sdk-for-cpp/v1/developer-guide/build-cmake.html

`
mkdir sdk_build
sudo cmake ../aws-sdk-cpp -D CMAKE_BUILD_TYPE=Release -D BUILD_ONLY="s3;iam;sagemaker" 
sudo make && make install
`

### Install JSON parser

* install through https://github.com/nlohmann/json#package-managers

## Documentation

* AWS C++ SDK - http://sdk.amazonaws.com/cpp/api/LATEST/index.html


copy *.so file to




### running command to generate model
`./tutorial compile`

### unzip compiler model inside the build repository
tar -zxvf ./compiled_model.tar.gz -c ./compiled_model

### running command to do inferencing
`./tutorial inference`


## Windows Installation

`cmake ..\aws-sdk-cpp -D CMAKE_BUILD_TYPE=Release -D BUILD_ONLY="s3;iam;sagemaker" -D ENABLE_TESTING=OFF`
`msbuild INSTALL.vcxproj /p:Configuration=Release`

cmake .. -DAWSSDK_DIR="C:\Program Files (x86)\aws-cpp-sdk-all"

mkdir compiled_model
tar zxvf ./compiled_model.tar.gz -C ./compled_model



### using the dev desktop for testing at the moment

<!-- image process alternative  -->
<!-- numpy alternative -->
https://www.boost.org/doc/libs/1_49_0/libs/numeric/ublas/doc/index.htm

https://xtensor.readthedocs.io/en/latest/numpy.html

https://code.amazon.com/packages/NeoLenovoBenchmark-2020-03-DLR/trees/heads/dev