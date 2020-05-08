
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