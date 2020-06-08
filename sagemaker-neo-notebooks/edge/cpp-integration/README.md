
# C++ Tutorial

## Prerequisite

Please review the following prerequisites tasks before proceed with tutorial

### Prerequisite Software and Environment for Windows

1. In order to compile DLR and tutorial dependencies, please install [Visual Studio 2019](https://visualstudio.microsoft.com/vs/).
2. Setup msbuild path properly fo PowerShell.
   1. ppend your msbuild path, such as C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\MSBuild\Current\Bin, to your system path.

#### Install CMake

Please make sure CMake is installed prior start this tutorial. Both linux/windows distributions can be located here [https://cmake.org/download/]

### Install AWS C++ SDK

Please validate AWS C++ SDK is installed before starting this tutorial. In this sample tutorial, you'd need install AWS S3, IAM, and SageMaker service for Neo model compilation. For more information, please refer to [https://docs.aws.amazon.com/sdk-for-cpp/v1/developer-guide/welcome.html]. Installation guide can also be located [here](https://docs.aws.amazon.com/sdk-for-cpp/v1/developer-guide/setup.html). CMake instruction can be found [here](https://docs.aws.amazon.com/sdk-for-cpp/v1/developer-guide/build-cmake.html).

#### Install SDK on Linux

1. Checkout SDK from source `git clone https://github.com/aws/aws-sdk-cpp.git`
2. Create build directory for aws-sdk and install through CMake

``` shell
cd aws-sdk-cpp
mkdir build && cd build

# install through cmake
cmake .. \
    -DBUILD_ONLY="s3;iam;sagemaker" \
    -DCMAKE_BUILD_TYPE=Release \

# install the sdk
make
make install

#### Install SDK on Windows

1. Checkout SDK from source `git clone https://github.com/aws/aws-sdk-cpp.git`
2. Create build directory for aws-sdk and install through CMake. Installed binaries can be located in C:\Program Files (x86)\aws-cpp-sdk-all

``` shell
cd aws-sdk-cpp
mkdir build && cd build

# install through cmake
cmake ..\aws-sdk-cpp -D CMAKE_BUILD_TYPE=Release -D BUILD_ONLY="s3;iam;sagemaker" -D ENABLE_TESTING=OFF

# build/install the sdk
msbuild ALL_BUILD.vcxproj
msbuild INSTALL.vcxproj /p:Configuration=Release
```

### Install JSON Parser

For this tutorial, we'll use open source JSON parser "nlohmann/json" for handling IAM policy creation. For more information, please refer to [https://github.com/nlohmann/json]. We're going to build this from source; however you're more than welcome to install this through different approach through [https://github.com/nlohmann/json#package-managers].

#### Install nolhmann/json on Linux

For linux based instruction, please check the following

``` shell
# checkout project and make install
git clone https://github.com/nlohmann/json && \
    cd json && \
    cmake . && \
    make && \
    make install
```

#### Install nolhmann/json on Windows

``` shell
# checkout project
git clone https://github.com/nlohmann/json
cd json

# build/install
cmake .
msbuild ALL_BUILD.vcxproj
msbuild INSTALL.vcxproj /p:Configuration=Release
```

## Tutorial

In this tutorial, we're going to compile a pre-train model through Neo compiler, then using DLR runtime for inference.

### Linux Tutorial

#### Install DLR on Linux

Please follow this [link](https://neo-ai-dlr.readthedocs.io/en/latest/install.html#building-on-linux) to build DLR. Once done so, please locate libdlr.so in your build repository and copy to sagemaker-neo-notebooks/edge/windows/lib.


Then you'll need to link libdlr.so to sagemaker-neo-notebooks/edge/windows/lib.

``` shell
cd sagemaker-neo-notebooks/edge/windows/lib
ln -s neo-ai-dlr/build/lib/libdlr.so
```

#### Build tutorial program

```shell
# create a new build folder
cd sagemaker-neo-notebooks/edge/windows
mkdir build
cd build

# build through cmake
cmake .. -DCMAKE_BUILD_TYPE=Release -DDLR_HOME="/path/to/neo-ai-dlr"
make
```

#### Compile model on Linux

1. To see how compilation work, just perform the following

    ```shell
    cd sagemaker-neo-notebooks/edge/windows/build/

    #export your AWS access key
    export AWS_ACCESS_KEY_ID="<your AWS_ACCESS_KEY_ID>"
    export AWS_SECRET_ACCESS_KEY="<your AWS_SECRET_ACCESS_KEY>"
    export AWS_SESSION_TOKEN="<your AWS_SESSION_TOKEN>"

    # execute tutorial.exe to run Neo compiler
    ./tutorial compile ml_c4
    ```

2. Once done so, unzip the compiled models for inference.

   ``` shell
   mkdir compiled_model
   tar -zxvf ./compiled_model.tar.gz -C ./compled_model
   ```

3. To run the inference against the compiled model, just execute `./tutorial inference` to see inference against dog.npy

### Windows Tutorial

#### Install DLR on Windows

Please follow instruction to install DLR.

```
cd neo-ai-dlr
mkdir build
cd build
cmake .. -G"Visual Studio 16 2019" -A x64 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CONFIGURATION_TYPES=Release
```

Then open neo-ai-dlr\dlr.sln in Visual Studio and click Build -> Build Solution. You will see dlr.dll and dlr.lib in the neo-ai-dlr\build\lib directory.

#### Build tutorial to .exe

```shell
# create a new build folder
cd sagemaker-neo-notebooks/edge/windows
mkdir build
cd build

# build through cmake
cmake ../ \`
-DCMAKE_PREFIX_PATH="C:/Program Files (x86)/aws-cpp-sdk-all/lib/;C:/Program Files (x86)/aws-cpp-sdk-all/lib/cmake" \`
-DBUILD_SHARED_LIBS=ON \`
-DDLR_HOME="C:/Users/Administrator/Desktop/project/neo-ai-dlr"

# msbuild to release repo
msbuild .\tutorial.sln /p:Configuration=Release
```

After build, please locate and copy the following .dll into sagemaker-neo-notebooks/edge/windows/build/Release.

* aws-c-common.dll (should be inside C:\Program Files (x86)\aws-cpp-sdk-all\bin)
* aws-c-event-stream.dll
* aws-checksums.dll
* aws-cpp-sdk-core.dll
* aws-cpp-sdk-iam.dll
* aws-cpp-sdk-s3.dll
* aws-cpp-sdk-sagemaker.dll
* dlr.dll (inside neo-ai-dlr/build/lib)

#### Compile model on Windows

1. To see how compilation work, just perform the following

    ```shell
    cd sagemaker-neo-notebooks/edge/windows/build/Release

    # export AWS credentials
    $Env:AWS_ACCESS_KEY_ID="<YOUR AWS_ACCESS_KEY_ID>"
    $Env:AWS_SECRET_ACCESS_KEY="<YOUR AWS_SECRET_ACCESS_KEY>"
    $Env:AWS_SESSION_TOKEN="<YOUR AWS_SESSION_TOKEN>"

    # execute tutorial.exe to run Neo compiler
    & .\tutorial.exe compile x86_win64
    ```

2. Once done so, unzip the compiled models for inference.

   ``` shell
   mkdir compiled_model
   tar zxvf ./compiled_model.tar.gz -C ./compled_model
   ```

3. Copy the pre-generated data from sagemaker-neo-notebooks/edge/windows/data/dog.npy to sagemaker-neo-notebooks\edge\windows\build\Release.
4. To run the inference against the compiled model, just execute `& .\tutorial.exe inference` to see inference against dog.npy

## Notes

### Support Documents

* AWS C++ SDK  http://sdk.amazonaws.com/cpp/api/LATEST/index.html
* Xtensor https://xtensor.readthedocs.io/en/latest/numpy.html
* Numeric https://www.boost.org/doc/libs/1_49_0/libs/numeric/ublas/doc/index.htm