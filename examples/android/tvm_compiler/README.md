# Compile Deep Learning Models for Android

## Overview
This folder contain Dockerfile and scripts to compile different Deep Learning Models for Android platform.
Currently we support the following Deep Learning Frameworks:
* GluonCV
* Keras
* Tensorflow

## Prepare Docker Container
You can build `tvm_android_compiler` docker image yourself or pull pre-built image

To build docker image yourself run the following
```
docker build -t tvm_android_compiler .

docker run -ti tvm_android_compiler
```

To use pre-built Docker Image run
```
docker run -ti x4444/tvm_android_compiler
```

## Compile Deep Learning Models
Once the Docker container is started you can compile Deep Learning Models for Android

### Supported Android Architectures
The compilation scripts support the following Android architectures:
* arm64-v8a
* armeabi-v7a
* x86_64
* x86

The compilation scripts compile the models for all supported Android architectures.
Output files fill be saved under folders `<arch>/<dlr_model_name>`.
Each folder will have three files:
* model.json
* model.so
* model.params

Compiled models can be run using `neo-ai-dlr` API/lib.
See [DLR Android examples](https://github.com/neo-ai/neo-ai-dlr/tree/main/examples/android)

### GluonCV
To compile GluonCV models use script `compile_gluoncv.py`.
By default the script downloads pre-trained `mobilenetv2_0.75` model from GluonCV Model Zoo.
Edit the script to change GluonCV Model Zoo model.

To compile GluonCV model run the following:
```
./compile_gluoncv.py
```

### Keras
To compile Keras models use script `compile_keras.py`.
By default the script downloads pre-trained `MobileNet_v2` model from Keras applications.
Edit the script to change Keras Application model.

To compile Keras model run the following:
```
./compile_keras.py
```

### Tensorflow
To compile Tensorflow models use script `compile_tensorflow.py`.
By default the script compiles frozen model from [mobilenet_v1_1.0_224.tgz](http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz).
Download tgz file and extract it to some folder. Copy or link `mobilenet_v1_1.0_224_frozen.pb` file to `~/tvm_compiler` folder.

To compile Tensorflow model run the following:
```
mkdir /tmp/mobilenet_v1_1.0_224
cd /tmp/mobilenet_v1_1.0_224
wget http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz
tar zxf mobilenet_v1_1.0_224.tgz
rm mobilenet_v1_1.0_224.tgz
cd ~/tvm_compiler
ln -s /tmp/mobilenet_v1_1.0_224/mobilenet_v1_1.0_224_frozen.pb

./compile_tensorflow.py
```
