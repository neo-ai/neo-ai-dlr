# Compile Deep Learning Models for EC2

## Overview
This folder contain Dockerfile and scripts to compile different Deep Learning Models for EC2.
Currently we support the following Deep Learning Frameworks:
* GluonCV
* Keras
* Tensorflow

## Prepare Docker Container
You can build `ec2_compilation_container` docker image yourself or pull pre-built image

To build docker image yourself run the following
```
docker build -t ec2_compilation_container .

docker run -ti ec2_compilation_container
```

To use pre-built Docker Image run
```
docker run -ti x4444/ec2_compilation_container
```

## Compile Deep Learning Models
Once the Docker container is started you can compile Deep Learning Models for EC2

### Supported EC2 types
The compilation scripts support the following EC2 types:
* c4, m4
* c5, m5
* p3, ml_p3
* p2, ml_p2
* lambda

The compilation scripts compile the models for all supported EC2 architectures.
Output files fill be saved under folders `<arch>/<dlr_model_name>`.
Each folder will have three files:
* model.json
* model.so
* model.params

Compiled models can be run using `neo-ai-dlr` API/lib.
See [DLR EC2 examples](https://github.com/neo-ai/neo-ai-dlr/tree/master/examples/ec2)

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
Download tgz file and extract it to some folder. Copy or link `mobilenet_v1_1.0_224_frozen.pb` file to `~/ec2_compilation_container` folder.

To compile Tensorflow model run the following:
```
mkdir /tmp/mobilenet_v1_1.0_224
cd /tmp/mobilenet_v1_1.0_224
wget http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz
tar zxf mobilenet_v1_1.0_224.tgz
rm mobilenet_v1_1.0_224.tgz
cd ~/ec2_compilation_container
ln -s /tmp/mobilenet_v1_1.0_224/mobilenet_v1_1.0_224_frozen.pb

./compile_tensorflow.py
```
