# DLR inference container

This directory contains Dockerfile and other files needed to build DLR inference containers. The containers make use of [MXNet Model Server](https://github.com/awslabs/mxnet-model-server) to serve HTTP requests.

## How to build
* XGBoost container: Handle requests containing CSV or LIBSVM format. Suitable for serving XGBoost models.
```
docker build --build-arg APP=xgboost -t xgboost-cpu -f Dockerfile.cpu .
```
* Image Classification container: Handle requests containing JPEG or PNG format. Suitable for serving image classifiers produced by the [SageMaker Image Classification algorithm](https://docs.aws.amazon.com/sagemaker/latest/dg/image-classification.html).
  - Build for CPU target
  ```
  docker build --build-arg APP=image_classification -t ic-cpu -f Dockerfile.cpu .
  ```
  - Build for GPU target: First download `TensorRT-5.1.5.0.Ubuntu-18.04.2.x86_64-gnu.cuda-10.0.cudnn7.5.tar.gz` from NVIDIA into the directory `neo-ai-dlr/container/`. Then run
  ```
  docker build --build-arg APP=image_classification -t ic-gpu -f Dockerfile.gpu .
  ```
* MXNet BYOM (Bring Your Own Model): Handle requests of any form. See [this example notebook](https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/mxnet_mnist/mxnet_mnist_neo.ipynb) for more details.
  - Build for CPU target
  ```
  docker build --build-arg APP=mxnet_byom -t mxnet-byom-cpu -f Dockerfile.cpu .
  ```
  - Build for GPU target: First download `TensorRT-5.1.5.0.Ubuntu-18.04.2.x86_64-gnu.cuda-10.0.cudnn7.5.tar.gz` from NVIDIA into the directory `neo-ai-dlr/container/`. Then run
  ```
  docker build --build-arg APP=mxnet_byom -t mxnet-byom-gpu -f Dockerfile.gpu .
  ```
