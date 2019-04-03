# DLR inference container

This directory contains Dockerfile and other files needed to build DLR inference containers.

## How to build
* XGBoost container: Handle requests containing CSV or LIBSVM format. Suitable for serving XGBoost models.
```
docker build --build-arg APP=xgboost -t xgboost-cpu -f Dockerfile.cpu .
```
* Image Classification container: Handle requests containing JPEG or PNG format. Suitable for serving image classifiers produced by the [SageMaker Image Classification algorithm](https://docs.aws.amazon.com/sagemaker/latest/dg/image-classification.html).
```
docker build --build-arg APP=image_classification -t ic-cpu -f Dockerfile.cpu .
```
* Bring Your Own Model (BYOM): coming soon
