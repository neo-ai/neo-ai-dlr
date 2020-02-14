# DLR inference container

This directory contains Dockerfile and other files needed to build DLR inference containers. The containers make use of [MXNet Model Server](https://github.com/awslabs/mxnet-model-server) to serve HTTP requests.

## How to build
* XGBoost container: Handle requests containing CSV or LIBSVM format. Suitable for serving XGBoost models.
```bash
# Run the following command at the root directory of the neo-ai-dlr repository
docker build --build-arg APP=xgboost -t xgboost-cpu -f container/Dockerfile.cpu .
```
* Image Classification container: Handle requests containing JPEG or PNG format. Suitable for serving image classifiers produced by the [SageMaker Image Classification algorithm](https://docs.aws.amazon.com/sagemaker/latest/dg/image-classification.html).
  - Build for CPU target
  ```
  # Run the following command at the root directory of the neo-ai-dlr repository
  docker build --build-arg APP=image_classification -t ic-cpu -f container/Dockerfile.cpu .
  ```
  - Build for GPU target: First download `TensorRT-7.0.0.11.Ubuntu-18.04.x86_64-gnu.cuda-10.0.cudnn7.6.tar.gz` from NVIDIA into the directory `neo-ai-dlr/container/`. Then run
  ```
  # Run the following command at the root directory of the neo-ai-dlr repository
  docker build --build-arg APP=image_classification -t ic-gpu -f container/Dockerfile.gpu .
  ```
* MXNet BYOM (Bring Your Own Model): Handle requests of any form. See [this example notebook](https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/mxnet_mnist/mxnet_mnist_neo.ipynb) for more details.
  - Build for CPU target
  ```
  # Run the following command at the root directory of the neo-ai-dlr repository
  docker build --build-arg APP=mxnet_byom -t mxnet-byom-cpu -f container/Dockerfile.cpu .
  ```
  - Build for GPU target: First download `TensorRT-7.0.0.11.Ubuntu-18.04.x86_64-gnu.cuda-10.0.cudnn7.6.tar.gz` from NVIDIA into the directory `neo-ai-dlr/container/`. Then run
  ```
  # Run the following command at the root directory of the neo-ai-dlr repository
  docker build --build-arg APP=mxnet_byom -t mxnet-byom-gpu -f container/Dockerfile.gpu .
  ```

## How to test container locally
The following command runs `xgboost-cpu:latest` container locally. You can run other containers by replacing `xgboost-cpu` with the appropriate tag. 
```bash
docker run -v ${PWD}/model:/opt/ml/model -v ${PWD}/errors:/opt/ml/errors -p 127.0.0.1:8888:8080/tcp \
    xgboost-cpu:latest serve
```
Once the serving container finishes initializing, you can send HTTP requests to the URL `http://localhost:8888/invocations`:
```python
payload = ('106,0,274.4,120,198.6,82,160.8,62,6.0,3,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,'
  + '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,0')

r = requests.post('http://localhost:8888/invocations',
                  data=payload.encode('utf-8'),
                  headers={'Content-type': 'text/csv'})
print(r.status_code)   # should print 200 for successful response
print(r.text)          # prints response content
```

Non-200 responses indicate an error. To investigate the root cause of an error, you can look at the `errors.log` file under the mounted `errors/` directory.

When you are done, stop the running container with the command
```bash
docker stop $(docker ps -a -q --filter ancestor=xgboost-cpu:latest)
```
