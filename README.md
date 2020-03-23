# DLR

DLR is a compact, common runtime for deep learning models and decision tree models compiled by [AWS SageMaker Neo](https://aws.amazon.com/sagemaker/neo/), [TVM](https://tvm.ai/), or [Treelite](https://treelite.readthedocs.io/en/latest/install.html). DLR uses the TVM runtime, Treelite runtime, NVIDIA TensorRT™, and can include other hardware-specific runtimes. DLR provides unified Python/C++ APIs for loading and running compiled models on various devices. DLR currently supports platforms from Intel, NVIDIA, and ARM, with support for Xilinx, Cadence, and Qualcomm coming soon.

## Installation
On X86_64 CPU targets running Linux, you can install latest release of DLR package via 

`pip install dlr`

For installation of DLR on GPU targets, non-x86 edge devices, or building DLR from source, please refer to [Installing DLR](https://neo-ai-dlr.readthedocs.io/en/latest/install.html)

## Documentation
For instructions on using DLR, please refer to [Amazon SageMaker Neo – Train Your Machine Learning Models Once, Run Them Anywhere](https://aws.amazon.com/blogs/aws/amazon-sagemaker-neo-train-your-machine-learning-models-once-run-them-anywhere/)

Also check out the [API documentation](https://neo-ai-dlr.readthedocs.io/en/latest/)

## Examples
We prepared several examples demonstrating how to use DLR API on different platforms

* [Neo AI DLR image classification Android example application](examples/android/image_classification)
* [DL Model compiler for Android](examples/android/tvm_compiler)
* [DL Model compiler for AWS EC2 instances](container/ec2_compilation_container)

## License

This library is licensed under the Apache License Version 2.0. 
