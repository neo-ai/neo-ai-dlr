# DLR

DLR is a compact, common runtime for deep learning models and decision tree models compiled by [AWS SageMaker Neo](https://aws.amazon.com/sagemaker/neo/), [TVM](https://github.com/neo-ai/tvm), or [Treelite](https://treelite.readthedocs.io/en/latest/install.html). DLR uses the TVM runtime, Treelite runtime, NVIDIA TensorRT™, and can include other hardware-specific runtimes. DLR provides unified Python/C++ APIs for loading and running compiled models on various devices. DLR currently supports platforms from Intel, NVIDIA, and ARM, with support for Xilinx, Cadence, and Qualcomm coming soon.

## Installation
On x86_64 CPU targets running Linux, you can install latest release of DLR package via 

`pip install dlr`

For installation of DLR on GPU targets or non-x86 edge devices, please refer to [Releases](https://github.com/neo-ai/neo-ai-dlr/releases) for prebuilt binaries, or [Installing DLR](https://neo-ai-dlr.readthedocs.io/en/latest/install.html) for building DLR from source.

## Usage

```python

import dlr
import numpy as np

# Load model.
# /path/to/model is a directory containing the compiled model artifacts (.so, .params, .json)
model = dlr.DLRModel('/path/to/model', 'cpu', 0)

# Prepare some input data.
x = np.random.rand(1, 3, 224, 224)

# Run inference.
y = model.run(x)

```

## Release compatibility with different versions of TVM

Each release of DLR is capable of executing models compiled with the same corresponding release of [neo-ai/tvm](https://github.com/neo-ai/tvm). For example, if you used the [release-1.2.0 branch of neo-ai/tvm](https://github.com/neo-ai/tvm/tree/release-1.2.0) to compile your model, then you should use the [release-1.2.0 branch of neo-ai/neo-ai-dlr](https://github.com/neo-ai/neo-ai-dlr/tree/release-1.2.0) to execute the compiled model. Please see [DLR Releases](https://github.com/neo-ai/neo-ai-dlr/releases) for more information.

## Documentation
For instructions on using DLR, please refer to [Amazon SageMaker Neo – Train Your Machine Learning Models Once, Run Them Anywhere](https://aws.amazon.com/blogs/aws/amazon-sagemaker-neo-train-your-machine-learning-models-once-run-them-anywhere/)

Also check out the [API documentation](https://neo-ai-dlr.readthedocs.io/en/latest/)

### Call Home Feature

You acknowledge and agree that DLR collects the following metrics to help improve its performance. 
By default, Amazon will collect and store the following information from your device: 

    record_type: <enum, internal record status, such as model_loaded, model_>, 
    arch: <string, platform architecture, eg 64bit>, 
    osname: <string, platform os name, eg. Linux>, 
    uuid: <string, one-way non-identifable hashed mac address, eg. 8fb35b79f7c7aa2f86afbcb231b1ba6e>, 
    dist: <string, distribution of os, eg. Ubuntu 16.04 xenial>, 
    machine: <string, retuns the machine type, eg. x86_64 or i386>, 
    model: <string, one-way non-identifable hashed model name, eg. 36f613e00f707dbe53a64b1d9625ae7d> 

If you wish to opt-out of this data collection feature, please follow the steps below: 
    
    1. Disable through code
      ``` 
      from dlr.counter.phone_home import PhoneHome
      PhoneHome.disable_feature()
      ```
    2. Or, create a config file, ccm_config.json inside your DLR target directory path, i.e. python3.6/site-packages/dlr/counter/ccm_config.json, then add below format content in it, ```{ "enable_phone_home" : false } ``` 
    3. Restart DLR application. 
    4. Validate this feature is disabled by verifying this notification is no longer displayed, or programmatically with following command: 
        ```
        from dlr.counter.phone_home import PhoneHome 
        PhoneHome.is_enabled() # false if disabled 
        ```

## Examples
We prepared several examples demonstrating how to use DLR API on different platforms

* [Neo AI DLR image classification Android example application](https://github.com/neo-ai/neo-ai-dlr/tree/main/examples/android/image_classification)
* [DL Model compiler for Android](https://github.com/neo-ai/neo-ai-dlr/tree/main/examples/android/tvm_compiler)
* [DL Model compiler for AWS EC2 instances](https://github.com/neo-ai/neo-ai-dlr/tree/main/container/ec2_compilation_container)

## License

This library is licensed under the Apache License Version 2.0. 
