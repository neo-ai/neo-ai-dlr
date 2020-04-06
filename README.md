# DLR

DLR is a compact, common runtime for deep learning models and decision tree models compiled by [AWS SageMaker Neo](https://aws.amazon.com/sagemaker/neo/), [TVM](https://tvm.ai/), or [Treelite](https://treelite.readthedocs.io/en/latest/install.html). DLR uses the TVM runtime, Treelite runtime, NVIDIA TensorRT™, and can include other hardware-specific runtimes. DLR provides unified Python/C++ APIs for loading and running compiled models on various devices. DLR currently supports platforms from Intel, NVIDIA, and ARM, with support for Xilinx, Cadence, and Qualcomm coming soon.

## Installation
On X86_64 CPU targets running Linux, you can install latest release of DLR package via 

`pip install dlr`

For installation of DLR on GPU targets, non-x86 edge devices, or building DLR from source, please refer to [Installing DLR](https://neo-ai-dlr.readthedocs.io/en/latest/install.html)

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
    
    1. Create a config file, ccm_config.json inside your DLR target directory path, i.e. python3.6/site-packages/dlr/counter/ccm_config.json 
    2. Added below format content in it, 
      {       "ccm" : "false"      } 
    3. Restart DLR application. 
    4. Validate this feature is disabled by verifying this notification is no longer displayed, or programmatically with following command: 
        
        from dlr.counter.counter_mgr_lite import CounterMgrLite 
        CounterMgrLite.is_feature_enabled() # false if disabled 

## Examples
We prepared several examples demonstrating how to use DLR API on different platforms

* [Neo AI DLR image classification Android example application](examples/android/image_classification)
* [DL Model compiler for Android](examples/android/tvm_compiler)
* [DL Model compiler for AWS EC2 instances](container/ec2_compilation_container)

## License

This library is licensed under the Apache License Version 2.0. 
