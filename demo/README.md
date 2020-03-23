# Demo
This folder contains demo projects for different frontend languages that can be built and deployed on target platforms.

## CPP 
C++ demo source code that can be built into executables on Linux or Android platforms 

### Build 
Follow corresponding sections of [Installing DLR](https://neo-ai-dlr.readthedocs.io/en/latest/install.html) document to build the C library from source on Linux or Android. Then run 

`make demo`

to build executables.

### Running demo executables:
**Model_peeker**: a light-weight utility that prints out TVM model metadata.  
usage: 
`./model_peeker <model_dir> [device_type]`  
where device_type defaults to 'cpu'.

**Run_resnet**: a simple example that takes an image in the format of numpy array file (.npy), and outputs prediction result from typical image classification models like resnet or mobilenet.  
usage: 
`./run_resnet <model_dir> <ndarray file> [device_type] [input name]`  
where device_type defaults to "cpu", and input_name defaults to "data". 

## Python
Python demos coming soon.
