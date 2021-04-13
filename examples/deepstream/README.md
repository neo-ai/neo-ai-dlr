# Deepstream DLR Plugin

This example shows how a DLR model can be used as part of an NVIDIA deepstream pipeline.


## Usage

1. Configure the inputs in `nvdsinfer_custom_impl_neodlr.cpp` to match your model and update the path to your neo compiled model `auto* dlr_plugin = new NeoDLRLayer("path/to/neo/compiled/model");`.

2. Update paths to CUDA, TensorRT, and Deepstream directories in `Makefile`.

3. `make`

4. You can now use `config_infer_primary.txt` as a stage in your deepstream app config. See `deepstream_app_config.txt` for an example deepstream app.