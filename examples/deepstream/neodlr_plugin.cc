/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "neodlr_plugin.h"

#include <dlfcn.h>
#include <dlpack/dlpack.h>

#include <algorithm>
#include <iostream>
#include <memory>

#include "NvInfer.h"

namespace {
const char* NEODLR_PLUGIN_VERSION{"1"};
const char* NEODLR_PLUGIN_NAME{"NeoDLR_TRT"};
}  // namespace

Dims ToTrtDims(const std::vector<int64_t>& vec) {
  Dims ret;
  ret.nbDims = vec.size() - 1;
  ret.d[0] = 0;
  for (int i = 1; i < vec.size(); ++i) ret.d[i - 1] = vec[i];
  return ret;
}

DataType ToTrtDataType(const std::string& dtype) {
  if (dtype == "float32") return DataType::kFLOAT;
  exit(-1);
}

using namespace nvinfer1;

NeoDLRLayer::NeoDLRLayer(const std::string& model_path) : model_path_(model_path) { LoadDLRInfo(); }

NeoDLRLayer::NeoDLRLayer(const void* data, size_t length) : model_path_((char*)data, length) {
  LoadDLRInfo();
}

NeoDLRLayer::~NeoDLRLayer() {
  DeleteDLRModel(&model_);
  dlclose(libdlr_);
}

int NeoDLRLayer::getNbOutputs() const noexcept { return output_shapes_.size(); }

int NeoDLRLayer::initialize() noexcept { return 0; }

void NeoDLRLayer::terminate() noexcept {}

Dims NeoDLRLayer::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept {
  return ToTrtDims(output_shapes_[index]);
}

size_t NeoDLRLayer::getWorkspaceSize(int maxBatchSize) const noexcept { return 0; }

int NeoDLRLayer::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace,
                         cudaStream_t stream) noexcept {
  // Set inputs.
  for (size_t i = 0; i < input_names_.size(); ++i) {
    DLTensor dltensor;
    dltensor.ctx = {kDLGPU, 0};
    dltensor.ndim = input_shapes_[i].size();
    dltensor.shape = input_shapes_[i].data();
    dltensor.strides = 0;
    dltensor.byte_offset = 0;
    dltensor.dtype = {kDLFloat, 32, 1};  // todo allow other types
    dltensor.data = const_cast<void*>(inputs[i]);
    if (SetDLRInputTensor(&model_, input_names_[i].c_str(), &dltensor) != 0) {
      std::cout << DLRGetLastError() << std::endl;
      return -1;
    }
  }
  // Run.
  if (RunDLRModel(&model_) != 0) {
    std::cout << DLRGetLastError() << std::endl;
    return -1;
  }
  // Get outputs.
  for (size_t i = 0; i < output_shapes_.size(); ++i) {
    DLTensor dltensor;
    dltensor.ctx = {kDLGPU, 0};
    dltensor.ndim = output_shapes_[i].size();
    dltensor.shape = output_shapes_[i].data();
    dltensor.strides = 0;
    dltensor.byte_offset = 0;
    dltensor.dtype = {kDLFloat, 32, 1};  // todo allow other types
    dltensor.data = outputs[i];
    if (GetDLROutputTensor(&model_, i, &dltensor) != 0) {
      std::cout << DLRGetLastError() << std::endl;
    }
  }
  return 0;
}

size_t NeoDLRLayer::getSerializationSize() const noexcept { return model_path_.size(); }

void NeoDLRLayer::serialize(void* buffer) const noexcept {
  char* d = reinterpret_cast<char*>(buffer);
  model_path_.copy(d, model_path_.size(), 0);
}

void NeoDLRLayer::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims,
                                  int nbOutputs, const DataType* inputTypes,
                                  const DataType* outputTypes, const bool* inputIsBroadcast,
                                  const bool* outputIsBroadcast, nvinfer1::PluginFormat format,
                                  int maxBatchSize) noexcept {}

bool NeoDLRLayer::supportsFormat(DataType type, PluginFormat format) const noexcept { return true; }

const char* NeoDLRLayer::getPluginType() const noexcept { return NEODLR_PLUGIN_NAME; }

const char* NeoDLRLayer::getPluginVersion() const noexcept { return NEODLR_PLUGIN_VERSION; }

void NeoDLRLayer::destroy() noexcept { delete this; }

IPluginV2Ext* NeoDLRLayer::clone() const noexcept { return new NeoDLRLayer(model_path_); }

void NeoDLRLayer::setPluginNamespace(const char* pluginNamespace) noexcept {
  namespace_ = pluginNamespace;
}

const char* NeoDLRLayer::getPluginNamespace() const noexcept { return namespace_.c_str(); }

nvinfer1::DataType NeoDLRLayer::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
                                                  int nbInputs) const noexcept {
  return ToTrtDataType(output_types_[index]);
}

bool NeoDLRLayer::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted,
                                               int nbInputs) const noexcept {
  return false;
}

bool NeoDLRLayer::canBroadcastInputAcrossBatch(int inputIndex) const noexcept { return false; }

int NeoDLRLayer::LoadDLRInfo() {
  std::string dlr_lib_path = model_path_ + "/libdlr.so";
  libdlr_ = dlopen(dlr_lib_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
  if (libdlr_ == nullptr) {
    std::cout << "Could not load libdlr.so: " << dlerror();
    return -1;
  }
  // Load functions from libdlr.so
  *(void**)(&CreateDLRModel) = dlsym(libdlr_, "CreateDLRModel");
  *(void**)(&DeleteDLRModel) = dlsym(libdlr_, "DeleteDLRModel");
  *(void**)(&RunDLRModel) = dlsym(libdlr_, "RunDLRModel");
  *(void**)(&GetDLRNumInputs) = dlsym(libdlr_, "GetDLRNumInputs");
  *(void**)(&GetDLRNumWeights) = dlsym(libdlr_, "GetDLRNumWeights");
  *(void**)(&GetDLRInputName) = dlsym(libdlr_, "GetDLRInputName");
  *(void**)(&GetDLRInputType) = dlsym(libdlr_, "GetDLRInputType");
  *(void**)(&GetDLRWeightName) = dlsym(libdlr_, "GetDLRWeightName");
  *(void**)(&SetDLRInput) = dlsym(libdlr_, "SetDLRInput");
  *(void**)(&SetDLRInputTensor) = dlsym(libdlr_, "SetDLRInputTensor");
  *(void**)(&GetDLRInput) = dlsym(libdlr_, "GetDLRInput");
  *(void**)(&GetDLRInputShape) = dlsym(libdlr_, "GetDLRInputShape");
  *(void**)(&GetDLRInputSizeDim) = dlsym(libdlr_, "GetDLRInputSizeDim");
  *(void**)(&GetDLROutputShape) = dlsym(libdlr_, "GetDLROutputShape");
  *(void**)(&GetDLROutput) = dlsym(libdlr_, "GetDLROutput");
  *(void**)(&GetDLROutputPtr) = dlsym(libdlr_, "GetDLROutputPtr");
  *(void**)(&GetDLROutputTensor) = dlsym(libdlr_, "GetDLROutputTensor");
  *(void**)(&GetDLRNumOutputs) = dlsym(libdlr_, "GetDLRNumOutputs");
  *(void**)(&GetDLROutputSizeDim) = dlsym(libdlr_, "GetDLROutputSizeDim");
  *(void**)(&GetDLROutputType) = dlsym(libdlr_, "GetDLROutputType");
  *(void**)(&GetDLRHasMetadata) = dlsym(libdlr_, "GetDLRHasMetadata");
  *(void**)(&GetDLROutputName) = dlsym(libdlr_, "GetDLROutputName");
  *(void**)(&GetDLROutputIndex) = dlsym(libdlr_, "GetDLROutputIndex");
  *(void**)(&GetDLROutputByName) = dlsym(libdlr_, "GetDLROutputByName");
  *(void**)(&DLRGetLastError) = dlsym(libdlr_, "DLRGetLastError");
  *(void**)(&GetDLRBackend) = dlsym(libdlr_, "GetDLRBackend");
  *(void**)(&GetDLRDeviceType) = dlsym(libdlr_, "GetDLRDeviceType");

  // Load model
  int dev_type = GetDLRDeviceType(model_path_.c_str());
  if (dev_type == -1) {
    // Device type couldn't be read from model. Assume it is GPU.
    dev_type = kDLGPU;
  }
  if (CreateDLRModel(&model_, model_path_.c_str(), dev_type, 0) != 0) {
    std::cout << DLRGetLastError() << std::endl;
    return -1;
  }

  // Read model info
  // Inputs
  int num_inputs = 0;
  if (GetDLRNumInputs(&model_, &num_inputs) != 0) {
    std::cout << DLRGetLastError() << std::endl;
    return -1;
  }
  for (int i = 0; i < num_inputs; ++i) {
    // Get name
    const char* input_name;
    GetDLRInputName(&model_, i, &input_name);
    input_names_.emplace_back(input_name);
    // Get shape
    int64_t input_size;
    int input_dim;
    GetDLRInputSizeDim(&model_, i, &input_size, &input_dim);
    std::vector<int64_t> shape(input_dim);
    GetDLRInputShape(&model_, i, shape.data());
    input_shapes_.emplace_back(shape);
  }
  // Outputs
  // TODO: Check if dynamic
  int num_outputs = 0;
  if (GetDLRNumOutputs(&model_, &num_outputs) != 0) {
    std::cout << DLRGetLastError() << std::endl;
    return -1;
  }
  for (int i = 0; i < num_outputs; ++i) {
    // Get shape
    int64_t output_size;
    int output_dim;
    GetDLROutputSizeDim(&model_, i, &output_size, &output_dim);
    std::vector<int64_t> shape(output_dim);
    GetDLROutputShape(&model_, i, shape.data());
    output_shapes_.emplace_back(shape);
    // Type
    const char* dtype;
    GetDLROutputType(&model_, i, &dtype);
    output_types_.emplace_back(dtype);
  }
  return 0;
}
