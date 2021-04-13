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

#ifndef __NEO_DLR_PLUGIN__
#define __NEO_DLR_PLUGIN__

#include <string>
#include <vector>

#include "NvInferPlugin.h"

using namespace nvinfer1;

class NeoDLRLayer : public nvinfer1::IPluginV2Ext {
 public:
  NeoDLRLayer(const std::string& model_path);
  NeoDLRLayer(const void* data, size_t length);
  ~NeoDLRLayer();

  // IPluginV2 methods
  const char* getPluginType() const noexcept override;
  const char* getPluginVersion() const noexcept override;
  int getNbOutputs() const noexcept override;
  Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept override;
  bool supportsFormat(DataType type, PluginFormat format) const noexcept override;
  size_t getWorkspaceSize(int maxBatchSize) const noexcept override;
  int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace,
              cudaStream_t stream) noexcept override;
  int initialize() noexcept override;
  void terminate() noexcept override;
  size_t getSerializationSize() const noexcept override;
  void serialize(void* buffer) const noexcept override;
  void destroy() noexcept override;
  void setPluginNamespace(const char* libNamespace) noexcept override;
  const char* getPluginNamespace() const noexcept override;

  // IPluginV2Ext methods
  nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputType,
                                       int nbInputs) const noexcept override;
  bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted,
                                    int nbInputs) const noexcept override;
  bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override;
  void configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
                       const DataType* inputTypes, const DataType* outputTypes,
                       const bool* inputIsBroadcast, const bool* outputIsBroadcast,
                       PluginFormat floatFormat, int maxBatchSize) noexcept override;
  IPluginV2Ext* clone() const noexcept override;

 private:
  std::string namespace_;
  std::string model_path_;
  // DLR info
  std::vector<std::string> input_names_;
  std::vector<std::vector<int64_t>> input_shapes_;
  std::vector<std::vector<int64_t>> output_shapes_;
  std::vector<std::string> output_types_;

  // TRT info
  // std::vector<nvinfer1::Dims> input_dims_;
  // std::vector<nvinfer1::DataType> input_dtypes_;
  // std::vector<nvinfer1::Dims> output_dims_;
  // std::vector<nvinfer1::DataType> output_dtypes_;

  // libdlr.so
  int LoadDLRInfo();
  void* libdlr_;

  typedef void* DLRModelHandle;
  DLRModelHandle model_;
  int (*CreateDLRModel)(DLRModelHandle* handle, const char* model_path, int dev_type, int dev_id);
  int (*DeleteDLRModel)(DLRModelHandle* handle);
  int (*RunDLRModel)(DLRModelHandle* handle);
  int (*GetDLRNumInputs)(DLRModelHandle* handle, int* num_inputs);
  int (*GetDLRNumWeights)(DLRModelHandle* handle, int* num_weights);
  int (*GetDLRInputName)(DLRModelHandle* handle, int index, const char** input_name);
  int (*GetDLRInputType)(DLRModelHandle* handle, int index, const char** input_type);
  int (*GetDLRWeightName)(DLRModelHandle* handle, int index, const char** weight_name);
  int (*SetDLRInput)(DLRModelHandle* handle, const char* name, const int64_t* shape,
                     const void* input, int dim);
  int (*SetDLRInputTensor)(DLRModelHandle* handle, const char* name, void* tensor);
  int (*GetDLRInput)(DLRModelHandle* handle, const char* name, void* input);
  int (*GetDLRInputShape)(DLRModelHandle* handle, int index, int64_t* shape);
  int (*GetDLRInputSizeDim)(DLRModelHandle* handle, int index, int64_t* size, int* dim);
  int (*GetDLROutputShape)(DLRModelHandle* handle, int index, int64_t* shape);
  int (*GetDLROutput)(DLRModelHandle* handle, int index, void* out);
  int (*GetDLROutputPtr)(DLRModelHandle* handle, int index, const void** out);
  int (*GetDLROutputTensor)(DLRModelHandle* handle, int index, void* tensor);
  int (*GetDLRNumOutputs)(DLRModelHandle* handle, int* num_outputs);
  int (*GetDLROutputSizeDim)(DLRModelHandle* handle, int index, int64_t* size, int* dim);
  int (*GetDLROutputType)(DLRModelHandle* handle, int index, const char** output_type);
  int (*GetDLRHasMetadata)(DLRModelHandle* handle, bool* has_metadata);
  int (*GetDLROutputName)(DLRModelHandle* handle, const int index, const char** name);
  int (*GetDLROutputIndex)(DLRModelHandle* handle, const char* name, int* index);
  int (*GetDLROutputByName)(DLRModelHandle* handle, const char* name, void* out);
  const char* (*DLRGetLastError)();
  int (*GetDLRBackend)(DLRModelHandle* handle, const char** name);
  int (*GetDLRDeviceType)(const char* model_path);
};

#endif  // __NEO_DLR_PLUGIN__
