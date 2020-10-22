#include "dlr_data_transform.h"

using namespace dlr;

bool DataTransform::HasInputTransform(const nlohmann::json& metadata, int index) const {
  auto index_str = std::to_string(index);
  return metadata.count("DataTransform") && metadata["DataTransform"].count("Input") &&
         metadata["DataTransform"]["Input"].count(index_str) &&
         metadata["DataTransform"]["Input"][index_str].count("CategoricalString");
}

bool DataTransform::HasOutputTransform(const nlohmann::json& metadata, int index) const {
  auto index_str = std::to_string(index);
  return metadata.count("DataTransform") && metadata["DataTransform"].count("Output") &&
         metadata["DataTransform"]["Output"].count(index_str) &&
         metadata["DataTransform"]["Output"][index_str].count("CategoricalString");
}

tvm::runtime::NDArray DataTransform::TransformInput(const nlohmann::json& metadata, int index,
                                                    const int64_t* shape, void* input, int dim,
                                                    DLDataType dtype, DLContext ctx) const {
  auto& mapping = metadata["DataTransform"]["Input"][std::to_string(index)]["CategoricalString"];
  nlohmann::json input_json = GetAsJson(shape, input, dim);
  tvm::runtime::NDArray input_array = InitNDArray(index, input_json, mapping, dtype, ctx);
  MapToNDArray(input_json, input_array, mapping);
  return input_array;
}

nlohmann::json DataTransform::GetAsJson(const int64_t* shape, void* input, int dim) const {
  CHECK_EQ(dim, 1) << "String input must be 1-D vector.";
  // Interpret input as json
  const char* input_str = static_cast<char*>(input);
  nlohmann::json input_json;
  try {
    input_json = nlohmann::json::parse(input_str, input_str + shape[0]);
  } catch (nlohmann::json::parse_error& e) {
    LOG(ERROR) << "Invalid JSON input: " << e.what();
  }
  CHECK(input_json.is_array() && input_json.size() > 0 && input_json[0].is_array())
      << "Invalid JSON input: Must be 2-D array.";
  return input_json;
}

tvm::runtime::NDArray DataTransform::InitNDArray(int index, const nlohmann::json& input_json,
                                                 const nlohmann::json& mapping, DLDataType dtype,
                                                 DLContext ctx) const {
  // Create NDArray for transformed input which will be passed to TVM.
  std::vector<int64_t> arr_shape = {static_cast<int64_t>(input_json.size()),
                                    static_cast<int64_t>(input_json[0].size())};
  CHECK(dtype.code == kDLFloat && dtype.bits == 32 && dtype.lanes == 1)
      << "DataTransform CategoricalString is only supported for float32 inputs.";
  return tvm::runtime::NDArray::Empty(arr_shape, dtype, ctx);
}

void DataTransform::MapToNDArray(const nlohmann::json& input_json,
                                 tvm::runtime::NDArray& input_array,
                                 const nlohmann::json& mapping) const {
  DLTensor* input_tensor = const_cast<DLTensor*>(input_array.operator->());
  // Writing directly to the DLTensor will only work for CPU context. For other contexts, we would
  // need to create an intermediate buffer on CPU and copy that to the context.
  CHECK_EQ(input_tensor->ctx.device_type, DLDeviceType::kDLCPU)
      << "DataTransform CategoricalString is only supported for CPU.";
  float* data = static_cast<float*>(input_tensor->data);
  CHECK_EQ(input_json[0].size(), mapping.size())
      << "Number of columns should match, got " << input_json[0].size() << " but expected "
      << mapping.size();
  // Copy data into data, mapping strings to float along the way.
  for (size_t r = 0; r < input_json.size(); ++r) {
    CHECK_EQ(input_json[r].size(), mapping.size()) << "Inconsistent number of columns";
    for (size_t c = 0; c < input_json[r].size(); ++c) {
      if (mapping[c].size()) {
        // Look up in map. If not found, use kMissingValue.
        try {
          std::string data_str;
          input_json[r][c].get_to(data_str);
          auto it = mapping[c].find(data_str);
          data[r * input_json.size() + c] =
              it != mapping[c].end() ? it->operator float() : kMissingValue;
        } catch (const std::exception& ex) {
          // Any error will fallback safely to kMissingValue.
          data[r * input_json.size() + c] = kMissingValue;
        }
      } else {
        // Data is numeric, pass through. Attempt to convert string to float.
        try {
          data[r * input_json.size() + c] =
              input_json[r][c].is_number()
                  ? input_json[r][c].get<float>()
                  : std::stof(input_json[r][c].get_ref<const std::string&>());
        } catch (const std::exception& ex) {
          // Any error will fallback safely to kBadValue.
          data[r * input_json.size() + c] = kBadValue;
        }
      }
    }
  }
}
