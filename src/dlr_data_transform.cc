#include "dlr_data_transform.h"

using namespace dlr;

bool DataTransform::HasInputTransform(const nlohmann::json& metadata) const {
  try {
    if (metadata.at("DataTransform").at("Input").count("ColumnTransform")) {
      return true;
    }
  } catch (nlohmann::json::out_of_range& e) {
    // ignore
  }
  return false;
}

bool DataTransform::HasOutputTransform(const nlohmann::json& metadata, int index) const {
  auto index_str = std::to_string(index);
  return metadata.count("DataTransform") && metadata["DataTransform"].count("Output") &&
         metadata["DataTransform"]["Output"].count(index_str) &&
         metadata["DataTransform"]["Output"][index_str].count("CategoricalString");
}

void DataTransform::TransformInput(const nlohmann::json& metadata, const int64_t* shape,
                                   const void* input, int dim,
                                   const std::vector<DLDataType>& dtypes, DLContext ctx,
                                   std::vector<tvm::runtime::NDArray>* tvm_inputs) const {
  nlohmann::json input_json = GetAsJson(shape, input, dim);
  const auto& transforms = metadata["DataTransform"]["Input"]["ColumnTransform"];
  for (int i = 0; i < transforms.size(); i++) {
    tvm_inputs->at(i) = InitNDArray(input_json, dtypes[i], ctx);

    const std::string& transformer_type = transforms[i]["Type"].get_ref<const std::string&>();
    auto it = GetTransformerMap()->find(transformer_type);
    CHECK(it != GetTransformerMap()->end())
        << transformer_type << " is not a valid DataTransform type.";
    const auto transformer = it->second;

    transformer->MapToNDArray(input_json, transforms[i], tvm_inputs->at(i));
  }
}

nlohmann::json DataTransform::GetAsJson(const int64_t* shape, const void* input, int dim) const {
  CHECK_EQ(dim, 1) << "String input must be 1-D vector.";
  // Interpret input as json
  const char* input_str = static_cast<const char*>(input);
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

tvm::runtime::NDArray DataTransform::InitNDArray(const nlohmann::json& input_json, DLDataType dtype,
                                                 DLContext ctx) const {
  // Create NDArray for transformed input which will be passed to TVM.
  std::vector<int64_t> arr_shape = {static_cast<int64_t>(input_json.size()),
                                    static_cast<int64_t>(input_json[0].size())};
  CHECK(dtype.code == kDLFloat && dtype.bits == 32 && dtype.lanes == 1)
      << "DataTransform CategoricalString is only supported for float32 inputs.";
  return tvm::runtime::NDArray::Empty(arr_shape, dtype, ctx);
}

void FloatTransformer::MapToNDArray(const nlohmann::json& input_json,
                                    const nlohmann::json& transform,
                                    tvm::runtime::NDArray& input_array) const {
  DLTensor* input_tensor = const_cast<DLTensor*>(input_array.operator->());
  CHECK_EQ(input_tensor->ctx.device_type, DLDeviceType::kDLCPU)
      << "DataTransform is only supported for CPU.";
  float* data = static_cast<float*>(input_tensor->data);
  for (size_t r = 0; r < input_json.size(); ++r) {
    CHECK_EQ(input_json[r].size(), input_json[0].size()) << "Inconsistent number of columns";
    for (size_t c = 0; c < input_json[r].size(); ++c) {
      const int out_index = r * input_json[r].size() + c;
      // Data is numeric, pass through. Attempt to convert string to float.
      try {
        data[out_index] = input_json[r][c].is_number()
                              ? input_json[r][c].get<float>()
                              : std::stof(input_json[r][c].get_ref<const std::string&>());
      } catch (const std::exception& ex) {
        // Any error will fallback safely to kBadValue.
        data[out_index] = kBadValue;
      }
    }
  }
}

void CategoricalStringTransformer::MapToNDArray(const nlohmann::json& input_json,
                                                const nlohmann::json& transform,
                                                tvm::runtime::NDArray& input_array) const {
  const nlohmann::json& mapping = transform["Map"];
  DLTensor* input_tensor = const_cast<DLTensor*>(input_array.operator->());
  // Writing directly to the DLTensor will only work for CPU context. For other contexts, we would
  // need to create an intermediate buffer on CPU and copy that to the context.
  CHECK_EQ(input_tensor->ctx.device_type, DLDeviceType::kDLCPU)
      << "DataTransform CategoricalString is only supported for CPU.";
  CHECK_EQ(input_json[0].size(), mapping.size())
      << "Input has " << input_json[0].size() << " columns, but model requires " << mapping.size();
  float* data = static_cast<float*>(input_tensor->data);
  // Copy data into data, mapping strings to float along the way.
  for (size_t r = 0; r < input_json.size(); ++r) {
    CHECK_EQ(input_json[r].size(), input_json[0].size()) << "Inconsistent number of columns";
    for (size_t c = 0; c < input_json[r].size(); ++c) {
      const int out_index = r * input_json[r].size() + c;
      // Look up in map. If not found, use kMissingValue.
      try {
        const std::string& data_str = input_json[r][c].get_ref<const std::string&>();
        auto it = mapping[c].find(data_str);
        data[out_index] = it != mapping[c].end() ? it->operator float() : kMissingValue;
      } catch (const std::exception& ex) {
        // Any error will fallback safely to kMissingValue.
        data[out_index] = kMissingValue;
      }
    }
  }
}

const std::shared_ptr<std::unordered_map<std::string, std::shared_ptr<Transformer>>>
DataTransform::GetTransformerMap() const {
  static auto map =
      std::make_shared<std::unordered_map<std::string, std::shared_ptr<Transformer>>>();
  if (!map->empty()) return map;
  map->emplace("Float", std::make_shared<FloatTransformer>());
  map->emplace("CategoricalString", std::make_shared<CategoricalStringTransformer>());
  return map;
}
