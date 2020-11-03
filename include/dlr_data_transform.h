#ifndef DLR_DATA_TRANSFORM_H_
#define DLR_DATA_TRANSFORM_H_

#include <tvm/runtime/ndarray.h>

#include <nlohmann/json.hpp>

#include "dlr_common.h"

namespace dlr {

/*! \brief Base case for input transformers. */
class DLR_DLL Transformer {
 public:
  virtual void MapToNDArray(const nlohmann::json& input_json, const nlohmann::json& transform,
                            tvm::runtime::NDArray& input_array) const = 0;
};

class DLR_DLL FloatTransformer : public Transformer {
 private:
  /*! \brief When there is a value stof cannot convert to float, this value is used. */
  const float kBadValue = std::numeric_limits<float>::quiet_NaN();

 public:
  void MapToNDArray(const nlohmann::json& input_json, const nlohmann::json& transform,
                    tvm::runtime::NDArray& input_array) const;
};

class DLR_DLL CategoricalStringTransformer : public Transformer {
 private:
  /*! \brief When there is no mapping entry for TransformInput, this value is used. */
  const float kMissingValue = -1.0f;

 public:
  void MapToNDArray(const nlohmann::json& input_json, const nlohmann::json& transform,
                    tvm::runtime::NDArray& input_array) const;
};

/*! \brief Handles transformations of input and output data. */
class DLR_DLL DataTransform {
 private:
  /*! \brief Helper function for TransformInput. Interpets 1-D char input as JSON. */
  nlohmann::json GetAsJson(const int64_t* shape, const void* input, int dim) const;

  /*! \brief Helper function for TransformInput. Allocates NDArray to store mapped input data. */
  tvm::runtime::NDArray InitNDArray(const nlohmann::json& input_json, DLDataType dtype,
                                    DLContext ctx) const;

  const std::shared_ptr<std::unordered_map<std::string, std::shared_ptr<Transformer>>>
  GetTransformerMap() const;

 public:
  /*! \brief Returns true if the input requires a data transform */
  bool HasInputTransform(const nlohmann::json& metadata) const;

  /*! \brief Returns true if the output requires a data transform */
  bool HasOutputTransform(const nlohmann::json& metadata, int index) const;

  /*! \brief Transform string input using CategoricalString input DataTransform. When
   * this map is present in the metadata file, the user is expected to provide string inputs to
   * SetDLRInput as 1-D vector. This function will interpret the user's input as JSON, apply the
   * mapping to convert strings to numbers, and produce a numeric NDArray which can be given to TVM
   * for the model input.
   */
  void TransformInput(const nlohmann::json& metadata, const int64_t* shape, const void* input, int dim,
                      const std::vector<DLDataType>& dtypes, DLContext ctx,
                      std::vector<tvm::runtime::NDArray>* tvm_inputs) const;
};

}  // namespace dlr

#endif  // DLR_DATA_TRANSFORM_H_
