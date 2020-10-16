#ifndef DLR_DATA_TRANSFORM_H_
#define DLR_DATA_TRANSFORM_H_

#include <tvm/runtime/ndarray.h>

#include <nlohmann/json.hpp>

#include "dlr_common.h"

namespace dlr {

/*! \brief Handles transformations of input and output data. */
class DLR_DLL DataTransform {
 private:
  /*! \brief Helper function for TransformInput. Interpets 1-D char input as JSON. */
  nlohmann::json GetAsJson(const int64_t* shape, void* input, int dim);

  /*! \brief Helper function for TransformInput. Allocates NDArray to store mapped input data. */
  tvm::runtime::NDArray InitNDArray(int index, const nlohmann::json& input_json,
                                    const nlohmann::json& mapping, DLDataType dtype, DLContext ctx);

  /*! \brief Helper function for TransformInput. Applies mapping and writes transformed input data
   * to input_array. */
  void MapToNDArray(const nlohmann::json& input_json, tvm::runtime::NDArray& input_array,
                    const nlohmann::json& mapping);

 public:
  /*! \brief Returns true if the input requires a data transform */
  bool HasInputTransform(const nlohmann::json& metadata, int index);

  /*! \brief Returns true if the output requires a data transform */
  bool HasOutputTransform(const nlohmann::json& metadata, int index);

  /*! \brief Transform string input using CategoricalString input DataTransform. When
   * this map is present in the metadata file, the user is expected to provide string inputs to
   * SetDLRInput as 1-D vector. This function will interpret the user's input as JSON, apply the
   * mapping to convert strings to numbers, and produce a numeric NDArray which can be given to TVM
   * for the model input.
   */
  tvm::runtime::NDArray TransformInput(const nlohmann::json& metadata, int index,
                                       const int64_t* shape, void* input, int dim, DLDataType dtype,
                                       DLContext ctx);
};

}  // namespace dlr

#endif  // DLR_DATA_TRANSFORM_H_
