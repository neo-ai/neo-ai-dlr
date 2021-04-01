#ifndef DLR_DATA_TRANSFORM_H_
#define DLR_DATA_TRANSFORM_H_

#include <tvm/runtime/ndarray.h>

#include <ctime>
#include <nlohmann/json.hpp>

#include "dlr_common.h"

namespace dlr {

/*! \brief Base case for input transformers. */
class DLR_DLL Transformer {
 protected:
  const tvm::runtime::NDArray empty_;

 public:
  virtual void MapToNDArray(const nlohmann::json& input_json, const nlohmann::json& transform,
                            tvm::runtime::NDArray& input_array) const = 0;

  /*! \brief Helper function for TransformInput. Allocates NDArray to store mapped input data. */
  virtual void InitNDArray(const nlohmann::json& input_json, const nlohmann::json& transform,
                           DLDataType dtype, DLContext ctx,
                           tvm::runtime::NDArray& input_array) const;
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

class DLR_DLL DateTimeTransformer : public Transformer {
 private:
  /*! \brief Number of columns defined by Autopilot Sagemaker-Scikit-Learn-Extension for
   * DateTimeVectorizer */
  const int kNumDateTimeCols = 7;

  const std::vector<std::string> datetime_templates = {
      "%h %dth, %Y, %I:%M:%S%p",
      "%h %dth, %Y, %I:%M%p",
      "%h %dth, %Y, %I%p",
      "%Y-%m-%d %I:%M:%S%p",
      "%Y-%m-%d %H:%M:%S+%Z",
      "%Y-%m-%d %H:%M:%S-%Z ",
      "%Y-%m-%d %H:%M:%S",
      "%Y-%m-%d",
      "%H:%M:%S+00",
      "%H:%M:%S",
  };

  /*! \brief Convert a given string to an array of digits representing [WEEKDAY,
   * YEAR, HOUR, MINUTE, SECOND, MONTH, WEEK_OF_YEAR*/
  void DigitizeDateTime(std::string& input_string, std::vector<int64_t>& datetime_digits) const;

  bool isLeap(int64_t year) const;

  int64_t GetWeekNumber(std::tm tm) const;

 public:
  void MapToNDArray(const nlohmann::json& input_json, const nlohmann::json& transform,
                    tvm::runtime::NDArray& input_array) const;

  void InitNDArray(const nlohmann::json& input_json, const nlohmann::json& transform,
                   DLDataType dtype, DLContext ctx, tvm::runtime::NDArray& input_array) const;
};

/*! \brief Handles transformations of input and output data. */
class DLR_DLL DataTransform {
 private:
  /*! \brief When there is no mapping entry for TransformOutput, this value is used. */
  const char* kUnknownLabel = "<unseen_label>";

  /*! \brief Buffers to store transformed outputs. Maps output index to transformed data. */
  std::unordered_map<int, std::string> transformed_outputs_;

  /*! \brief Helper function for TransformInput. Interpets 1-D char input as JSON. */
  nlohmann::json GetAsJson(const int64_t* shape, const void* input, int dim) const;

  const std::shared_ptr<std::unordered_map<std::string, std::shared_ptr<Transformer>>>
  GetTransformerMap() const;

  template <typename T>
  nlohmann::json TransformOutputHelper1D(const nlohmann::json& mapping, const T* data,
                                         const std::vector<int64_t>& shape) const;

  template <typename T>
  nlohmann::json TransformOutputHelper2D(const nlohmann::json& mapping, const T* data,
                                         const std::vector<int64_t>& shape) const;

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
  void TransformInput(const nlohmann::json& metadata, const int64_t* shape, const void* input,
                      int dim, const std::vector<DLDataType>& dtypes, DLContext ctx,
                      std::vector<tvm::runtime::NDArray>* tvm_inputs) const;

  /*! \brief Transform integer output using CategoricalString output DataTransform. When this map is
   * present in the metadata file, the model's output will be converted from an integer array to a
   * JSON string, where numbers are mapped back to strings according to the CategoricalString map in
   * the metadata file. A buffer is created to store the transformed output, and it's contents can
   * be accessed using the GetOutputShape, GetOutputSizeDim, GetOutput and GetOutputPtr methods.
   */
  void TransformOutput(const nlohmann::json& metadata, int index,
                       const tvm::runtime::NDArray& output_array);

  /*! \brief Get shape of transformed output. */
  void GetOutputShape(int index, int64_t* shape) const;

  /*! \brief Get size and dims of transformed output. */
  void GetOutputSizeDim(int index, int64_t* size, int* dim) const;

  /*! \brief Copy transformed output to a buffer. */
  void GetOutput(int index, void* output) const;

  /*! \brief Get pointer to transformed output data. */
  const void* GetOutputPtr(int index) const;
};

}  // namespace dlr

#endif  // DLR_DATA_TRANSFORM_H_
