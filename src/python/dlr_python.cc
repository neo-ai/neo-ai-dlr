#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <dlr_model.h>

namespace py = pybind11;

PYBIND11_MODULE(_dlr, m) {
    py::class_<dlr::DLRModel>(m, "DLRModel")
        .def(py::init([](std::string path, std::string device_type = "cpu", int device_id = 0){
            return std::unique_ptr<dlr::DLRModel>(dlr::DLRModel::create_model(path, device_type, device_id));
        }))
        .def("get_num_inputs", &dlr::DLRModel::GetNumInputs)
        .def("get_input_dim", &dlr::DLRModel::GetInputDim)
        .def("get_input_size", &dlr::DLRModel::GetInputSize)
        .def("get_input_name", &dlr::DLRModel::GetInputName)
        .def("get_input_names", &dlr::DLRModel::GetInputNames)
        .def("get_input_dtype", &dlr::DLRModel::GetInputType)
        .def("get_input_dtypes", &dlr::DLRModel::GetInputTypes)
        .def("get_input_shape", &dlr::DLRModel::GetInputShape)
        .def("get_num_outputs", &dlr::DLRModel::GetNumOutputs)
        .def("get_output_dim", &dlr::DLRModel::GetOutputDim)
        .def("get_output_size", &dlr::DLRModel::GetOutputSize)
        .def("get_output_name", &dlr::DLRModel::GetOutputName)
        .def("get_output_names", &dlr::DLRModel::GetOutputNames)
        .def("get_output_dtype", &dlr::DLRModel::GetOutputType)
        .def("get_output_dtypes", &dlr::DLRModel::GetOutputTypes)
        .def("get_output_shape", &dlr::DLRModel::GetOutputShape)
        .def("has_metadata", &dlr::DLRModel::HasMetadata)
        .def("get_backend", &dlr::DLRModel::GetBackend)
        .def("run", [](dlr::DLRModel &model, std::map<std::string, py::buffer> inputs) -> std::vector<py::array> {
            size_t batch_size;
            for (std::pair<std::string, py::buffer> input : inputs) {
                std::string input_name = input.first;
                py::buffer_info input_buf = input.second.request();
                batch_size = input_buf.shape[0];
                model.SetInput(input_name, batch_size, static_cast<void *>(input_buf.ptr));
            }

            model.Run();

            int num_outputs = model.GetNumOutputs();
            std::vector<std::vector<ssize_t>> output_shapes(num_outputs);
            std::vector<std::vector<ssize_t>> strides(num_outputs);


            for (int i = 0; i < num_outputs; i++) {
                output_shapes[i] = model.GetOutputShape(i);
                strides[i] = model.GetOutputShape(i);
                for (int j = 1; j < strides.size(); j++) {
                    strides[i][j] *= strides[i][j-1];
                }
                std::reverse(strides[i].begin(), strides[i].end());
                output_shapes[i][0] = batch_size;
            }

            auto get_type_size_and_python_format_descriptor = [](std::string typestr) -> std::pair <size_t, std::string> {
                if(typestr == "float32" || typestr == "float") {
                    return std::make_pair(sizeof(float), py::format_descriptor<float>::format());
                } else if(typestr == "float64" || typestr == "double") {
                    return std::make_pair(sizeof(double), py::format_descriptor<double>::format());
                } else if(typestr == "int16" ) {
                    return std::make_pair(sizeof(int16_t), py::format_descriptor<int16_t>::format());
                } else if(typestr == "int32" ) {
                    return std::make_pair(sizeof(int32_t), py::format_descriptor<int32_t>::format());
                } else if(typestr == "int64" ) {
                    return std::make_pair(sizeof(int64_t), py::format_descriptor<int64_t>::format());
                } else {
                    throw dmlc::Error("Invalid type string!");
                }
            };

            auto get_py_array_from_buffer = [](std::string typestr, py::buffer_info info) -> py::array {
                if(typestr == "float32" || typestr == "float") {
                    return py::array_t<float>(info);
                } else if(typestr == "float64" || typestr == "double") {
                    return py::array_t<double>(info);
                } else if(typestr == "int16" ) {
                    return py::array_t<int16_t>(info);
                } else if(typestr == "int32" ) {
                    return py::array_t<int32_t>(info);
                } else if(typestr == "int64" ) {
                    return py::array_t<int64_t>(info);
                } else {
                    throw dmlc::Error("Invalid type string!");
                }
            };

            std::vector<py::array> outputs(num_outputs);
            for(int i = 0; i < num_outputs; i++) {
                auto typestr = model.GetOutputType(i);
                auto type_size_and_format_descriptor = get_type_size_and_python_format_descriptor(typestr);
                auto type_size = type_size_and_format_descriptor.first;
                auto format_descriptor = type_size_and_format_descriptor.second;
                auto shape = output_shapes[i];
                std::transform(strides[i].begin(), strides[i].end(), strides[i].begin(), [&type_size](auto &stride_elem){return stride_elem*type_size;});
                auto stride = strides[i];
                outputs[i] = get_py_array_from_buffer(typestr, py::buffer_info(
                    nullptr,
                    type_size,
                    format_descriptor,
                    shape.size(),
                    shape,
                    stride
                ));
                py::buffer_info result_buffer = outputs[i].request();
                model.GetOutput(i, result_buffer.ptr);
            }

            return outputs;
        });
}

