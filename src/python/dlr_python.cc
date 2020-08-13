#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/iostream.h>
#include <dlr_model.h>

namespace py = pybind11;


std::pair <size_t, std::string> GetTypeSizeAndPythonFormatDescriptor (std::string typestr) {
    if(typestr == "float32" || typestr == "float") {
        return std::make_pair(sizeof(float), py::format_descriptor<float>::format());
    } else if(typestr == "float64" || typestr == "double") {
        return std::make_pair(sizeof(double), py::format_descriptor<double>::format());
    } else if(typestr == "int16" ) {
        return std::make_pair(sizeof(int16_t), py::format_descriptor<int16_t>::format());
    } else if(typestr == "uint8" ) {
        return std::make_pair(sizeof(uint8_t), py::format_descriptor<uint8_t>::format());
    } else if(typestr == "int32" ) {
        return std::make_pair(sizeof(int32_t), py::format_descriptor<int32_t>::format());
    } else if(typestr == "int64" ) {
        return std::make_pair(sizeof(int64_t), py::format_descriptor<int64_t>::format());
    } else {
        throw dmlc::Error("Invalid type string!");
    }
};

py::array GetPyArrayFromBuffer(std::string typestr, py::buffer_info info) {
    if(typestr == "float32" || typestr == "float") {
        return py::array_t<float>(info);
    } else if(typestr == "float64" || typestr == "double") {
        return py::array_t<double>(info);
    } else if(typestr == "uint8" ) {
        return py::array_t<uint8_t>(info);
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

std::vector<py::array> Run(dlr::DLRModel &model, std::map<std::string, py::buffer> inputs) {
    size_t batch_size;
    for (std::pair<std::string, py::buffer> input : inputs) {
        std::string input_name = input.first;
        auto index = model.GetInputIndex(input_name);
        py::buffer_info input_buf = input.second.request();
        auto expected_input_type = model.GetInputType(index);
        auto type_size_and_format_descriptor = GetTypeSizeAndPythonFormatDescriptor(expected_input_type);
        auto type_size = type_size_and_format_descriptor.first;
        auto format_descriptor = type_size_and_format_descriptor.second;
        if (format_descriptor != input_buf.format) {
            std::string msg = "input data with name " + input_name + " should have dtype " + expected_input_type;
            throw py::value_error(msg);
        }
        model.SetInputStrides(input_name, input_buf.strides);
        batch_size = input_buf.shape[0];
        std::cout << "Input Size " << batch_size << std::endl;
        model.SetInput(input_name, batch_size, static_cast<void *>(input_buf.ptr));
    }

    model.Run();

    int num_outputs = model.GetNumOutputs();
    std::vector<std::vector<ssize_t>> output_shapes(num_outputs);
    std::vector<std::vector<ssize_t>> strides(num_outputs);


    for (int i = 0; i < num_outputs; i++) {
        output_shapes[i] = model.GetOutputShape(i);
        strides[i] = model.GetOutputShape(i);
        strides[i][0] = 1;
        for (int j = 1; j < strides[i].size(); j++) {
            strides[i][j] *= strides[i][j-1];
        }
        std::reverse(strides[i].begin(), strides[i].end());
        output_shapes[i][0] = batch_size;
    }


    std::vector<py::array> outputs(num_outputs);
    for(int i = 0; i < num_outputs; i++) {
        auto typestr = model.GetOutputType(i);
        auto type_size_and_format_descriptor = GetTypeSizeAndPythonFormatDescriptor(typestr);
        auto type_size = type_size_and_format_descriptor.first;
        auto format_descriptor = type_size_and_format_descriptor.second;
        auto shape = output_shapes[i];
        auto stride = strides[i];
        std::transform(stride.begin(), stride.end(), stride.begin(), [&type_size](auto &stride_elem){return stride_elem*type_size;});
        std::cout << "Strides: " << stride[0] <<  " , " << stride[1] << std::endl;
        outputs[i] = GetPyArrayFromBuffer(typestr, py::buffer_info(
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
}

PYBIND11_MODULE(_dlr, m) {
    py::class_<dlr::DLRModel>(m, "DLRModel")
        .def(py::init([](std::string path, std::string dev_type, int dev_id){
            return std::unique_ptr<dlr::DLRModel>(dlr::DLRModel::CreateModel(path, dev_type, dev_id));
        }), py::arg("path"), py::arg("dev_type") = "cpu", py::arg("dev_id") = 0, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("get_num_inputs", &dlr::DLRModel::GetNumInputs, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("get_input_strides", &dlr::DLRModel::GetInputStrides, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("get_input_dim", &dlr::DLRModel::GetInputDim, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("get_input_size", &dlr::DLRModel::GetInputSize, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("get_input_name", &dlr::DLRModel::GetInputName, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("get_input_names", &dlr::DLRModel::GetInputNames, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("get_input_dtype", &dlr::DLRModel::GetInputType, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("get_input_dtypes", &dlr::DLRModel::GetInputTypes, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("get_input_shape", &dlr::DLRModel::GetInputShape, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("get_num_outputs", &dlr::DLRModel::GetNumOutputs, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("get_output_dim", &dlr::DLRModel::GetOutputDim, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("get_output_size", &dlr::DLRModel::GetOutputSize, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("get_output_name", &dlr::DLRModel::GetOutputName, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("get_output_names", &dlr::DLRModel::GetOutputNames, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("get_output_dtype", &dlr::DLRModel::GetOutputType, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("get_output_dtypes", &dlr::DLRModel::GetOutputTypes, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("get_output_shape", &dlr::DLRModel::GetOutputShape, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("has_metadata", &dlr::DLRModel::HasMetadata, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("get_backend", &dlr::DLRModel::GetBackend, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("get_input", [](dlr::DLRModel &model, std::string name) {
            auto index = model.GetInputIndex(name);
            auto typestr = model.GetInputType(index);
            auto type_size_and_format_descriptor = GetTypeSizeAndPythonFormatDescriptor(typestr);
            auto type_size = type_size_and_format_descriptor.first;
            auto format_descriptor = type_size_and_format_descriptor.second;
            auto shape = model.GetInputShape(index);
            std::vector<ssize_t> stride = model.GetInputStrides(index);
            auto input = GetPyArrayFromBuffer(typestr, py::buffer_info(
                nullptr,
                type_size,
                format_descriptor,
                shape.size(),
                shape,
                stride
            ));
            py::buffer_info result_buffer = input.request();
            model.GetInput(name.c_str(), result_buffer.ptr);
            return input;
        }, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("run", &Run, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("run", [](dlr::DLRModel &model, py::buffer input) -> std::vector<py::array> {
            auto input_name = model.GetInputName(0);
            std::map<std::string, py::buffer> inputs{
                {input_name, input}
            };
            return Run(model, inputs);
        }, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());
}

