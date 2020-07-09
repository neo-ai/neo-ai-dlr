#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <dlr_model.h>

namespace py = pybind11;

PYBIND11_MODULE(pydlr, m) {
    py::class_<dlr::DLRModel>(m, "DLRModel")
        .def(py::init([](std::string path, int device_type, int device_id){
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
        .def("get_backend", &dlr::DLRModel::GetBackend);
}

