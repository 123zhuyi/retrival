/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <ctime>
#include <iostream>
#include <map>
#include <new>
#include <string>
#include <vector>
#include <complex>

#include "array_ops.h"
#include "ge_api.h"
#include "ge_api_types.h"
#include "ge_error_codes.h"
#include "ge_ir_build.h"
#include "graph.h"
#include "nn_other.h"
#include "tensor.h"
#include "types.h"

#include "../op_graph/matrix_diag_part_v3_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

namespace {
string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime(&timep));
    return tmp;
}

template <typename T>
int32_t GenTensor(const vector<int64_t> &shape, TensorDesc &desc, const vector<T> &values, Tensor &tensor)
{
    size_t size = 1;
    for (int64_t dim : shape) {
        size *= static_cast<size_t>(dim);
    }
    if (size != values.size()) {
        return FAILED;
    }
    T *data = new (std::nothrow) T[size];
    if (data == nullptr) {
        return FAILED;
    }
    for (size_t i = 0; i < size; ++i) {
        data[i] = values[i];
    }
    desc.SetPlacement(ge::kPlacementHost);
    desc.SetFormat(FORMAT_ND);
    desc.SetRealDimCnt(shape.size());
    tensor = Tensor(desc, reinterpret_cast<uint8_t *>(data), sizeof(T) * size);
    return SUCCESS;
}

template <typename T, typename Setter>
int AddInputTensor(Graph &graph, vector<Tensor> &input_tensors, vector<Operator> &input_ops, const string &name,
                   const vector<int64_t> &shape, DataType data_type, const vector<T> &values, Setter setter)
{
    auto placeholder = op::Data(name.c_str()).set_attr_index(0);
    TensorDesc desc(ge::Shape(shape), FORMAT_ND, data_type);
    Tensor tensor;
    if (GenTensor(shape, desc, values, tensor) != SUCCESS) {
        return FAILED;
    }
    placeholder.update_input_desc_x(desc);
    graph.AddOp(placeholder);
    input_tensors.push_back(tensor);
    input_ops.push_back(placeholder);
    setter(placeholder, desc);
    return SUCCESS;
}

template <typename T, typename Setter>
int AddConstInputTensor(Graph &graph, vector<Operator> &input_ops, const string &name, const vector<int64_t> &shape,
                        DataType data_type, const vector<T> &values, Setter setter)
{
    auto placeholder = op::Const(name.c_str());
    TensorDesc desc(ge::Shape(shape), FORMAT_ND, data_type);
    Tensor tensor;
    if (GenTensor(shape, desc, values, tensor) != SUCCESS) {
        return FAILED;
    }
    placeholder.SetAttr("value", tensor);
    placeholder.update_output_desc_y(desc);
    graph.AddOp(placeholder);
    input_ops.push_back(placeholder);
    setter(placeholder, desc);
    return SUCCESS;
}

template <typename T>
bool CompareTensor(const Tensor &tensor, const vector<T> &expected)
{
    const T *data = reinterpret_cast<const T *>(tensor.GetData());
    if (data == nullptr) {
        return false;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        if (data[i] != expected[i]) {
            return false;
        }
    }
    return true;
}

template <typename T>
void PrintTensor(const Tensor &tensor, size_t size)
{
    const T *data = reinterpret_cast<const T *>(tensor.GetData());
    if (data == nullptr) {
        printf("%s - ERROR - [XIR]: Get output data failed\n", GetTime().c_str());
        return;
    }
    std::cout << GetTime() << " - INFO - [XIR]: Output: [";
    for (size_t i = 0; i < size; ++i) {
        if (i > 0) {
            std::cout << ", ";
        }
        std::cout << data[i];
    }
    std::cout << "]" << std::endl;
}

int CreateOppInGraph(vector<Tensor> &input, vector<Operator> &inputs, vector<Operator> &outputs, Graph &graph)
{
    auto matrix_diag_part_v3 = op::MatrixDiagPartV3("matrix_diag_part_v3");
    matrix_diag_part_v3.set_attr_align("RIGHT_LEFT");

    const vector<int64_t> x_shape = {3, 3};
    const vector<std::complex<double>> x_data = {{1.0, 1.0}, {2.0, 0.0}, {3.0, -1.0},
                                                 {4.0, 2.0}, {5.0, 0.0}, {6.0, -2.0},
                                                 {7.0, 3.0}, {8.0, 0.0}, {9.0, -3.0}};
    if (AddInputTensor(graph, input, inputs, "x", x_shape, DT_COMPLEX128, x_data,
                       [&](ge::Operator &data, const TensorDesc &desc) {
                           matrix_diag_part_v3.set_input_x(data);
                           matrix_diag_part_v3.update_input_desc_x(desc);
                       }) != SUCCESS) {
        return FAILED;
    }

    const vector<int64_t> k_shape = {2};
    const vector<int32_t> k_data = {0, 0};
    if (AddConstInputTensor(graph, inputs, "k", k_shape, DT_INT32, k_data,
                            [&](ge::Operator &data, const TensorDesc &desc) {
                                matrix_diag_part_v3.set_input_k(data);
                                matrix_diag_part_v3.update_input_desc_k(desc);
                            }) != SUCCESS) {
        return FAILED;
    }

    const vector<int64_t> scalar_shape = {};
    const vector<std::complex<double>> padding_value_data = {{0.0, 0.0}};
    if (AddInputTensor(graph, input, inputs, "padding_value", scalar_shape, DT_COMPLEX128, padding_value_data,
                       [&](ge::Operator &data, const TensorDesc &desc) {
                           matrix_diag_part_v3.set_input_padding_value(data);
                           matrix_diag_part_v3.update_input_desc_padding_value(desc);
                       }) != SUCCESS) {
        return FAILED;
    }

    TensorDesc output_desc(ge::Shape({3}), FORMAT_ND, DT_COMPLEX128);
    matrix_diag_part_v3.update_output_desc_y(output_desc);
    outputs.push_back(matrix_diag_part_v3);
    return SUCCESS;
}
} // namespace

int main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;

    Graph graph("tc_ge_irrun_test");
    vector<Tensor> input;
    vector<Operator> inputs;
    vector<Operator> outputs;

    printf("%s - INFO - [XIR]: Start to initialize ge using ge global options\n", GetTime().c_str());
    map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        return FAILED;
    }

    if (CreateOppInGraph(input, inputs, outputs, graph) != SUCCESS) {
        GEFinalize();
        return FAILED;
    }
    graph.SetInputs(inputs).SetOutputs(outputs);

    map<AscendString, AscendString> build_options = {};
    Session *session = new (std::nothrow) Session(build_options);
    if (session == nullptr) {
        GEFinalize();
        return FAILED;
    }

    uint32_t graph_id = 0;
    map<AscendString, AscendString> graph_options = {};
    ret = session->AddGraph(graph_id, graph, graph_options);
    if (ret != SUCCESS) {
        delete session;
        GEFinalize();
        return FAILED;
    }

    vector<Tensor> output;
    ret = session->RunGraph(graph_id, input, output);
    delete session;
    if (ret != SUCCESS || output.size() != 1) {
        GEFinalize();
        return FAILED;
    }

    const vector<std::complex<double>> expected = {{1.0, 1.0}, {5.0, 0.0}, {9.0, -3.0}};
    PrintTensor<std::complex<double>>(output[0], expected.size());
    if (!CompareTensor(output[0], expected)) {
        printf("%s - ERROR - [XIR]: Output validation failed\n", GetTime().c_str());
        GEFinalize();
        return FAILED;
    }

    printf("%s - INFO - [XIR]: MatrixDiagPartV3 example passed\n", GetTime().c_str());
    GEFinalize();
    return SUCCESS;
}
