/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_geir_greater_equal.cpp
 * \brief Test GreaterEqual via GE IR graph mode
 */

#include <cstdint>
#include <cstdio>
#include <ctime>
#include <map>
#include <new>
#include <string>
#include <vector>

#include "array_ops.h"
#include "ge_api.h"
#include "ge_api_types.h"
#include "ge_error_codes.h"
#include "ge_ir_build.h"
#include "graph.h"
#include "tensor.h"
#include "types.h"

#include "../op_graph/greater_equal_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64];
    struct tm tm_info;
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime_r(&timep, &tm_info));
    return tmp;
}

template <typename T>
int32_t GenTensorData(const vector<int64_t>& shapes, const vector<T>& values, TensorDesc& tensor_desc, Tensor& tensor)
{
    tensor_desc.SetRealDimCnt(shapes.size());
    size_t element_count = 1;
    for (size_t i = 0; i < shapes.size(); ++i) {
        element_count *= static_cast<size_t>(shapes[i]);
    }
    if (values.size() != element_count) {
        return FAILED;
    }

    tensor = Tensor(
        tensor_desc, reinterpret_cast<const uint8_t*>(values.data()),
        values.size() * sizeof(typename vector<T>::value_type));
    return SUCCESS;
}

int CreateOpInGraph(vector<Tensor>& input_tensors, vector<Operator>& inputs, vector<Operator>& outputs, Graph& graph)
{
    Status ret = SUCCESS;
    auto greater_equal_op = op::GreaterEqual("greater_equal_graph");
    vector<int64_t> input_shape = {4, 2};

    auto x1 = op::Data("x1").set_attr_index(0);
    TensorDesc x1_desc(Shape(input_shape), FORMAT_ND, DT_DOUBLE);
    x1_desc.SetPlacement(ge::kPlacementHost);
    x1_desc.SetFormat(FORMAT_ND);
    Tensor x1_tensor;
    ret = GenTensorData<double>(input_shape, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}, x1_desc, x1_tensor);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate x1 data failed\n", GetTime().c_str());
        return FAILED;
    }
    x1.update_input_desc_x(x1_desc);
    x1.update_output_desc_y(x1_desc);
    input_tensors.push_back(x1_tensor);
    graph.AddOp(x1);
    greater_equal_op.set_input_x1(x1);
    greater_equal_op.update_input_desc_x1(x1_desc);
    inputs.push_back(x1);

    auto x2 = op::Data("x2").set_attr_index(1);
    TensorDesc x2_desc(Shape(input_shape), FORMAT_ND, DT_DOUBLE);
    x2_desc.SetPlacement(ge::kPlacementHost);
    x2_desc.SetFormat(FORMAT_ND);
    Tensor x2_tensor;
    ret = GenTensorData<double>(input_shape, {1.0, 3.0, 2.0, 5.0, 5.0, 0.0, 9.0, 8.0}, x2_desc, x2_tensor);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate x2 data failed\n", GetTime().c_str());
        return FAILED;
    }
    x2.update_input_desc_x(x2_desc);
    x2.update_output_desc_y(x2_desc);
    input_tensors.push_back(x2_tensor);
    graph.AddOp(x2);
    greater_equal_op.set_input_x2(x2);
    greater_equal_op.update_input_desc_x2(x2_desc);
    inputs.push_back(x2);

    TensorDesc y_desc(Shape(input_shape), FORMAT_ND, DT_BOOL);
    greater_equal_op.update_output_desc_y(y_desc);
    outputs.push_back(greater_equal_op);
    return SUCCESS;
}

int VerifyOutput(const vector<Tensor>& output_tensors)
{
    static const bool expected[] = {true, false, true, false, true, true, false, true};
    if (output_tensors.size() != 1) {
        printf("Unexpected output tensor count: %zu\n", output_tensors.size());
        return FAILED;
    }

    const Tensor& output_tensor = output_tensors[0];
    int64_t element_count = output_tensor.GetTensorDesc().GetShape().GetShapeSize();
    if (element_count != static_cast<int64_t>(sizeof(expected) / sizeof(expected[0]))) {
        printf("Unexpected output element count: %ld\n", element_count);
        return FAILED;
    }

    const auto* result_data = reinterpret_cast<const bool*>(output_tensor.GetData());
    for (int64_t i = 0; i < element_count; ++i) {
        printf("result[%ld] is: %u\n", i, static_cast<unsigned int>(result_data[i]));
        if (result_data[i] != expected[i]) {
            printf(
                "GreaterEqual output mismatch at index %ld, expect %u but got %u\n", i,
                static_cast<unsigned int>(expected[i]), static_cast<unsigned int>(result_data[i]));
            return FAILED;
        }
    }
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;

    const char* graph_name = "tc_ge_irrun_greater_equal";
    Graph graph(graph_name);
    vector<Tensor> input_tensors;

    printf("%s - INFO - [XIR]: Start to initialize ge using ge global options\n", GetTime().c_str());
    map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Initialize ge using ge global options failed\n", GetTime().c_str());
        return FAILED;
    }

    vector<Operator> inputs;
    vector<Operator> outputs;
    ret = CreateOpInGraph(input_tensors, inputs, outputs, graph);
    if (ret != SUCCESS) {
        ge::GEFinalize();
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    map<AscendString, AscendString> build_options;
    ge::Session* session = new (std::nothrow) Session(build_options);
    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create ir session failed\n", GetTime().c_str());
        ge::GEFinalize();
        return FAILED;
    }

    map<AscendString, AscendString> graph_options = {{"ge.exec.precision_mode", "allow_mix_precision"}};
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Add graph failed\n", GetTime().c_str());
        delete session;
        ge::GEFinalize();
        return FAILED;
    }

    vector<Tensor> output_tensors;
    ret = session->RunGraph(graph_id, input_tensors, output_tensors);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Run graph failed\n", GetTime().c_str());
        delete session;
        ge::GEFinalize();
        return FAILED;
    }

    ret = VerifyOutput(output_tensors);
    delete session;
    if (ret != SUCCESS) {
        ge::GEFinalize();
        return FAILED;
    }

    printf("%s - INFO - [XIR]: GreaterEqual double graph example verified successfully\n", GetTime().c_str());
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: GE Finalize failed\n", GetTime().c_str());
        return FAILED;
    }
    return SUCCESS;
}