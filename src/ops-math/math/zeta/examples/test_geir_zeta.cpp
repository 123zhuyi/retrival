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
 * \file test_geir_zeta.cpp
 * \brief GE IR test for Zeta operator
 */

#include <iostream>
#include <fstream>
#include <string.h>
#include <stdint.h>
#include <ctime>
#include <vector>
#include <string>
#include <map>
#include "assert.h"

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "array_ops.h"
#include "ge_ir_build.h"

#include "../op_graph/zeta_proto.h"

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
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime(&timep));
    return tmp;
}

uint32_t GetDataTypeSize(DataType dt)
{
    switch (dt) {
        case ge::DT_BOOL:
            return 1U;
        case ge::DT_INT8:
        case ge::DT_UINT8:
            return 1U;
        case ge::DT_FLOAT16:
        case ge::DT_INT16:
        case ge::DT_UINT16:
            return 2U;
        case ge::DT_FLOAT:
        case ge::DT_INT32:
        case ge::DT_UINT32:
            return 4U;
        case ge::DT_DOUBLE:
        case ge::DT_INT64:
        case ge::DT_UINT64:
            return 8U;
        default:
            return 0U;
    }
}

int32_t GenData(vector<int64_t> shapes, Tensor& tensor, TensorDesc& tensor_desc, DataType data_type, double value)
{
    tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= static_cast<size_t>(shapes[i]);
    }
    uint32_t type_size = GetDataTypeSize(data_type);
    if (type_size == 0U) {
        printf("%s - ERROR - [XIR]: GenData: unsupported data type\n", GetTime().c_str());
        return FAILED;
    }
    uint32_t data_len = static_cast<uint32_t>(size * type_size);
    uint8_t* buf = new (std::nothrow) uint8_t[data_len];
    if (buf == nullptr) {
        printf("%s - ERROR - [XIR]: GenData: allocate memory failed\n", GetTime().c_str());
        return FAILED;
    }

    if (data_type == ge::DT_FLOAT) {
        float* p = reinterpret_cast<float*>(buf);
        for (size_t i = 0; i < size; ++i) {
            p[i] = static_cast<float>(value);
        }
    } else if (data_type == ge::DT_DOUBLE) {
        double* p = reinterpret_cast<double*>(buf);
        for (size_t i = 0; i < size; ++i) {
            p[i] = static_cast<double>(value);
        }
    }

    tensor = Tensor(tensor_desc, buf, data_len);
    delete[] buf;
    return SUCCESS;
}

int32_t WriteDataToFile(string bin_file, uint64_t data_size, uint8_t* inputData)
{
    FILE* fp = fopen(bin_file.c_str(), "wb");
    if (fp == nullptr) {
        return FAILED;
    }
    fwrite(inputData, sizeof(uint8_t), data_size, fp);
    fclose(fp);
    return SUCCESS;
}

void ProcessInputData(vector<Tensor>& input)
{
    for (size_t i = 0; i < input.size(); i++) {
        string input_file = "./tc_ge_irrun_test_0008_npu_input_" + std::to_string(i) + ".bin";
        uint8_t* input_data_i = input[i].GetData();
        int64_t input_shape = input[i].GetTensorDesc().GetShape().GetShapeSize();
        uint32_t type_size = GetDataTypeSize(input[i].GetTensorDesc().GetDataType());
        if (type_size == 0U) {
            printf("ERROR: input %zu has unsupported dtype\n", i);
            continue;
        }
        uint32_t data_size = static_cast<uint32_t>(input_shape * type_size);
        WriteDataToFile(input_file.c_str(), data_size, input_data_i);
    }
}

void ProcessOutputData(vector<Tensor>& output)
{
    for (size_t i = 0; i < output.size(); i++) {
        string output_file = "./tc_ge_irrun_test_0008_npu_output_" + std::to_string(i) + ".bin";
        uint8_t* output_data_i = output[i].GetData();
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        uint32_t type_size = GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        if (type_size == 0U) {
            printf("ERROR: output %zu has unsupported dtype\n", i);
            continue;
        }
        uint32_t data_size = static_cast<uint32_t>(output_shape * type_size);
        WriteDataToFile(output_file.c_str(), data_size, output_data_i);
    }
}

int CreateOppInGraph(DataType inDtype, vector<Tensor>& input, vector<Operator>& inputs, vector<Operator>& outputs,
                     Graph& graph)
{
    vector<int64_t> shape = {2, 3};

    // Input x
    auto x_data = op::Data("x_data").set_attr_index(0);
    TensorDesc x_desc = TensorDesc(ge::Shape(shape), FORMAT_ND, inDtype);
    x_desc.SetPlacement(ge::kPlacementHost);
    x_desc.SetFormat(FORMAT_ND);
    Tensor x_tensor;
    int32_t ret = GenData(shape, x_tensor, x_desc, inDtype, 2.0);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate x data failed\n", GetTime().c_str());
        return FAILED;
    }
    x_data.update_input_desc_x(x_desc);
    x_data.update_output_desc_y(x_desc);
    graph.AddOp(x_data);
    input.push_back(x_tensor);
    inputs.push_back(x_data);

    // Input q
    auto q_data = op::Data("q_data").set_attr_index(1);
    TensorDesc q_desc = TensorDesc(ge::Shape(shape), FORMAT_ND, inDtype);
    q_desc.SetPlacement(ge::kPlacementHost);
    q_desc.SetFormat(FORMAT_ND);
    Tensor q_tensor;
    ret = GenData(shape, q_tensor, q_desc, inDtype, 1.5);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate q data failed\n", GetTime().c_str());
        return FAILED;
    }
    q_data.update_input_desc_x(q_desc);
    q_data.update_output_desc_y(q_desc);
    graph.AddOp(q_data);
    input.push_back(q_tensor);
    inputs.push_back(q_data);

    // Zeta operator
    auto zeta_op = op::Zeta("zeta_op");
    zeta_op.set_input_x(x_data);
    zeta_op.set_input_q(q_data);
    TensorDesc z_desc = TensorDesc(ge::Shape(shape), FORMAT_ND, inDtype);
    zeta_op.update_output_desc_z(z_desc);
    graph.AddOp(zeta_op);
    outputs.push_back(zeta_op);

    return SUCCESS;
}

int main()
{
    const char* graph_name = "tc_ge_irrun_test";
    Graph graph(graph_name);
    vector<Tensor> input;

    printf("%s - INFO - [XIR]: Start to initialize ge using ge global options\n", GetTime().c_str());
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Initialize ge using ge global options failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Initialize ge using ge global options success\n", GetTime().c_str());

    vector<Operator> inputs{};
    vector<Operator> outputs{};

    DataType inDtype = DT_FLOAT;

    ret = CreateOppInGraph(inDtype, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create graph failed\n", GetTime().c_str());
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    std::map<AscendString, AscendString> build_options = {};
    printf("%s - INFO - [XIR]: Start to create ir session\n", GetTime().c_str());
    ge::Session* session = new Session(build_options);
    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create session failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Create ir session success\n", GetTime().c_str());

    uint32_t graph_id = 0;
    std::map<AscendString, AscendString> graph_options = {};

    printf("%s - INFO - [XIR]: Add graph\n", GetTime().c_str());
    ret = session->AddGraph(graph_id, graph, graph_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Add graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }

    printf("%s - INFO - [XIR]: Start to run graph\n", GetTime().c_str());
    vector<Tensor> output;
    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Run graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Run graph success\n", GetTime().c_str());

    ProcessInputData(input);
    ProcessOutputData(output);

    printf("%s - INFO - [XIR]: Start to finalize ir graph session\n", GetTime().c_str());
    delete session;
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Finalize ir graph session failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize ir graph session success\n", GetTime().c_str());
    return SUCCESS;
}
