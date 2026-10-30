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
 * \file test_geir_topk.cpp
 * \brief GE IR test for TopK operator (force AICPU by excluding AiCore engine)
 */

#include <iostream>
#include <fstream>
#include <string.h>
#include <stdint.h>
#include <ctime>
#include <vector>
#include <string>
#include <map>
#include <numeric>
#include "assert.h"

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "array_ops.h"
#include "ge_ir_build.h"

#include "nn_other.h"
#include "../op_graph/topk_proto.h"

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
        case ge::DT_FLOAT:
            return 4U;
        case ge::DT_INT32:
            return 4U;
        case ge::DT_INT64:
            return 8U;
        case ge::DT_INT8:
            return 1U;
        case ge::DT_UINT8:
            return 1U;
        default:
            return 0U;
    }
}

int32_t GenTestData(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, DataType data_type)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= static_cast<size_t>(shapes[i]);
    }
    uint32_t type_size = GetDataTypeSize(data_type);
    if (type_size == 0U) {
        printf("%s - ERROR - [XIR]: GenTestData: unsupported data type\n", GetTime().c_str());
        return FAILED;
    }
    uint32_t data_len = static_cast<uint32_t>(size * type_size);

    if (data_type == DT_FLOAT) {
        float* pData = new (std::nothrow) float[size];
        for (size_t i = 0; i < size; ++i) {
            pData[i] = static_cast<float>(i + 1);
        }
        input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t*>(pData), data_len);
        delete[] pData;
    } else if (data_type == DT_INT32) {
        int32_t* pData = new (std::nothrow) int32_t[size];
        for (size_t i = 0; i < size; ++i) {
            pData[i] = static_cast<int32_t>(i + 1);
        }
        input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t*>(pData), data_len);
        delete[] pData;
    } else {
        int32_t* pData = new (std::nothrow) int32_t[size];
        for (size_t i = 0; i < size; ++i) {
            pData[i] = static_cast<int32_t>(i + 1);
        }
        input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t*>(pData), data_len);
        delete[] pData;
    }
    return SUCCESS;
}

int32_t GenKTensor(int32_t k, Tensor& input_tensor, TensorDesc& input_tensor_desc)
{
    input_tensor_desc.SetRealDimCnt(1);
    int32_t* pData = new (std::nothrow) int32_t[1];
    pData[0] = k;
    input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t*>(pData), sizeof(int32_t));
    delete[] pData;
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
    for (int i = 0; i < static_cast<int>(input.size()); i++) {
        string input_file = "./tc_ge_irrun_test_0008_npu_input_" + std::to_string(i) + ".bin";
        uint8_t* input_data_i = input[i].GetData();
        int64_t input_shape = input[i].GetTensorDesc().GetShape().GetShapeSize();
        uint32_t type_size = GetDataTypeSize(input[i].GetTensorDesc().GetDataType());
        if (type_size == 0U) {
            printf("ERROR: input %d has unsupported dtype\n", i);
            continue;
        }
        uint32_t data_size = static_cast<uint32_t>(input_shape * type_size);
        WriteDataToFile(input_file.c_str(), data_size, input_data_i);
    }
}

void ProcessOutputData(vector<Tensor>& output)
{
    for (int i = 0; i < static_cast<int>(output.size()); i++) {
        string output_file = "./tc_ge_irrun_test_0008_npu_output_" + std::to_string(i) + ".bin";
        uint8_t* output_data_i = output[i].GetData();
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        uint32_t type_size = GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        if (type_size == 0U) {
            printf("ERROR: output %d has unsupported dtype\n", i);
            continue;
        }
        uint32_t data_size = static_cast<uint32_t>(output_shape * type_size);
        WriteDataToFile(output_file.c_str(), data_size, output_data_i);
    }
}

int CreateOppInGraph(DataType inDtype, vector<Tensor>& input, vector<Operator>& inputs, vector<Operator>& outputs,
                     Graph& graph)
{
    vector<int64_t> x_shape = {4, 6};
    vector<int64_t> k_shape = {1};
    vector<int64_t> values_shape = {4, 3};
    vector<int64_t> indices_shape = {4, 3};

    TensorDesc x_desc = TensorDesc(ge::Shape(x_shape), FORMAT_ND, inDtype);
    x_desc.SetPlacement(ge::kPlacementHost);
    x_desc.SetFormat(FORMAT_ND);

    TensorDesc k_desc = TensorDesc(ge::Shape(k_shape), FORMAT_ND, DT_INT32);
    k_desc.SetPlacement(ge::kPlacementHost);
    k_desc.SetFormat(FORMAT_ND);

    Tensor x_tensor;
    GenTestData(x_shape, x_tensor, x_desc, inDtype);
    input.push_back(x_tensor);

    Tensor k_tensor;
    GenKTensor(3, k_tensor, k_desc);
    input.push_back(k_tensor);

    auto x_data = op::Data("x_data").set_attr_index(0);
    x_data.update_input_desc_x(x_desc);
    x_data.update_output_desc_y(x_desc);

    auto k_data = op::Data("k_data").set_attr_index(1);
    k_data.update_input_desc_x(k_desc);
    k_data.update_output_desc_y(k_desc);

    auto topk_op = op::TopK("topk_op");
    topk_op.set_input_x(x_data);
    topk_op.set_input_k(k_data);
    topk_op.set_attr_dim(-1);
    topk_op.set_attr_largest(true);
    topk_op.set_attr_sorted(true);

    TensorDesc values_desc = TensorDesc(ge::Shape(values_shape), FORMAT_ND, inDtype);
    TensorDesc indices_desc = TensorDesc(ge::Shape(indices_shape), FORMAT_ND, DT_INT32);
    topk_op.update_output_desc_values(values_desc);
    topk_op.update_output_desc_indices(indices_desc);

    graph.AddOp(x_data);
    graph.AddOp(k_data);
    graph.AddOp(topk_op);

    inputs.push_back(x_data);
    inputs.push_back(k_data);
    outputs.push_back(topk_op);

    return SUCCESS;
}

int main(int argc, char* argv[])
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
    if (argc >= 2) {
        std::string dtype_str = argv[1];
        if (dtype_str == "float") {
            inDtype = DT_FLOAT;
        } else if (dtype_str == "int32") {
            inDtype = DT_INT32;
        } else {
            std::cout << "Unknown dtype: " << dtype_str << ", using default: float" << std::endl;
        }
    }

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
        printf("%s - ERROR - [XIR]: Create ir session failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Create ir session success\n", GetTime().c_str());

    uint32_t graph_id = 0;

    std::map<AscendString, AscendString> graph_options = {{"ge.exec.exclude_engines", "AiCore"}};

    printf("%s - INFO - [XIR]: Add graph with exclude AiCore engine (force AICPU)\n", GetTime().c_str());
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
