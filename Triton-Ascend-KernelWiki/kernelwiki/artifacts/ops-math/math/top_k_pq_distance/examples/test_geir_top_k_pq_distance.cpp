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
 * \file test_geir_top_k_pq_distance.cpp
 * \brief GE IR test for TopKPQDistance operator
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

#include "../op_graph/top_k_pq_distance_proto.h"

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
        case ge::DT_INT64:
        case ge::DT_UINT64:
            return 8U;
        default:
            return 0U;
    }
}

int32_t GenData(vector<int64_t> shapes, Tensor& tensor, TensorDesc& desc, DataType dt, double value)
{
    desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (size_t i = 0; i < shapes.size(); i++) {
        size *= static_cast<size_t>(shapes[i]);
    }
    uint32_t type_size = GetDataTypeSize(dt);
    if (type_size == 0U) {
        printf("%s - ERROR - [XIR]: GenData: unsupported data type\n", GetTime().c_str());
        return FAILED;
    }
    uint32_t data_len = static_cast<uint32_t>(size * type_size);
    uint8_t* buf = new (std::nothrow) uint8_t[data_len];
    if (buf == nullptr) {
        return FAILED;
    }
    if (dt == DT_FLOAT) {
        float* p = reinterpret_cast<float*>(buf);
        for (size_t i = 0; i < size; i++) {
            p[i] = static_cast<float>(value + static_cast<double>(i));
        }
    } else if (dt == DT_INT32) {
        int32_t* p = reinterpret_cast<int32_t*>(buf);
        for (size_t i = 0; i < size; i++) {
            p[i] = static_cast<int32_t>(value + static_cast<double>(i));
        }
    } else {
        delete[] buf;
        return FAILED;
    }
    tensor = Tensor(desc, buf, data_len);
    delete[] buf;
    return SUCCESS;
}

int32_t WriteDataToFile(string bin_file, uint64_t data_size, uint8_t* inputData)
{
    FILE* fp = fopen(bin_file.c_str(), "wb");
    if (fp == nullptr) {
        return FAILED;
    }
    size_t written = fwrite(inputData, 1, data_size, fp);
    fclose(fp);
    if (written != data_size) {
        return FAILED;
    }
    return SUCCESS;
}

void ProcessInputData(vector<Tensor>& input)
{
    for (size_t i = 0; i < input.size(); i++) {
        string input_file = "./tc_ge_irrun_test_0008_npu_input_" + std::to_string(i) + ".bin";
        uint8_t* data = input[i].GetData();
        int64_t shape_size = input[i].GetTensorDesc().GetShape().GetShapeSize();
        uint32_t type_size = GetDataTypeSize(input[i].GetTensorDesc().GetDataType());
        if (type_size == 0U) {
            printf("ERROR: input %zu has unsupported dtype\n", i);
            continue;
        }
        uint32_t data_size = static_cast<uint32_t>(shape_size * type_size);
        WriteDataToFile(input_file.c_str(), data_size, data);
    }
}

void ProcessOutputData(vector<Tensor>& output)
{
    for (size_t i = 0; i < output.size(); i++) {
        string output_file = "./tc_ge_irrun_test_0008_npu_output_" + std::to_string(i) + ".bin";
        uint8_t* data = output[i].GetData();
        int64_t shape_size = output[i].GetTensorDesc().GetShape().GetShapeSize();
        uint32_t type_size = GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        if (type_size == 0U) {
            printf("ERROR: output %zu has unsupported dtype\n", i);
            continue;
        }
        uint32_t data_size = static_cast<uint32_t>(shape_size * type_size);
        WriteDataToFile(output_file.c_str(), data_size, data);
    }
}

#define ADD_DYNAMIC_INPUT(portName, attrIndex, inputDtype, inputShape, nodeName, dataName, genValue) \
    do {                                                                                             \
        auto dataName = op::Data(nodeName).set_attr_index(attrIndex);                                \
        TensorDesc dataName##Desc(ge::Shape(inputShape), FORMAT_ND, inputDtype);                     \
        dataName##Desc.SetPlacement(ge::kPlacementHost);                                             \
        dataName##Desc.SetFormat(FORMAT_ND);                                                         \
        Tensor dataName##Tensor;                                                                     \
        GenData(inputShape, dataName##Tensor, dataName##Desc, inputDtype, genValue);                 \
        input.push_back(dataName##Tensor);                                                           \
        dataName.update_input_desc_x(dataName##Desc);                                                \
        dataName.update_output_desc_y(dataName##Desc);                                               \
        node.UpdateDynamicInputDesc(#portName, 0, dataName##Desc);                                   \
        node.set_dynamic_input_##portName(0, dataName);                                              \
        graph.AddOp(dataName);                                                                       \
        inputs.push_back(dataName);                                                                  \
    } while (0)

int CreateOppInGraph(DataType inDtype, vector<Tensor>& input, vector<Operator>& inputs, vector<Operator>& outputs,
                     Graph& graph)
{
    constexpr int32_t k = 3;
    constexpr int32_t group_size = 2;
    constexpr int32_t actual_count = 6;

    auto node = op::TopKPQDistance("topkpqdistance_1");
    node.create_dynamic_input_actual_count(1);
    node.create_dynamic_input_pq_distance(1);
    node.create_dynamic_input_grouped_extreme_distance(1);
    node.create_dynamic_input_pq_ivf(1);
    node.create_dynamic_input_pq_index(1);

    vector<int64_t> actualCountShape = {1};
    vector<int64_t> pqDistanceShape = {actual_count};
    vector<int64_t> groupedExtremeShape = {actual_count / group_size};
    vector<int64_t> topkShape = {k};

    ADD_DYNAMIC_INPUT(actual_count, 0, DT_INT32, actualCountShape, "actual_count_data", acData,
                      static_cast<double>(actual_count));
    ADD_DYNAMIC_INPUT(pq_distance, 1, inDtype, pqDistanceShape, "pq_distance_data", pqData, 1.0);
    ADD_DYNAMIC_INPUT(grouped_extreme_distance, 2, inDtype, groupedExtremeShape, "grouped_extreme_data", gedData, 1.0);
    ADD_DYNAMIC_INPUT(pq_ivf, 3, DT_INT32, pqDistanceShape, "pq_ivf_data", ivfData, 1.0);
    ADD_DYNAMIC_INPUT(pq_index, 4, DT_INT32, pqDistanceShape, "pq_index_data", idxData, 1.0);

    node.set_attr_order("DES");
    node.set_attr_k(k);
    node.set_attr_group_size(group_size);

    TensorDesc topkDistDesc(ge::Shape(topkShape), FORMAT_ND, inDtype);
    node.update_output_desc_topk_distance(topkDistDesc);
    TensorDesc topkIvfDesc(ge::Shape(topkShape), FORMAT_ND, DT_INT32);
    node.update_output_desc_topk_ivf(topkIvfDesc);
    TensorDesc topkIdxDesc(ge::Shape(topkShape), FORMAT_ND, DT_INT32);
    node.update_output_desc_topk_index(topkIdxDesc);

    graph.AddOp(node);
    outputs.push_back(node);
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    const char* graph_name = "tc_ge_irrun_test";
    Graph graph(graph_name);
    vector<Tensor> input;

    printf("%s - INFO - [XIR]: Start to initialize ge\n", GetTime().c_str());
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Initialize ge failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Initialize ge success\n", GetTime().c_str());

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
    ge::Session* session = new Session(build_options);
    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create session failed\n", GetTime().c_str());
        return FAILED;
    }

    uint32_t graph_id = 0;
    std::map<AscendString, AscendString> graph_options = {};
    ret = session->AddGraph(graph_id, graph, graph_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Add graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }

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

    delete session;
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Finalize failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize ir graph session success\n", GetTime().c_str());
    return SUCCESS;
}
