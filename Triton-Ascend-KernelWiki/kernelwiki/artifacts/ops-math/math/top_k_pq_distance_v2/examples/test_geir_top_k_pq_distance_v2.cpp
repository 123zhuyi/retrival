/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstring>
#include <ctime>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "array_ops.h"
#include "ge_ir_build.h"

#include "../op_graph/top_k_pq_distance_v2_proto.h"

#define FAILED (-1)
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

namespace {
// TopKPQDistanceV2 supports only float16 / float / int32, so the float32 path is used here: it is the widest
// floating-point type the operator accepts and keeps the expected values exactly representable.
constexpr int64_t kBatch = 1;
constexpr int64_t kElementNum = 8;
constexpr int64_t kGroupSize = 2;
constexpr int64_t kTopK = 4;
constexpr int64_t kGroupNum = kElementNum / kGroupSize;

// Tie-free distances: groups {5,1} {9,3} {7,2} {8,6}; ASC keeps the 4 smallest {1,2,3,5} at indices {1,5,3,0}.
const float kPqDistance[kElementNum] = {5.0f, 1.0f, 9.0f, 3.0f, 7.0f, 2.0f, 8.0f, 6.0f};
const float kGroupedExtreme[kGroupNum] = {1.0f, 3.0f, 2.0f, 6.0f};
const float kExpectDistance[kTopK] = {1.0f, 2.0f, 3.0f, 5.0f};
const int32_t kExpectIndex[kTopK] = {1, 5, 3, 0};

string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime(&timep));
    return tmp;
}

string DataTypeToString(DataType dt)
{
    switch (dt) {
        case DT_FLOAT:
            return "DT_FLOAT";
        case DT_FLOAT16:
            return "DT_FLOAT16";
        case DT_INT32:
            return "DT_INT32";
        default:
            return "DTYPE(" + std::to_string(static_cast<int32_t>(dt)) + ")";
    }
}

Tensor BuildTensor(const vector<int64_t>& shape, DataType dtype, const void* data, size_t byteSize)
{
    TensorDesc desc(ge::Shape(shape), FORMAT_ND, dtype);
    desc.SetPlacement(ge::kPlacementHost);
    desc.SetRealDimCnt(shape.size());
    Tensor tensor(desc, reinterpret_cast<const uint8_t*>(data), byteSize);
    return tensor;
}

int CreateOppInGraph(vector<ge::Tensor>& input, vector<Operator>& inputs, vector<Operator>& outputs, Graph& graph)
{
    vector<int64_t> pqShape = {kBatch, kElementNum};
    vector<int64_t> groupedShape = {kBatch, kGroupNum};

    TensorDesc pqDesc(ge::Shape(pqShape), FORMAT_ND, DT_FLOAT);
    pqDesc.SetPlacement(ge::kPlacementHost);
    auto pqData = op::Data("pq_distance").set_attr_index(0);
    pqData.update_input_desc_x(pqDesc);
    pqData.update_output_desc_y(pqDesc);
    input.push_back(BuildTensor(pqShape, DT_FLOAT, kPqDistance, sizeof(kPqDistance)));
    graph.AddOp(pqData);
    inputs.push_back(pqData);

    TensorDesc groupedDesc(ge::Shape(groupedShape), FORMAT_ND, DT_FLOAT);
    groupedDesc.SetPlacement(ge::kPlacementHost);
    auto groupedData = op::Data("grouped_extreme_distance").set_attr_index(1);
    groupedData.update_input_desc_x(groupedDesc);
    groupedData.update_output_desc_y(groupedDesc);
    input.push_back(BuildTensor(groupedShape, DT_FLOAT, kGroupedExtreme, sizeof(kGroupedExtreme)));
    graph.AddOp(groupedData);
    inputs.push_back(groupedData);

    auto topkOp = op::TopKPQDistanceV2("top_k_pq_distance_v2");
    topkOp.set_input_pq_distance(pqData);
    topkOp.set_input_grouped_extreme_distance(groupedData);
    topkOp.update_input_desc_pq_distance(pqDesc);
    topkOp.update_input_desc_grouped_extreme_distance(groupedDesc);
    topkOp.set_attr_order("ASC");
    topkOp.set_attr_k(kTopK);
    topkOp.set_attr_group_size(kGroupSize);
    graph.AddOp(topkOp);

    outputs.push_back(topkOp);
    return SUCCESS;
}

int CheckOutput(vector<ge::Tensor>& output)
{
    if (output.size() != 2U) {
        std::cout << "ERROR: expect 2 outputs, but got " << output.size() << std::endl;
        return FAILED;
    }

    // Print the real output tensors before validating, so a failing run still shows what the kernel produced.
    const auto* distance = reinterpret_cast<const float*>(output[0].GetData());
    const auto* index = reinterpret_cast<const int32_t*>(output[1].GetData());
    std::cout << "output[0] topk_distance dtype: " << DataTypeToString(output[0].GetTensorDesc().GetDataType())
              << ", shape size: " << output[0].GetTensorDesc().GetShape().GetShapeSize() << std::endl;
    std::cout << "output[1] topk_index    dtype: " << DataTypeToString(output[1].GetTensorDesc().GetDataType())
              << ", shape size: " << output[1].GetTensorDesc().GetShape().GetShapeSize() << std::endl;

    std::cout << "topk_distance:";
    for (int64_t i = 0; i < kTopK; i++) {
        std::cout << " " << distance[i];
    }
    std::cout << std::endl;
    std::cout << "topk_index   :";
    for (int64_t i = 0; i < kTopK; i++) {
        std::cout << " " << index[i];
    }
    std::cout << std::endl;

    std::cout << "expect_distance:";
    for (int64_t i = 0; i < kTopK; i++) {
        std::cout << " " << kExpectDistance[i];
    }
    std::cout << std::endl;
    std::cout << "expect_index   :";
    for (int64_t i = 0; i < kTopK; i++) {
        std::cout << " " << kExpectIndex[i];
    }
    std::cout << std::endl;

    int ret = SUCCESS;
    for (int64_t i = 0; i < kTopK; i++) {
        if (distance[i] != kExpectDistance[i]) {
            std::cout << "ERROR: topk_distance[" << i << "] is " << distance[i] << ", expect " << kExpectDistance[i]
                      << std::endl;
            ret = FAILED;
        }
        if (index[i] != kExpectIndex[i]) {
            std::cout << "ERROR: topk_index[" << i << "] is " << index[i] << ", expect " << kExpectIndex[i]
                      << std::endl;
            ret = FAILED;
        }
    }
    return ret;
}

void PrintGeMessage()
{
    // On success these are empty; printing them unconditionally would emit confusing blank error lines.
    std::string errorStr(ge::GEGetErrorMsgV2().GetString());
    if (!errorStr.empty()) {
        std::cout << "Error message: " << errorStr << std::endl;
    }
    std::string warningStr(ge::GEGetWarningMsgV2().GetString());
    if (!warningStr.empty()) {
        std::cout << "Warning message: " << warningStr << std::endl;
    }
}
} // namespace

int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;
    const char* graph_name = "tc_ge_irrun_test";
    Graph graph(graph_name);
    vector<ge::Tensor> input;

    printf("%s - INFO - [XIR]: Start to initialize ge using ge global options\n", GetTime().c_str());
    map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Initialize ge using ge global options failed\n", GetTime().c_str());
        return FAILED;
    }

    vector<Operator> inputs{};
    vector<Operator> outputs{};
    if (CreateOppInGraph(input, inputs, outputs, graph) != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create graph failed\n", GetTime().c_str());
        (void)ge::GEFinalize();
        return FAILED;
    }
    graph.SetInputs(inputs).SetOutputs(outputs);

    map<AscendString, AscendString> build_options = {};
    ge::Session* session = new (std::nothrow) Session(build_options);
    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        (void)ge::GEFinalize();
        return FAILED;
    }

    map<AscendString, AscendString> graph_options = {};
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Session add ir compute graph failed\n", GetTime().c_str());
        PrintGeMessage();
        delete session;
        (void)ge::GEFinalize();
        return FAILED;
    }

    printf("%s - INFO - [XIR]: Start to run ir compute graph\n", GetTime().c_str());
    vector<ge::Tensor> output;
    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Run graph failed\n", GetTime().c_str());
        PrintGeMessage();
        delete session;
        (void)ge::GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Session run ir compute graph success\n", GetTime().c_str());

    int check_ret = CheckOutput(output);
    PrintGeMessage();

    delete session;
    if (ge::GEFinalize() != SUCCESS) {
        printf("%s - ERROR - [XIR]: Finalize ir graph session failed\n", GetTime().c_str());
        return FAILED;
    }

    if (check_ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: TopKPQDistanceV2 example result check failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: TopKPQDistanceV2 example result check success\n", GetTime().c_str());
    return SUCCESS;
}
