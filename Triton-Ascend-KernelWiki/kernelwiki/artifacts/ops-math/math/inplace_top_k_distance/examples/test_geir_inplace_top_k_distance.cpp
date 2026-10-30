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

#include "../op_graph/inplace_top_k_distance_proto.h"

#define FAILED (-1)
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

namespace {
// InplaceTopKDistance supports only float16 / float for the distance inputs, so the float32 path is used here:
// it is the widest floating-point type the operator accepts and keeps the values exactly representable.
//
// The operator declares no output: it rewrites topk_pq_distance / topk_pq_index / topk_pq_ivf in place. The graph
// therefore drives it as a target node (side-effect only), and the merged result stays in device memory rather than
// coming back through RunGraph. Value correctness is covered by the AICPU UT; this example proves the custom AICPU
// kernel is dispatched and executes on device.
constexpr int64_t kTopkNum = 5;
constexpr int64_t kPqNum = 3;

// Tie-free distances: merging {1,3,5,7,9} with {2,4,6} and keeping the 5 smallest yields {1,2,3,4,5}.
const float kTopkPqDistance[kTopkNum] = {1.0f, 3.0f, 5.0f, 7.0f, 9.0f};
const int32_t kTopkPqIndex[kTopkNum] = {10, 11, 12, 13, 14};
const int32_t kTopkPqIvf[kTopkNum] = {3, 3, 3, 3, 3};
const float kPqDistance[kPqNum] = {2.0f, 4.0f, 6.0f};
const int32_t kPqIndex[kPqNum] = {20, 21, 22};
const int32_t kPqIvf = 9;

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
    return Tensor(desc, reinterpret_cast<const uint8_t*>(data), byteSize);
}

op::Data BuildDataNode(const string& name, int32_t index, const vector<int64_t>& shape, DataType dtype,
                       const void* data, size_t byteSize, vector<ge::Tensor>& input, vector<Operator>& inputs,
                       Graph& graph)
{
    TensorDesc desc(ge::Shape(shape), FORMAT_ND, dtype);
    desc.SetPlacement(ge::kPlacementHost);
    auto node = op::Data(name.c_str()).set_attr_index(index);
    node.update_input_desc_x(desc);
    node.update_output_desc_y(desc);
    input.push_back(BuildTensor(shape, dtype, data, byteSize));
    graph.AddOp(node);
    inputs.push_back(node);
    return node;
}

void PrintInputs()
{
    std::cout << "input topk_pq_distance dtype: " << DataTypeToString(DT_FLOAT) << ", value:";
    for (int64_t i = 0; i < kTopkNum; i++) {
        std::cout << " " << kTopkPqDistance[i];
    }
    std::cout << std::endl;
    std::cout << "input topk_pq_index    dtype: " << DataTypeToString(DT_INT32) << ", value:";
    for (int64_t i = 0; i < kTopkNum; i++) {
        std::cout << " " << kTopkPqIndex[i];
    }
    std::cout << std::endl;
    std::cout << "input pq_distance      dtype: " << DataTypeToString(DT_FLOAT) << ", value:";
    for (int64_t i = 0; i < kPqNum; i++) {
        std::cout << " " << kPqDistance[i];
    }
    std::cout << std::endl;
    std::cout << "input pq_index         dtype: " << DataTypeToString(DT_INT32) << ", value:";
    for (int64_t i = 0; i < kPqNum; i++) {
        std::cout << " " << kPqIndex[i];
    }
    std::cout << std::endl;
    std::cout << "input pq_ivf           dtype: " << DataTypeToString(DT_INT32) << ", value: " << kPqIvf << std::endl;
    std::cout << "attr order: asc, expected in-place merge result: topk_pq_distance = 1 2 3 4 5, "
              << "topk_pq_index = 10 20 11 21 12, topk_pq_ivf = 3 9 3 9 3" << std::endl;
}

int CreateOppInGraph(vector<ge::Tensor>& input, vector<Operator>& inputs, vector<Operator>& targets, Graph& graph)
{
    vector<int64_t> topkShape = {kTopkNum};
    vector<int64_t> pqShape = {kPqNum};
    vector<int64_t> ivfShape = {};

    auto topkDistanceData = BuildDataNode("topk_pq_distance", 0, topkShape, DT_FLOAT, kTopkPqDistance,
                                          sizeof(kTopkPqDistance), input, inputs, graph);
    auto topkIndexData = BuildDataNode("topk_pq_index", 1, topkShape, DT_INT32, kTopkPqIndex, sizeof(kTopkPqIndex),
                                       input, inputs, graph);
    auto topkIvfData = BuildDataNode("topk_pq_ivf", 2, topkShape, DT_INT32, kTopkPqIvf, sizeof(kTopkPqIvf), input,
                                     inputs, graph);
    auto pqDistanceData = BuildDataNode("pq_distance", 3, pqShape, DT_FLOAT, kPqDistance, sizeof(kPqDistance), input,
                                        inputs, graph);
    auto pqIndexData = BuildDataNode("pq_index", 4, pqShape, DT_INT32, kPqIndex, sizeof(kPqIndex), input, inputs,
                                     graph);
    auto pqIvfData = BuildDataNode("pq_ivf", 5, ivfShape, DT_INT32, &kPqIvf, sizeof(kPqIvf), input, inputs, graph);

    auto inplaceOp = op::InplaceTopKDistance("inplace_top_k_distance");
    inplaceOp.set_input_topk_pq_distance(topkDistanceData);
    inplaceOp.set_input_topk_pq_index(topkIndexData);
    inplaceOp.set_input_topk_pq_ivf(topkIvfData);
    inplaceOp.set_input_pq_distance(pqDistanceData);
    inplaceOp.set_input_pq_index(pqIndexData);
    inplaceOp.set_input_pq_ivf(pqIvfData);
    inplaceOp.set_attr_order("asc");
    graph.AddOp(inplaceOp);

    // The operator has no output, so it is driven as a graph target instead of a graph output.
    targets.push_back(inplaceOp);
    return SUCCESS;
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

    PrintInputs();

    vector<Operator> inputs{};
    vector<Operator> targets{};
    if (CreateOppInGraph(input, inputs, targets, graph) != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create graph failed\n", GetTime().c_str());
        (void)ge::GEFinalize();
        return FAILED;
    }
    graph.SetInputs(inputs).SetTargets(targets);

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
    std::cout << "graph output tensor num: " << output.size()
              << " (InplaceTopKDistance declares no output; it updates its inputs in place on device)" << std::endl;
    PrintGeMessage();

    delete session;
    if (ge::GEFinalize() != SUCCESS) {
        printf("%s - ERROR - [XIR]: Finalize ir graph session failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: InplaceTopKDistance example run success\n", GetTime().c_str());
    return SUCCESS;
}
