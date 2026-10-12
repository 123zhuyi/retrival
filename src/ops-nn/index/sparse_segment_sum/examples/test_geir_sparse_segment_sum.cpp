/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <fstream>
#include <iostream>
#include <map>
#include <new>
#include <stdint.h>
#include <string>
#include <time.h>
#include <vector>

#include "array_ops.h"
#include "ge_api.h"
#include "ge_api_types.h"
#include "ge_error_codes.h"
#include "ge_ir_build.h"
#include "graph.h"
#include "tensor.h"
#include "types.h"

#include "../op_graph/sparse_segment_sum_proto.h"

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
    if (dt == ge::DT_FLOAT) {
        return sizeof(float);
    }
    if (dt == ge::DT_INT32) {
        return sizeof(int32_t);
    }
    return 1;
}

template <typename T>
int32_t GenData(const vector<int64_t>& shape, Tensor& tensor, TensorDesc& desc, const vector<T>& values)
{
    desc.SetRealDimCnt(shape.size());
    const uint8_t* data = values.empty() ? nullptr : reinterpret_cast<const uint8_t*>(values.data());
    tensor = Tensor(desc, data, values.size() * sizeof(T));
    return SUCCESS;
}

int32_t WriteDataToFile(const string& binFile, uint64_t dataSize, uint8_t* inputData)
{
    FILE* fp = fopen(binFile.c_str(), "w");
    if (fp == nullptr) {
        return FAILED;
    }
    fwrite(inputData, sizeof(uint8_t), dataSize, fp);
    fclose(fp);
    return SUCCESS;
}

int CreateGraph(vector<ge::Tensor>& input, vector<Operator>& inputs, vector<Operator>& outputs, Graph& graph)
{
    auto sparseSegmentSum = op::SparseSegmentSum("sparse_segment_sum");

    vector<int64_t> xShape = {3, 4};
    auto xData = op::Data("x").set_attr_index(0);
    TensorDesc xDesc = TensorDesc(ge::Shape(xShape), FORMAT_ND, DT_FLOAT);
    xDesc.SetPlacement(ge::kPlacementHost);
    xDesc.SetFormat(FORMAT_ND);
    Tensor tensorX;
    vector<float> xValues = {1.0f, 2.0f, 3.0f, 4.0f, 9.0f, 10.0f, 11.0f, 12.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    if (GenData(xShape, tensorX, xDesc, xValues) != SUCCESS) {
        return FAILED;
    }
    xData.update_input_desc_x(xDesc);
    xData.update_output_desc_y(xDesc);
    input.push_back(tensorX);
    graph.AddOp(xData);

    vector<int64_t> indicesShape = {3};
    auto indicesData = op::Data("indices").set_attr_index(1);
    TensorDesc indicesDesc = TensorDesc(ge::Shape(indicesShape), FORMAT_ND, DT_INT32);
    indicesDesc.SetPlacement(ge::kPlacementHost);
    indicesDesc.SetFormat(FORMAT_ND);
    Tensor tensorIndices;
    vector<int32_t> indicesValues = {0, 2, 1};
    if (GenData(indicesShape, tensorIndices, indicesDesc, indicesValues) != SUCCESS) {
        return FAILED;
    }
    indicesData.update_input_desc_x(indicesDesc);
    indicesData.update_output_desc_y(indicesDesc);
    input.push_back(tensorIndices);
    graph.AddOp(indicesData);

    vector<int64_t> segmentIdsShape = {3};
    auto segmentIdsData = op::Data("segment_ids").set_attr_index(2);
    TensorDesc segmentIdsDesc = TensorDesc(ge::Shape(segmentIdsShape), FORMAT_ND, DT_INT32);
    segmentIdsDesc.SetPlacement(ge::kPlacementHost);
    segmentIdsDesc.SetFormat(FORMAT_ND);
    Tensor tensorSegmentIds;
    vector<int32_t> segmentIdsValues = {0, 0, 1};
    if (GenData(segmentIdsShape, tensorSegmentIds, segmentIdsDesc, segmentIdsValues) != SUCCESS) {
        return FAILED;
    }
    segmentIdsData.update_input_desc_x(segmentIdsDesc);
    segmentIdsData.update_output_desc_y(segmentIdsDesc);
    input.push_back(tensorSegmentIds);
    graph.AddOp(segmentIdsData);

    sparseSegmentSum.set_input_x(xData);
    sparseSegmentSum.set_input_indices(indicesData);
    sparseSegmentSum.set_input_segment_ids(segmentIdsData);
    TensorDesc yDesc = TensorDesc(ge::Shape({2, 4}), FORMAT_ND, DT_FLOAT);
    sparseSegmentSum.update_output_desc_y(yDesc);

    inputs.push_back(xData);
    inputs.push_back(indicesData);
    inputs.push_back(segmentIdsData);
    outputs.push_back(sparseSegmentSum);
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    const char* graphName = "tc_ge_irrun_test_sparse_segment_sum";
    Graph graph(graphName);
    vector<ge::Tensor> input;

    printf("%s - INFO - [XIR]: Start to initialize ge using ge global options\n", GetTime().c_str());
    map<AscendString, AscendString> globalOptions = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(globalOptions);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Initialize ge failed\n", GetTime().c_str());
        return FAILED;
    }

    vector<Operator> inputs{};
    vector<Operator> outputs{};
    ret = CreateGraph(input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create graph failed\n", GetTime().c_str());
        GEFinalize();
        return FAILED;
    }
    graph.SetInputs(inputs).SetOutputs(outputs);

    map<AscendString, AscendString> buildOptions = {};
    ge::Session* session = new (std::nothrow) Session(buildOptions);
    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create ir session failed\n", GetTime().c_str());
        GEFinalize();
        return FAILED;
    }

    map<AscendString, AscendString> graphOptions = {};
    uint32_t graphId = 0;
    ret = session->AddGraph(graphId, graph, graphOptions);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: AddGraph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }

    string filePath = "./dump";
    aclgrphDumpGraph(graph, filePath.c_str(), filePath.length());

    vector<ge::Tensor> output;
    ret = session->RunGraph(graphId, input, output);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Run graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }

    for (size_t i = 0; i < output.size(); ++i) {
        string outputFile = "./tc_ge_irrun_sparse_segment_sum_output_" + std::to_string(i) + ".bin";
        int64_t outputShapeSize = output[i].GetTensorDesc().GetShape().GetShapeSize();
        uint32_t dataSize = outputShapeSize * GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        WriteDataToFile(outputFile, dataSize, output[i].GetData());
    }

    delete session;
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: GEFinalize failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize success\n", GetTime().c_str());
    return SUCCESS;
}
