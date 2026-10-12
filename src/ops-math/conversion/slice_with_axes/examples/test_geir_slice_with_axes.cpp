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
 * \file test_geir_slice_with_axes.cpp
 * \brief GE IR test for SliceWithAxes operator
 *
 * SliceWithAxes: input x[4,8], axes=[1], offsets=[0], size=[4] -> output y[4,4]
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

#include "../op_graph/slice_with_axes_proto.h"

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

int32_t GenFloatData(vector<int64_t> shapes, Tensor& tensor, TensorDesc& desc, DataType dt, float value)
{
    desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (size_t i = 0; i < shapes.size(); i++) {
        size *= static_cast<size_t>(shapes[i]);
    }
    uint32_t type_size = GetDataTypeSize(dt);
    if (type_size == 0U) {
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
            p[i] = value + static_cast<float>(i);
        }
    } else {
        delete[] buf;
        return FAILED;
    }
    tensor = Tensor(desc, buf, data_len);
    delete[] buf;
    return SUCCESS;
}

int32_t GenInt32Data(vector<int64_t> shapes, Tensor& tensor, TensorDesc& desc, const vector<int32_t>& values)
{
    desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (size_t i = 0; i < shapes.size(); i++) {
        size *= static_cast<size_t>(shapes[i]);
    }
    if (values.size() < size) {
        return FAILED;
    }
    uint32_t data_len = static_cast<uint32_t>(size * sizeof(int32_t));
    int32_t* buf = new (std::nothrow) int32_t[size];
    if (buf == nullptr) {
        return FAILED;
    }
    for (size_t i = 0; i < size; i++) {
        buf[i] = values[i];
    }
    tensor = Tensor(desc, reinterpret_cast<uint8_t*>(buf), data_len);
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
        uint8_t* input_data_i = input[i].GetData();
        int64_t input_shape = input[i].GetTensorDesc().GetShape().GetShapeSize();
        uint32_t type_size = GetDataTypeSize(input[i].GetTensorDesc().GetDataType());
        if (type_size == 0U) {
            continue;
        }
        uint32_t data_size = static_cast<uint32_t>(input_shape * type_size);
        WriteDataToFile(input_file, data_size, input_data_i);
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
            continue;
        }
        uint32_t data_size = static_cast<uint32_t>(output_shape * type_size);
        WriteDataToFile(output_file, data_size, output_data_i);
    }
}

int CreateOppInGraph(DataType inDtype, vector<Tensor>& input, vector<Operator>& inputs, vector<Operator>& outputs,
                     Graph& graph)
{
    Status ret = SUCCESS;
    auto node = op::SliceWithAxes("slice_with_axes_1");

    vector<int64_t> xShape = {4, 8};
    vector<int64_t> offsetsShape = {1};
    vector<int64_t> sizeShape = {1};
    vector<int64_t> yShape = {4, 4};

    TensorDesc xDesc(ge::Shape(xShape), FORMAT_ND, inDtype);
    xDesc.SetPlacement(ge::kPlacementHost);
    xDesc.SetFormat(FORMAT_ND);
    Tensor xTensor;
    ret = GenFloatData(xShape, xTensor, xDesc, inDtype, 1.0f);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate x data failed\n", GetTime().c_str());
        return FAILED;
    }
    auto xData = op::Data("x_data").set_attr_index(0);
    xData.update_input_desc_x(xDesc);
    xData.update_output_desc_y(xDesc);
    input.push_back(xTensor);
    graph.AddOp(xData);
    node.set_input_x(xData);
    inputs.push_back(xData);

    TensorDesc offsetsDesc(ge::Shape(offsetsShape), FORMAT_ND, DT_INT32);
    offsetsDesc.SetPlacement(ge::kPlacementHost);
    offsetsDesc.SetFormat(FORMAT_ND);
    Tensor offsetsTensor;
    ret = GenInt32Data(offsetsShape, offsetsTensor, offsetsDesc, {0});
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate offsets data failed\n", GetTime().c_str());
        return FAILED;
    }
    auto offsetsConst = op::Const("offsets_data");
    offsetsConst.SetAttr("value", offsetsTensor);
    offsetsConst.update_output_desc_y(offsetsDesc);
    graph.AddOp(offsetsConst);
    node.set_input_offsets(offsetsConst);
    node.update_input_desc_offsets(offsetsDesc);

    TensorDesc sizeDesc(ge::Shape(sizeShape), FORMAT_ND, DT_INT32);
    sizeDesc.SetPlacement(ge::kPlacementHost);
    sizeDesc.SetFormat(FORMAT_ND);
    Tensor sizeTensor;
    ret = GenInt32Data(sizeShape, sizeTensor, sizeDesc, {4});
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate size data failed\n", GetTime().c_str());
        return FAILED;
    }
    auto sizeConst = op::Const("size_data");
    sizeConst.SetAttr("value", sizeTensor);
    sizeConst.update_output_desc_y(sizeDesc);
    graph.AddOp(sizeConst);
    node.set_input_size(sizeConst);
    node.update_input_desc_size(sizeDesc);

    vector<int64_t> axesValue = {1};
    node.SetAttr("axes", axesValue);

    TensorDesc yDesc(ge::Shape(yShape), FORMAT_ND, inDtype);
    node.update_output_desc_y(yDesc);

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
