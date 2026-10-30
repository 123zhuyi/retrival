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
 * \file test_geir_permute.cpp
 * \brief GEIR example for Permute fusion pass.
 */

#include <cstdio>
#include <cstdint>
#include <ctime>
#include <iostream>
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
#include "transformation_ops.h"
#include "types.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

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
    const uint32_t oneByte = 1;
    const uint32_t twoByte = 2;
    const uint32_t fourByte = 4;
    const uint32_t eightByte = 8;

    if (dt == ge::DT_FLOAT || dt == ge::DT_INT32 || dt == ge::DT_UINT32) {
        return fourByte;
    }
    if (dt == ge::DT_FLOAT16 || dt == ge::DT_BF16 || dt == ge::DT_INT16 || dt == ge::DT_UINT16) {
        return twoByte;
    }
    if (dt == ge::DT_INT64 || dt == ge::DT_UINT64) {
        return eightByte;
    }
    return oneByte;
}

int64_t GetShapeSize(const vector<int64_t>& shape)
{
    int64_t size = 1;
    for (auto dim : shape) {
        size *= dim;
    }
    return size;
}

vector<int64_t> InferPermutedShape(const vector<int64_t>& inputShape, const vector<int64_t>& order)
{
    vector<int64_t> outputShape;
    outputShape.reserve(order.size());
    for (auto axis : order) {
        outputShape.push_back(inputShape[axis]);
    }
    return outputShape;
}

int32_t GenSequentialFloatData(const vector<int64_t>& shape, Tensor& tensor, TensorDesc& tensorDesc)
{
    tensorDesc.SetRealDimCnt(shape.size());
    const int64_t elementCount = GetShapeSize(shape);
    const uint32_t dataLen = elementCount * sizeof(float);
    float* data = new (std::nothrow) float[elementCount];
    if (data == nullptr) {
        return FAILED;
    }

    for (int64_t i = 0; i < elementCount; ++i) {
        data[i] = static_cast<float>(i + 1);
    }
    tensor = Tensor(tensorDesc, reinterpret_cast<uint8_t*>(data), dataLen);
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

int CreateOppInGraph(DataType inDtype, vector<Tensor>& input, vector<Operator>& inputs, vector<Operator>& outputs,
                     Graph& graph)
{
    Status ret = SUCCESS;

    auto permute1 = op::Permute("permute1");
    vector<int64_t> xShape = {2, 3, 4, 5};
    vector<int64_t> order = {0, 3, 2, 1};
    vector<int64_t> yShape = InferPermutedShape(xShape, order);

    auto x = op::Data("placeholder0").set_attr_index(0);
    TensorDesc xDesc = TensorDesc(ge::Shape(xShape), FORMAT_ND, inDtype);
    xDesc.SetPlacement(ge::kPlacementHost);
    xDesc.SetFormat(FORMAT_ND);

    Tensor xTensor;
    ret = GenSequentialFloatData(xShape, xTensor, xDesc);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());
        return FAILED;
    }

    x.update_input_desc_x(xDesc);
    x.update_output_desc_y(xDesc);
    graph.AddOp(x);
    input.push_back(xTensor);
    inputs.push_back(x);

    permute1.set_input_x(x);
    permute1.update_input_desc_x(xDesc);
    permute1.set_attr_order(order);

    TensorDesc yDesc = TensorDesc(ge::Shape(yShape), FORMAT_ND, inDtype);
    yDesc.SetPlacement(ge::kPlacementHost);
    yDesc.SetFormat(FORMAT_ND);
    permute1.update_output_desc_y(yDesc);

    outputs.push_back(permute1);
    return SUCCESS;
}

void SaveInputOutput(vector<Tensor>& input, vector<Tensor>& output)
{
    for (size_t i = 0; i < input.size(); ++i) {
        std::cout << "input " << i << " dtype :  " << input[i].GetTensorDesc().GetDataType() << std::endl;
        string inputFile = "./tc_ge_irrun_permute_npu_input_" + std::to_string(i) + ".bin";
        uint8_t* inputData = input[i].GetData();
        int64_t inputShapeSize = input[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th input, input shape size =" << inputShapeSize << std::endl;
        uint32_t dataSize = inputShapeSize * GetDataTypeSize(input[i].GetTensorDesc().GetDataType());
        WriteDataToFile(inputFile, dataSize, inputData);
    }

    for (size_t i = 0; i < output.size(); ++i) {
        std::cout << "output " << i << " dtype :  " << output[i].GetTensorDesc().GetDataType() << std::endl;
        string outputFile = "./tc_ge_irrun_permute_npu_output_" + std::to_string(i) + ".bin";
        uint8_t* outputData = output[i].GetData();
        int64_t outputShapeSize = output[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th output, output shape size =" << outputShapeSize << std::endl;
        uint32_t dataSize = outputShapeSize * GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        WriteDataToFile(outputFile, dataSize, outputData);

        float* resultData = reinterpret_cast<float*>(outputData);
        for (int64_t j = 0; j < outputShapeSize; ++j) {
            LOG_PRINT("result[%ld] is: %f\n", j, resultData[j]);
        }
    }
}

int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;

    const char* graphName = "tc_ge_irrun_permute";
    Graph graph(graphName);
    vector<Tensor> input;

    printf("%s - INFO - [XIR]: Start to initialize ge using ge global options\n", GetTime().c_str());
    map<AscendString, AscendString> globalOptions = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(globalOptions);
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
        printf("%s - ERROR - [XIR]: Create ir graph failed\n", GetTime().c_str());
        ge::GEFinalize();
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    map<AscendString, AscendString> buildOptions = {

    };
    printf("%s - INFO - [XIR]: Start to create ir session using build options\n", GetTime().c_str());
    ge::Session* session = new (std::nothrow) Session(buildOptions);
    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        ge::GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Create ir session using build options success\n", GetTime().c_str());

    map<AscendString, AscendString> graphOptions = {

    };
    uint32_t graphId = 0;
    printf("%s - INFO - [XIR]: Start to add compute graph to ir session\n", GetTime().c_str());
    ret = session->AddGraph(graphId, graph, graphOptions);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Add graph failed\n", GetTime().c_str());
        delete session;
        ge::GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Session add ir compute graph to ir session success\n", GetTime().c_str());

    printf("%s - INFO - [XIR]: dump graph to txt\n", GetTime().c_str());
    string filePath = "./dump";
    aclgrphDumpGraph(graph, filePath.c_str(), filePath.length());

    printf("%s - INFO - [XIR]: Start to run ir compute graph\n", GetTime().c_str());
    vector<Tensor> output;
    ret = session->RunGraph(graphId, input, output);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Run graph failed\n", GetTime().c_str());
        delete session;
        ge::GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Session run ir compute graph success\n", GetTime().c_str());

    SaveInputOutput(input, output);

    ge::AscendString errorMsg = ge::GEGetErrorMsgV2();
    std::cout << "Error message: " << errorMsg.GetString() << std::endl;
    ge::AscendString warningMsg = ge::GEGetWarningMsgV2();
    std::cout << "Warning message: " << warningMsg.GetString() << std::endl;

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
