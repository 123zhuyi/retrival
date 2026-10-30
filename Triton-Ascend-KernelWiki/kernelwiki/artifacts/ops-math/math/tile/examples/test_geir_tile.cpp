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
 * \file test_geir_tile.cpp
 * \brief GE IR test for Tile operator
 */

#include <iostream>
#include <fstream>
#include <string.h>
#include <stdint.h>
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
#include "nn_other.h"
#include "../op_graph/tile_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

#define ADD_INPUT(inputIndex, inputName, inputDtype, inputShape, inputValues)                                       \
    do {                                                                                                             \
        std::string name##inputIndex = "placeholder" + std::to_string(inputIndex);                                  \
        auto placeholder##inputIndex = op::Data(name##inputIndex.c_str()).set_attr_index(inputIndex - 1);          \
        TensorDesc placeholder##inputIndex##_desc =                                                                  \
            TensorDesc(ge::Shape(inputShape), FORMAT_ND, inputDtype);                                               \
        placeholder##inputIndex##_desc.SetPlacement(ge::kPlacementHost);                                            \
        placeholder##inputIndex##_desc.SetFormat(FORMAT_ND);                                                        \
        Tensor tensor_placeholder##inputIndex;                                                                      \
        ret = GenData(inputShape, tensor_placeholder##inputIndex,                                                   \
            placeholder##inputIndex##_desc, inputDtype, inputValues);                                               \
        if (ret != SUCCESS) {                                                                                        \
            printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());                          \
            return FAILED;                                                                                           \
        }                                                                                                            \
        placeholder##inputIndex.update_input_desc_x(placeholder##inputIndex##_desc);                                \
        placeholder##inputIndex.update_output_desc_y(placeholder##inputIndex##_desc);                                \
        input.push_back(tensor_placeholder##inputIndex);                                                            \
        graph.AddOp(placeholder##inputIndex);                                                                       \
        tile1.set_input_##inputName(placeholder##inputIndex);                                                        \
        inputs.push_back(placeholder##inputIndex);                                                                  \
    } while (0)

#define ADD_CONST_INPUT(inputIndex, inputName, inputDtype, inputShape, inputValues)                                  \
    do {                                                                                                             \
        std::string name##inputIndex = "placeholder" + std::to_string(inputIndex);                                  \
        auto placeholder##inputIndex = op::Const(name##inputIndex.c_str());                                        \
        TensorDesc placeholder##inputIndex##_desc =                                                                  \
            TensorDesc(ge::Shape(inputShape), FORMAT_ND, inputDtype);                                               \
        placeholder##inputIndex##_desc.SetPlacement(ge::kPlacementHost);                                            \
        placeholder##inputIndex##_desc.SetFormat(FORMAT_ND);                                                        \
        Tensor tensor_placeholder##inputIndex;                                                                      \
        ret = GenData(inputShape, tensor_placeholder##inputIndex,                                                   \
            placeholder##inputIndex##_desc, inputDtype, inputValues);                                               \
        if (ret != SUCCESS) {                                                                                        \
            printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());                          \
            return FAILED;                                                                                           \
        }                                                                                                            \
        placeholder##inputIndex.SetAttr("value", tensor_placeholder##inputIndex);                                   \
        placeholder##inputIndex.update_output_desc_y(placeholder##inputIndex##_desc);                                \
        graph.AddOp(placeholder##inputIndex);                                                                       \
        tile1.set_input_##inputName(placeholder##inputIndex);                                                        \
        tile1.update_input_desc_##inputName(placeholder##inputIndex##_desc);                                         \
        inputs.push_back(placeholder##inputIndex);                                                                  \
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
    switch (dt) {
        case ge::DT_BOOL:   return 1U;
        case ge::DT_INT8:   return 1U;
        case ge::DT_UINT8:  return 1U;
        case ge::DT_INT16:  return 2U;
        case ge::DT_UINT16: return 2U;
        case ge::DT_INT32:  return 4U;
        case ge::DT_UINT32: return 4U;
        case ge::DT_INT64:  return 8U;
        case ge::DT_UINT64: return 8U;
        case ge::DT_FLOAT:  return 4U;
        case ge::DT_DOUBLE: return 8U;
        default: return 0U;
    }
}

int32_t GenData(
    const vector<int64_t>& shapes, Tensor& inputTensor, TensorDesc& inputTensorDesc,
    DataType dataType, const vector<double>& values)
{
    uint32_t typeSize = GetDataTypeSize(dataType);
    if (typeSize == 0U) {
        printf("ERROR: data_type %d is not supported by GenData (no standard C++ type mapping)\n",
               static_cast<int>(dataType));
        return FAILED;
    }

    inputTensorDesc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) { size *= shapes[i]; }
    if (size != values.size()) {
        printf("ERROR: GenData shape size %zu != values size %zu\n", size, values.size());
        return FAILED;
    }

    uint32_t dataLen = size * typeSize;
    uint8_t* pData = new (std::nothrow) uint8_t[dataLen];
    if (pData == nullptr) { return FAILED; }

    switch (dataType) {
        case ge::DT_BOOL: {
            bool* data = reinterpret_cast<bool*>(pData);
            for (size_t i = 0; i < size; ++i) data[i] = (values[i] != 0);
            break;
        }
        case ge::DT_INT8: {
            int8_t* data = reinterpret_cast<int8_t*>(pData);
            for (size_t i = 0; i < size; ++i) data[i] = static_cast<int8_t>(values[i]);
            break;
        }
        case ge::DT_UINT8: {
            uint8_t* data = pData;
            for (size_t i = 0; i < size; ++i) data[i] = static_cast<uint8_t>(values[i]);
            break;
        }
        case ge::DT_INT16: {
            int16_t* data = reinterpret_cast<int16_t*>(pData);
            for (size_t i = 0; i < size; ++i) data[i] = static_cast<int16_t>(values[i]);
            break;
        }
        case ge::DT_UINT16: {
            uint16_t* data = reinterpret_cast<uint16_t*>(pData);
            for (size_t i = 0; i < size; ++i) data[i] = static_cast<uint16_t>(values[i]);
            break;
        }
        case ge::DT_INT32: {
            int32_t* data = reinterpret_cast<int32_t*>(pData);
            for (size_t i = 0; i < size; ++i) data[i] = static_cast<int32_t>(values[i]);
            break;
        }
        case ge::DT_UINT32: {
            uint32_t* data = reinterpret_cast<uint32_t*>(pData);
            for (size_t i = 0; i < size; ++i) data[i] = static_cast<uint32_t>(values[i]);
            break;
        }
        case ge::DT_INT64: {
            int64_t* data = reinterpret_cast<int64_t*>(pData);
            for (size_t i = 0; i < size; ++i) data[i] = static_cast<int64_t>(values[i]);
            break;
        }
        case ge::DT_UINT64: {
            uint64_t* data = reinterpret_cast<uint64_t*>(pData);
            for (size_t i = 0; i < size; ++i) data[i] = static_cast<uint64_t>(values[i]);
            break;
        }
        case ge::DT_FLOAT: {
            float* data = reinterpret_cast<float*>(pData);
            for (size_t i = 0; i < size; ++i) data[i] = static_cast<float>(values[i]);
            break;
        }
        case ge::DT_DOUBLE: {
            double* data = reinterpret_cast<double*>(pData);
            for (size_t i = 0; i < size; ++i) data[i] = values[i];
            break;
        }
        // FLOAT16/BF16/COMPLEX blocked by typeSize==0 check above
        default:
            delete[] pData;
            return FAILED;
    }

    inputTensor = Tensor(inputTensorDesc, pData, dataLen);
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

int CreateOppInGraph(DataType inDtype, vector<Tensor>& input, vector<Operator>& inputs, vector<Operator>& outputs, Graph& graph)
{
    Status ret = SUCCESS;
    auto tile1 = op::Tile("tile1");

    vector<int64_t> xShape = {2, 3};
    vector<int64_t> multiplesShape = {2};
    vector<double> xValues = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    vector<double> multiplesValues = {2.0, 2.0};

    ADD_INPUT(1, x, inDtype, xShape, xValues);
    ADD_CONST_INPUT(2, multiples, DT_INT32, multiplesShape, multiplesValues);

    outputs.push_back(tile1);
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    printf("%s - INFO - [XIR]: Start to initialize ge using ge global options\n", GetTime().c_str());
    map<AscendString, AscendString> globalOptions = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(globalOptions);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Initialize ge using ge global options failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Initialize ge using ge global options success\n", GetTime().c_str());

    DataType inDtype = DT_INT32;
    std::cout << inDtype << std::endl;

    const char* graphName = "tile_ge_irrun_test";
    Graph graph(graphName);
    vector<Tensor> input;
    vector<Operator> inputs{};
    vector<Operator> outputs{};

    ret = CreateOppInGraph(inDtype, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create ir graph failed\n", GetTime().c_str());
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    map<AscendString, AscendString> buildOptions = {};
    printf("%s - INFO - [XIR]: Start to create ir session using build options\n", GetTime().c_str());
    ge::Session* session = new Session(buildOptions);
    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create ir session failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Create ir session using build options success\n", GetTime().c_str());

    printf("%s - INFO - [XIR]: Start to add compute graph to ir session\n", GetTime().c_str());
    map<AscendString, AscendString> graphOptions = {};
    uint32_t graphId = 0;
    ret = session->AddGraph(graphId, graph, graphOptions);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Add graph failed\n", GetTime().c_str());
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
        printf("%s - ERROR - [XIR]: Run graph failed\n", GetTime().c_str());
        delete session;
        ge::GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Session run ir compute graph success\n", GetTime().c_str());

    int inputNum = input.size();
    for (int i = 0; i < inputNum; i++) {
        string inputFile = "./tile_ge_irrun_test_npu_input_" + std::to_string(i) + ".bin";
        uint8_t* inputData = input[i].GetData();
        int64_t inputShape = input[i].GetTensorDesc().GetShape().GetShapeSize();
        uint32_t typeSize = GetDataTypeSize(input[i].GetTensorDesc().GetDataType());
        if (typeSize == 0U) {
            printf("ERROR: input %d has unsupported dtype\n", i);
            continue;
        }
        uint32_t dataSize = inputShape * typeSize;
        WriteDataToFile(inputFile, dataSize, inputData);
    }

    int outputNum = output.size();
    for (int i = 0; i < outputNum; i++) {
        string outputFile = "./tile_ge_irrun_test_npu_output_" + std::to_string(i) + ".bin";
        uint8_t* outputData = output[i].GetData();
        int64_t outputShape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        uint32_t typeSize = GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        if (typeSize == 0U) {
            printf("ERROR: output %d has unsupported dtype\n", i);
            continue;
        }
        uint32_t dataSize = outputShape * typeSize;
        WriteDataToFile(outputFile, dataSize, outputData);
    }

    ge::AscendString errorMsg = ge::GEGetErrorMsgV2();
    string errorStr(errorMsg.GetString());
    std::cout << "Error message: " << errorStr << std::endl;
    ge::AscendString warningMsg = ge::GEGetWarningMsgV2();
    string warningStr(warningMsg.GetString());
    std::cout << "Warning message: " << warningStr << std::endl;

    delete session;

    printf("%s - INFO - [XIR]: Start to finalize ir graph session\n", GetTime().c_str());
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Finalize ir graph session failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize ir graph session success\n", GetTime().c_str());
    return SUCCESS;
}