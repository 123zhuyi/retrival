/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
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

#include "experiment_ops.h"
#include "nn_other.h"
#include "../op_graph/div_v3_proto.h"

#define FAILED -1
#define SUCCESS 0

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

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
    if (dt == ge::DT_FLOAT || dt == ge::DT_INT32) {
        return 4;
    } else if (dt == ge::DT_FLOAT16 || dt == ge::DT_BF16 || dt == ge::DT_INT16) {
        return 2;
    } else if (dt == ge::DT_INT8) {
        return 1;
    } else if (dt == ge::DT_INT64) {
        return 8;
    }
    return 4;
}

int32_t GenOnesDataFloat32(vector<int64_t> shapes, Tensor& inputTensor, TensorDesc& inputTensorDesc, float value)
{
    inputTensorDesc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    uint32_t dataLen = size * sizeof(float);
    float* pData = new (std::nothrow) float[size];
    for (size_t i = 0; i < size; ++i) {
        *(pData + i) = value;
    }
    inputTensor = Tensor(inputTensorDesc, (uint8_t*)pData, dataLen);
    delete[] pData;
    return SUCCESS;
}

int32_t WriteDataToFile(string binFile, uint64_t dataSize, uint8_t* inputData)
{
    FILE* fp = fopen(binFile.c_str(), "w");
    if (fp == nullptr) {
        return FAILED;
    }
    fwrite(inputData, sizeof(uint8_t), dataSize, fp);
    fclose(fp);
    return SUCCESS;
}

int CreateOppInGraph(
    DataType inDtype, std::vector<ge::Tensor>& input, std::vector<Operator>& inputs, std::vector<Operator>& outputs,
    Graph& graph)
{
    Status ret = SUCCESS;
    std::vector<int64_t> xShape = {4, 256};

    // x1
    auto placeholder1 = op::Data("placeholder1").set_attr_index(0);
    TensorDesc placeholder1Desc(ge::Shape(xShape), FORMAT_ND, inDtype);
    placeholder1Desc.SetPlacement(ge::kPlacementHost);
    Tensor tensor1;
    ret = GenOnesDataFloat32(xShape, tensor1, placeholder1Desc, 7.0f);
    if (ret != SUCCESS) {
        return FAILED;
    }
    placeholder1.update_input_desc_x(placeholder1Desc);
    input.push_back(tensor1);
    graph.AddOp(placeholder1);
    inputs.push_back(placeholder1);

    // x2
    auto placeholder2 = op::Data("placeholder2").set_attr_index(0);
    TensorDesc placeholder2Desc(ge::Shape(xShape), FORMAT_ND, inDtype);
    placeholder2Desc.SetPlacement(ge::kPlacementHost);
    Tensor tensor2;
    ret = GenOnesDataFloat32(xShape, tensor2, placeholder2Desc, 2.0f);
    if (ret != SUCCESS) {
        return FAILED;
    }
    placeholder2.update_input_desc_x(placeholder2Desc);
    input.push_back(tensor2);
    graph.AddOp(placeholder2);
    inputs.push_back(placeholder2);

    // DivV3 op with mode=2 (FloorDiv)
    auto divV3Op = op::DivV3("divv3_op");
    divV3Op.set_input_x1(placeholder1);
    divV3Op.set_input_x2(placeholder2);
    divV3Op.set_attr_mode(2);

    TensorDesc yDesc(ge::Shape(xShape), FORMAT_ND, inDtype);
    divV3Op.update_output_desc_y(yDesc);

    outputs.push_back(divV3Op);
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    const char* graphName = "tc_ge_irrun_div_v3";
    Graph graph(graphName);
    std::vector<ge::Tensor> input;

    printf("%s - INFO - Start to initialize ge\n", GetTime().c_str());
    std::map<AscendString, AscendString> globalOptions = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(globalOptions);
    if (ret != SUCCESS) {
        printf("%s - INFO - Initialize ge failed\n", GetTime().c_str());
        return FAILED;
    }

    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};

    DataType inDtype = DT_FLOAT;
    ret = CreateOppInGraph(inDtype, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - CreateOppInGraph failed\n", GetTime().c_str());
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    std::map<AscendString, AscendString> buildOptions = {};
    ge::Session* session = new Session(buildOptions);
    if (session == nullptr) {
        printf("%s - ERROR - Create session failed\n", GetTime().c_str());
        return FAILED;
    }

    uint32_t graphId = 0;
    std::map<AscendString, AscendString> graphOptions = {};
    ret = session->AddGraph(graphId, graph, graphOptions);

    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graphId, input, output);
    if (ret != SUCCESS) {
        printf("%s - INFO - Run graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - Run graph success\n", GetTime().c_str());

    int outputNum = output.size();
    for (int i = 0; i < outputNum; i++) {
        uint8_t* outputData = output[i].GetData();
        int64_t outputShape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        uint32_t dataSize = outputShape * GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        string outputFile = "./div_v3_output_" + std::to_string(i) + ".bin";
        WriteDataToFile(outputFile, dataSize, outputData);

        float* result = (float*)outputData;
        for (int64_t j = 0; j < 8 && j < outputShape; j++) {
            LOG_PRINT("result[%ld] is: %f\n", j, result[j]);
        }
    }

    ret = ge::GEFinalize();
    printf("%s - INFO - Finalize %s\n", GetTime().c_str(), ret == SUCCESS ? "success" : "failed");
    return SUCCESS;
}
