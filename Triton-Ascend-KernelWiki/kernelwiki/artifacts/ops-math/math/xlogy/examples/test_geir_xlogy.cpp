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
 * \file test_geir_xlogy.cpp
 * \brief
 */
#include <iostream>
#include <fstream>
#include <string.h>
#include <stdint.h>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include "assert.h"

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "array_ops.h"
#include "ge_ir_build.h"

#include "../op_graph/xlogy_proto.h"

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
    if (dt == ge::DT_FLOAT)
        return 4;
    if (dt == ge::DT_FLOAT16 || dt == ge::DT_BF16)
        return 2;
    if (dt == ge::DT_INT32 || dt == ge::DT_UINT32)
        return 4;
    if (dt == ge::DT_INT64 || dt == ge::DT_UINT64)
        return 8;
    if (dt == ge::DT_INT8 || dt == ge::DT_UINT8)
        return 1;
    return 1;
}

int32_t GenFloatData(
    vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, DataType data_type,
    const vector<float>& values)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++)
        size *= shapes[i];
    uint32_t data_len = size * sizeof(float);
    float* pData = new (std::nothrow) float[size];
    for (size_t i = 0; i < size; ++i)
        pData[i] = values[i % values.size()];
    input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t*>(pData), data_len);
    return SUCCESS;
}

int32_t WriteDataToFile(string bin_file, uint64_t data_size, uint8_t* inputData)
{
    FILE* fp = fopen(bin_file.c_str(), "w");
    fwrite(inputData, sizeof(uint8_t), data_size, fp);
    fclose(fp);
    return SUCCESS;
}

int CreateOppInGraph(
    DataType inDtype, std::vector<ge::Tensor>& input, std::vector<Operator>& inputs, std::vector<Operator>& outputs,
    Graph& graph)
{
    Status ret = SUCCESS;
    auto xlogyOp = op::Xlogy("xlogy1");

    std::vector<int64_t> xShape = {2, 3};

    // Input x1
    {
        auto data = op::Data("x1_data").set_attr_index(0);
        TensorDesc desc = TensorDesc(ge::Shape(xShape), FORMAT_ND, inDtype);
        desc.SetPlacement(ge::kPlacementHost);
        desc.SetFormat(FORMAT_ND);
        Tensor tensor;
        vector<float> xVals = {1.0f, 2.0f, 3.0f, 0.0f, 5.0f, 6.0f};
        ret = GenFloatData(xShape, tensor, desc, inDtype, xVals);
        if (ret != SUCCESS) {
            printf("%s - ERROR: Generate x1 data failed\n", GetTime().c_str());
            return FAILED;
        }
        data.update_input_desc_x(desc);
        input.push_back(tensor);
        graph.AddOp(data);
        xlogyOp.set_input_x1(data);
        inputs.push_back(data);
    }

    // Input x2 (y in the formula)
    {
        auto data = op::Data("x2_data").set_attr_index(1);
        TensorDesc desc = TensorDesc(ge::Shape(xShape), FORMAT_ND, inDtype);
        desc.SetPlacement(ge::kPlacementHost);
        desc.SetFormat(FORMAT_ND);
        Tensor tensor;
        vector<float> yVals = {2.0f, 3.0f, 4.0f, 5.0f, 1.0f, 2.0f};
        ret = GenFloatData(xShape, tensor, desc, inDtype, yVals);
        if (ret != SUCCESS) {
            printf("%s - ERROR: Generate x2 data failed\n", GetTime().c_str());
            return FAILED;
        }
        data.update_input_desc_x(desc);
        input.push_back(tensor);
        graph.AddOp(data);
        xlogyOp.set_input_x2(data);
        inputs.push_back(data);
    }

    TensorDesc outDesc = TensorDesc(ge::Shape(xShape), FORMAT_ND, inDtype);
    xlogyOp.update_output_desc_y(outDesc);
    outputs.push_back(xlogyOp);
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    const char* graph_name = "tc_ge_irrun_xlogy_test";
    Graph graph(graph_name);
    std::vector<ge::Tensor> input;

    printf("%s - [XIR]: Initializing GE\n", GetTime().c_str());
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - [XIR]: GEInitialize failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - [XIR]: GEInitialize success\n", GetTime().c_str());

    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};
    DataType inDtype = DT_FLOAT;

    ret = CreateOppInGraph(inDtype, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - [XIR]: CreateOppInGraph failed\n", GetTime().c_str());
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    std::map<AscendString, AscendString> build_options = {};
    printf("%s - [XIR]: Creating IR session\n", GetTime().c_str());
    ge::Session* session = new Session(build_options);
    if (session == nullptr) {
        printf("%s - [XIR]: Create session failed\n", GetTime().c_str());
        return FAILED;
    }

    std::map<AscendString, AscendString> graph_options = {};
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);
    printf("%s - [XIR]: AddGraph success\n", GetTime().c_str());

    std::string file_path = "./dump_xlogy";
    aclgrphDumpGraph(graph, file_path.c_str(), file_path.length());

    printf("%s - [XIR]: Running xlogy graph\n", GetTime().c_str());
    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        printf("%s - [XIR]: RunGraph failed\n", GetTime().c_str());
        ge::AscendString error_msg = ge::GEGetErrorMsgV2();
        std::string error_str(error_msg.GetString());
        std::cout << "GE Error: " << error_str << std::endl;
        delete session;
        GEFinalize();
        return FAILED;
    }
    printf("%s - [XIR]: RunGraph success\n", GetTime().c_str());

    // Save input/output to bin files
    int input_num = input.size();
    for (int i = 0; i < input_num; i++) {
        string input_file = "./xlogy_input_" + std::to_string(i) + ".bin";
        uint8_t* data = input[i].GetData();
        int64_t shape_size = input[i].GetTensorDesc().GetShape().GetShapeSize();
        uint32_t data_size = shape_size * GetDataTypeSize(input[i].GetTensorDesc().GetDataType());
        WriteDataToFile(input_file, data_size, data);
    }

    int output_num = output.size();
    printf("\n========== GEIR NPU Output ==========\n");
    for (int i = 0; i < output_num; i++) {
        string output_file = "./xlogy_output_" + std::to_string(i) + ".bin";
        uint8_t* data = output[i].GetData();
        int64_t shape_size = output[i].GetTensorDesc().GetShape().GetShapeSize();
        uint32_t data_size = shape_size * GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        WriteDataToFile(output_file, data_size, data);

        // Read back and print
        float* floatData = reinterpret_cast<float*>(data);
        printf("Output tensor[%d] (%ld elements):\n", i, shape_size);
        for (int64_t j = 0; j < shape_size; j++) {
            printf("  [%ld] NPU=%.6f\n", j, floatData[j]);
        }
    }
    printf("======================================\n");

    // Print expected golden values
    printf("\n========== Golden Reference ==========\n");
    float x1Vals[] = {1.0f, 2.0f, 3.0f, 0.0f, 5.0f, 6.0f};
    float x2Vals[] = {2.0f, 3.0f, 4.0f, 5.0f, 1.0f, 2.0f};
    int allMatch = 1;
    for (int i = 0; i < 6; i++) {
        float golden = (x1Vals[i] == 0.0f) ? 0.0f : x1Vals[i] * logf(x2Vals[i]);
        float npuVal = reinterpret_cast<float*>(output[0].GetData())[i];
        int match = (fabs(npuVal - golden) < 1e-5f);
        if (!match)
            allMatch = 0;
        printf(
            "  [%d] x1=%.1f x2=%.1f -> NPU=%.6f golden=%.6f %s\n", i, x1Vals[i], x2Vals[i], npuVal, golden,
            match ? "OK" : "MISMATCH");
    }
    printf("======================================\n");
    printf("Result: %s\n", allMatch ? "ALL PASS" : "FAIL");

    ge::AscendString error_msg = ge::GEGetErrorMsgV2();
    std::string error_str(error_msg.GetString());
    std::cout << "GE Error: " << error_str << std::endl;

    printf("%s - [XIR]: Finalizing GE\n", GetTime().c_str());
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - [XIR]: GEFinalize failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - [XIR]: Success\n", GetTime().c_str());
    return allMatch ? SUCCESS : FAILED;
}
