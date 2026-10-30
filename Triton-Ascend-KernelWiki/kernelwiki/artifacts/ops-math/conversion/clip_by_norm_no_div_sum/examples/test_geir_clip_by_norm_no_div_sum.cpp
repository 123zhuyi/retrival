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
 * \file test_geir_clip_by_norm_no_div_sum.cpp
 * \brief GE IR (Graph mode) test for ClipByNormNoDivSum (4 inputs, 1 output)
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

#include "conversion/clip_by_norm_no_div_sum/op_graph/clip_by_norm_no_div_sum_proto.h"

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
    return 4;
}

int32_t GenFloatData(
    vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, const vector<float>& values)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++)
        size *= shapes[i];
    uint32_t data_len = size * sizeof(float);
    float* pData = new (std::nothrow) float[size];
    for (size_t i = 0; i < size; ++i)
        pData[i] = values[i % values.size()];
    input_tensor = Tensor(input_tensor_desc, static_cast<uint8_t*>(pData), data_len);
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
    auto clipOp = op::ClipByNormNoDivSum("clip_by_norm_no_div_sum1");
    std::vector<int64_t> shape = {8};

    // Input 0: x
    {
        auto data = op::Data("x_data").set_attr_index(0);
        TensorDesc desc(ge::Shape(shape), FORMAT_ND, inDtype);
        desc.SetPlacement(ge::kPlacementHost);
        desc.SetFormat(FORMAT_ND);
        Tensor tensor;
        vector<float> vals = {-1.0f, 0.0f, 0.25f, 0.5f, 2.0f, 4.0f, -0.5f, 9.0f};
        ret = GenFloatData(shape, tensor, desc, vals);
        if (ret != SUCCESS) {
            printf("Generate x data failed\n");
            return FAILED;
        }
        data.update_input_desc_x(desc);
        input.push_back(tensor);
        graph.AddOp(data);
        clipOp.set_input_x(data);
        inputs.push_back(data);
    }

    // Input 1: greater_zeros
    {
        auto data = op::Data("greater_zeros_data").set_attr_index(1);
        TensorDesc desc(ge::Shape(shape), FORMAT_ND, inDtype);
        desc.SetPlacement(ge::kPlacementHost);
        desc.SetFormat(FORMAT_ND);
        Tensor tensor;
        vector<float> vals = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        ret = GenFloatData(shape, tensor, desc, vals);
        if (ret != SUCCESS) {
            printf("Generate greater_zeros data failed\n");
            return FAILED;
        }
        data.update_input_desc_x(desc);
        input.push_back(tensor);
        graph.AddOp(data);
        clipOp.set_input_greater_zeros(data);
        inputs.push_back(data);
    }

    // Input 2: select_ones
    {
        auto data = op::Data("select_ones_data").set_attr_index(2);
        TensorDesc desc(ge::Shape(shape), FORMAT_ND, inDtype);
        desc.SetPlacement(ge::kPlacementHost);
        desc.SetFormat(FORMAT_ND);
        Tensor tensor;
        vector<float> vals = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
        ret = GenFloatData(shape, tensor, desc, vals);
        if (ret != SUCCESS) {
            printf("Generate select_ones data failed\n");
            return FAILED;
        }
        data.update_input_desc_x(desc);
        input.push_back(tensor);
        graph.AddOp(data);
        clipOp.set_input_select_ones(data);
        inputs.push_back(data);
    }

    // Input 3: maximum_ones
    {
        auto data = op::Data("maximum_ones_data").set_attr_index(3);
        TensorDesc desc(ge::Shape(shape), FORMAT_ND, inDtype);
        desc.SetPlacement(ge::kPlacementHost);
        desc.SetFormat(FORMAT_ND);
        Tensor tensor;
        vector<float> vals = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
        ret = GenFloatData(shape, tensor, desc, vals);
        if (ret != SUCCESS) {
            printf("Generate maximum_ones data failed\n");
            return FAILED;
        }
        data.update_input_desc_x(desc);
        input.push_back(tensor);
        graph.AddOp(data);
        clipOp.set_input_maximum_ones(data);
        inputs.push_back(data);
    }

    TensorDesc outDesc(ge::Shape(shape), FORMAT_ND, inDtype);
    clipOp.update_output_desc_y(outDesc);
    outputs.push_back(clipOp);
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    const char* graph_name = "tc_ge_clip_by_norm_no_div_sum";
    Graph graph(graph_name);
    std::vector<ge::Tensor> input;

    printf("%s - INFO - [GEIR]: Start GEInitialize\n", GetTime().c_str());
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [GEIR]: GEInitialize failed\n", GetTime().c_str());
        return FAILED;
    }

    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};
    DataType inDtype = DT_FLOAT;

    ret = CreateOppInGraph(inDtype, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [GEIR]: CreateOppInGraph failed\n", GetTime().c_str());
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    std::map<AscendString, AscendString> build_options = {};
    printf("%s - INFO - [GEIR]: Creating session\n", GetTime().c_str());
    ge::Session* session = new Session(build_options);
    if (session == nullptr) {
        printf("%s - ERROR - [GEIR]: Create session failed\n", GetTime().c_str());
        return FAILED;
    }

    std::map<AscendString, AscendString> graph_options = {};
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);
    printf("%s - INFO - [GEIR]: AddGraph done\n", GetTime().c_str());

    std::string file_path = "./dump_clip_by_norm_no_div_sum";
    aclgrphDumpGraph(graph, file_path.c_str(), file_path.length());

    printf("%s - INFO - [GEIR]: Running graph...\n", GetTime().c_str());
    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [GEIR]: RunGraph failed\n", GetTime().c_str());
        ge::AscendString err = ge::GEGetErrorMsgV2();
        printf("GE Error: %s\n", err.GetString());
        ge::AscendString warn = ge::GEGetWarningMsgV2();
        printf("GE Warning: %s\n", warn.GetString());
        delete session;
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [GEIR]: RunGraph success\n", GetTime().c_str());

    int output_num = output.size();
    printf("\n========== GEIR NPU Output ==========\n");
    for (int i = 0; i < output_num; i++) {
        string output_file = "./clip_by_norm_no_div_sum_output_" + std::to_string(i) + ".bin";
        uint8_t* data = output[i].GetData();
        int64_t shape_size = output[i].GetTensorDesc().GetShape().GetShapeSize();
        uint32_t data_size = shape_size * GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        WriteDataToFile(output_file, data_size, data);

        float* floatData = static_cast<float*>(data);
        printf("Output tensor[%d] (%ld elements):\n", i, shape_size);
        for (int64_t j = 0; j < shape_size; j++) {
            printf("  [%ld] NPU=%.6f\n", j, floatData[j]);
        }
    }
    printf("======================================\n");

    // CPU golden verification
    // x            = {-1.0, 0.0, 0.25, 0.5, 2.0, 4.0, -0.5, 9.0}
    // greater_zeros = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    // select_ones   = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
    // maximum_ones  = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}
    float xVals[] = {-1.0f, 0.0f, 0.25f, 0.5f, 2.0f, 4.0f, -0.5f, 9.0f};
    float gtVals[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float selVals[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float maxVals[] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    int64_t elemCount = 8;

    printf("\n========== Golden Reference ==========\n");
    int allMatch = 1;
    float* npuData = static_cast<float*>(output[0].GetData());
    for (int64_t i = 0; i < elemCount; i++) {
        float temp1 = (xVals[i] > gtVals[i]) ? xVals[i] : selVals[i];
        float temp2 = std::sqrt(temp1);
        float temp3 = (xVals[i] <= gtVals[i]) ? xVals[i] : temp2;
        float golden = std::max(temp3, maxVals[i]);
        float npuVal = npuData[i];
        int match = (std::fabs(npuVal - golden) < 1e-4f);
        if (!match)
            allMatch = 0;
        printf(
            "  [%ld] x=%.2f gt=%.2f sel=%.2f max=%.2f -> NPU=%.6f golden=%.6f %s\n", i, xVals[i], gtVals[i], selVals[i],
            maxVals[i], npuVal, golden, match ? "OK" : "MISMATCH");
    }
    printf("======================================\n");
    printf("Result: %s\n", allMatch ? "ALL PASS" : "FAIL");

    ge::GEFinalize();
    return allMatch ? SUCCESS : FAILED;
}
