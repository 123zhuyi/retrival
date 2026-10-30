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
 * \file test_geir_select_v2.cpp
 * \brief Test SelectV2 AICPU kernel via GE IR (Graph mode)
 *        Uses DT_DOUBLE to trigger AICPU kernel dispatch
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

#include "../op_graph/select_v2_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64];
    struct tm tm_info;
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime_r(&timep, &tm_info));
    return tmp;
}

uint32_t GetDataTypeSize(DataType dt)
{
    uint32_t dilation = 1;
    uint32_t oneByte = 1;
    uint32_t twoByte = 2;
    uint32_t fourByte = 4;
    uint32_t eightByte = 8;

    if (dt == ge::DT_FLOAT) {
        dilation = fourByte;
    } else if (dt == ge::DT_FLOAT16) {
        dilation = twoByte;
    } else if (dt == ge::DT_BF16) {
        dilation = twoByte;
    } else if (dt == ge::DT_INT16) {
        dilation = twoByte;
    } else if (dt == ge::DT_UINT16) {
        dilation = twoByte;
    } else if (dt == ge::DT_INT32) {
        dilation = fourByte;
    } else if (dt == ge::DT_UINT32) {
        dilation = fourByte;
    } else if (dt == ge::DT_INT64) {
        dilation = eightByte;
    } else if (dt == ge::DT_UINT64) {
        dilation = eightByte;
    } else if (dt == ge::DT_INT8) {
        dilation = oneByte;
    } else if (dt == ge::DT_DOUBLE) {
        dilation = eightByte;
    } else if (dt == ge::DT_COMPLEX64) {
        dilation = eightByte;
    } else if (dt == ge::DT_COMPLEX128) {
        dilation = eightByte * twoByte;
    } else if (dt == ge::DT_BOOL) {
        dilation = oneByte;
    }
    return dilation;
}

int32_t GenBoolData(vector<int64_t> shapes, Tensor &input_tensor, TensorDesc &input_tensor_desc, bool value)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    uint32_t data_len = size * sizeof(bool);
    bool *pData = new (std::nothrow) bool[size];
    if (pData == nullptr) {
        return FAILED;
    }
    for (size_t i = 0; i < size; ++i) {
        pData[i] = value;
    }
    input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t *>(pData), data_len);
    delete[] pData;
    return SUCCESS;
}

int32_t GenDoubleData(vector<int64_t> shapes, Tensor &input_tensor, TensorDesc &input_tensor_desc, double value)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    uint32_t data_len = size * sizeof(double);
    double *pData = new (std::nothrow) double[size];
    if (pData == nullptr) {
        return FAILED;
    }
    for (size_t i = 0; i < size; ++i) {
        pData[i] = value;
    }
    input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t *>(pData), data_len);
    delete[] pData;
    return SUCCESS;
}

int32_t WriteDataToFile(string bin_file, uint64_t data_size, uint8_t *inputData)
{
    FILE *fp = fopen(bin_file.c_str(), "wb");
    if (fp == nullptr) {
        printf("Failed to open file: %s\n", bin_file.c_str());
        return FAILED;
    }
    fwrite(inputData, sizeof(uint8_t), data_size, fp);
    fclose(fp);
    return SUCCESS;
}

int CreateOppInGraph(DataType inDtype, std::vector<ge::Tensor> &input, std::vector<Operator> &inputs,
    std::vector<Operator> &outputs, Graph &graph)
{
    Status ret = SUCCESS;

    auto selectV2Op = op::SelectV2("selectV2_1");

    std::vector<int64_t> conditionShape = {4, 2};
    std::vector<int64_t> thenShape = {4, 2};
    std::vector<int64_t> elseShape = {4, 2};

    auto placeholder0 = op::Data("placeholder0").set_attr_index(0);
    TensorDesc condition_desc(ge::Shape(conditionShape), FORMAT_ND, DT_BOOL);
    condition_desc.SetPlacement(ge::kPlacementHost);
    condition_desc.SetFormat(FORMAT_ND);
    Tensor tensor_condition;
    ret = GenBoolData(conditionShape, tensor_condition, condition_desc, true);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate condition data failed\n", GetTime().c_str());
        return FAILED;
    }
    placeholder0.update_input_desc_x(condition_desc);
    placeholder0.update_output_desc_y(condition_desc);
    input.push_back(tensor_condition);
    graph.AddOp(placeholder0);
    selectV2Op.set_input_condition(placeholder0);
    inputs.push_back(placeholder0);

    auto placeholder1 = op::Data("placeholder1").set_attr_index(1);
    TensorDesc then_desc(ge::Shape(thenShape), FORMAT_ND, inDtype);
    then_desc.SetPlacement(ge::kPlacementHost);
    then_desc.SetFormat(FORMAT_ND);
    Tensor tensor_then;
    ret = GenDoubleData(thenShape, tensor_then, then_desc, 1.0);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate then data failed\n", GetTime().c_str());
        return FAILED;
    }
    placeholder1.update_input_desc_x(then_desc);
    placeholder1.update_output_desc_y(then_desc);
    input.push_back(tensor_then);
    graph.AddOp(placeholder1);
    selectV2Op.set_input_then(placeholder1);
    inputs.push_back(placeholder1);

    auto placeholder2 = op::Data("placeholder2").set_attr_index(2);
    TensorDesc else_desc(ge::Shape(elseShape), FORMAT_ND, inDtype);
    else_desc.SetPlacement(ge::kPlacementHost);
    else_desc.SetFormat(FORMAT_ND);
    Tensor tensor_else;
    ret = GenDoubleData(elseShape, tensor_else, else_desc, 2.0);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate else data failed\n", GetTime().c_str());
        return FAILED;
    }
    placeholder2.update_input_desc_x(else_desc);
    placeholder2.update_output_desc_y(else_desc);
    input.push_back(tensor_else);
    graph.AddOp(placeholder2);
    selectV2Op.set_input_else(placeholder2);
    inputs.push_back(placeholder2);

    outputs.push_back(selectV2Op);
    return SUCCESS;
}

void SaveInputOutput(std::vector<ge::Tensor> &input, std::vector<ge::Tensor> &output) {
    int input_num = input.size();
    for (int i = 0; i < input_num; i++) {
        std::cout << "input " << i << " dtype :  " << input[i].GetTensorDesc().GetDataType() << std::endl;
        string input_file = "./tc_ge_irrun_select_v2_npu_input_" + std::to_string(i) + ".bin";
        uint8_t *input_data_i = input[i].GetData();
        int64_t input_shape = input[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th input, input shape size =" << input_shape << std::endl;
        uint32_t data_size = input_shape * GetDataTypeSize(input[i].GetTensorDesc().GetDataType());
        WriteDataToFile((const char *)input_file.c_str(), data_size, input_data_i);
    }

    int output_num = output.size();
    for (int i = 0; i < output_num; i++) {
        std::cout << "output " << i << " dtype :  " << output[i].GetTensorDesc().GetDataType() << std::endl;
        string output_file = "./tc_ge_irrun_select_v2_npu_output_" + std::to_string(i) + ".bin";
        uint8_t *output_data_i = output[i].GetData();
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th output, output shape size =" << output_shape << std::endl;
        uint32_t data_size = output_shape * GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        WriteDataToFile((const char *)output_file.c_str(), data_size, output_data_i);
        double *resultData = reinterpret_cast<double *>(output_data_i);
        for (int64_t j = 0; j < output_shape; j++) {
            LOG_PRINT("result[%ld] is: %f\n", j, resultData[j]);
        }
    }
}

int main(int argc, char *argv[])
{
    const char *graph_name = "tc_ge_irrun_select_v2";
    Graph graph(graph_name);
    std::vector<ge::Tensor> input;

    printf("%s - INFO - [XIR]: Start to initialize ge using ge global options\n", GetTime().c_str());
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Initialize ge using ge global options failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Initialize ge using ge global options success\n", GetTime().c_str());

    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};

    if (argc >= 2) {
        std::cout << argv[1] << std::endl;
    }

    DataType inDtype = DT_DOUBLE;

    std::cout << "Input dtype: " << inDtype << std::endl;
    std::cout << "Note: DT_DOUBLE triggers AICPU kernel (AICore does not support DOUBLE)" << std::endl;

    ret = CreateOppInGraph(inDtype, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create operator in graph failed\n", GetTime().c_str());
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    std::map<AscendString, AscendString> build_options = {};
    printf("%s - INFO - [XIR]: Start to create ir session using build options\n", GetTime().c_str());
    ge::Session *session = new (std::nothrow) Session(build_options);

    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Create ir session using build options success\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: Start to add compute graph to ir session\n", GetTime().c_str());

    std::map<AscendString, AscendString> graph_options = {
        {"ge.exec.precision_mode", "allow_mix_precision"}
    };

    ret = session->AddGraph(0, graph, graph_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Add compute graph to ir session failed\n", GetTime().c_str());
        delete session;
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Add compute graph to ir session success\n", GetTime().c_str());

    printf("%s - INFO - [XIR]: Start to run graph in ir session\n", GetTime().c_str());

    std::vector<ge::Tensor> outputs_tensor;
    ret = session->RunGraph(0, input, outputs_tensor);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Run graph in ir session failed\n", GetTime().c_str());
        delete session;
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Run graph in ir session success\n", GetTime().c_str());

    SaveInputOutput(input, outputs_tensor);

    delete session;

    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: GE Finalize failed\n", GetTime().c_str());
        return FAILED;
    }

    return SUCCESS;
}