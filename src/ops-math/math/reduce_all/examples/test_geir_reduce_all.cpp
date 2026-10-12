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
 * \file test_geir_reduce_all.cpp
 * \brief Test ReduceAll via GE IR graph mode
 */

#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <string.h>
#include <stdint.h>
#include <vector>

#include "assert.h"
#include "array_ops.h"
#include "ge_api.h"
#include "ge_api_types.h"
#include "ge_error_codes.h"
#include "ge_ir_build.h"
#include "graph.h"
#include "tensor.h"
#include "types.h"

#include "../op_graph/reduce_all_proto.h"

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
    if (dt == ge::DT_BOOL || dt == ge::DT_INT8 || dt == ge::DT_UINT8) {
        return 1;
    }
    if (dt == ge::DT_INT16 || dt == ge::DT_UINT16 || dt == ge::DT_FLOAT16 || dt == ge::DT_BF16) {
        return 2;
    }
    if (dt == ge::DT_INT32 || dt == ge::DT_UINT32 || dt == ge::DT_FLOAT) {
        return 4;
    }
    if (dt == ge::DT_INT64 || dt == ge::DT_UINT64 || dt == ge::DT_DOUBLE) {
        return 8;
    }
    return 0;
}

int32_t WriteDataToFile(const string &bin_file, uint64_t data_size, const uint8_t *input_data)
{
    FILE *fp = fopen(bin_file.c_str(), "wb");
    if (fp == nullptr) {
        printf("Failed to open file: %s\n", bin_file.c_str());
        return FAILED;
    }
    size_t written = fwrite(input_data, sizeof(uint8_t), data_size, fp);
    fclose(fp);
    if (written != data_size) {
        printf("Failed to write file: %s\n", bin_file.c_str());
        return FAILED;
    }
    return SUCCESS;
}

int32_t GenBoolData(const vector<int64_t> &shapes, Tensor &input_tensor, TensorDesc &input_tensor_desc,
    const vector<bool> &values)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (size_t i = 0; i < shapes.size(); ++i) {
        size *= static_cast<size_t>(shapes[i]);
    }
    if (values.size() != size) {
        return FAILED;
    }

    vector<uint8_t> raw_data(size);
    for (size_t i = 0; i < size; ++i) {
        raw_data[i] = static_cast<uint8_t>(values[i]);
    }
    input_tensor = Tensor(input_tensor_desc, raw_data.data(), raw_data.size());
    return SUCCESS;
}

int32_t GenInt32Data(const vector<int64_t> &shapes, Tensor &input_tensor, TensorDesc &input_tensor_desc,
    const vector<int32_t> &values)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (size_t i = 0; i < shapes.size(); ++i) {
        size *= static_cast<size_t>(shapes[i]);
    }
    if (values.size() != size) {
        return FAILED;
    }

    input_tensor = Tensor(input_tensor_desc, reinterpret_cast<const uint8_t *>(values.data()),
        values.size() * sizeof(int32_t));
    return SUCCESS;
}

int CreateOppInGraph(vector<ge::Tensor> &input, vector<Operator> &inputs, vector<Operator> &outputs, Graph &graph)
{
    Status ret = SUCCESS;
    auto reduce_all = op::ReduceAll("reduce_all_graph");
    vector<int64_t> x_shape = {2, 3};
    vector<int64_t> axes_shape = {1};
    vector<int64_t> y_shape = {2};

    auto x = op::Data("placeholder0").set_attr_index(0);
    TensorDesc x_desc(ge::Shape(x_shape), FORMAT_ND, DT_BOOL);
    x_desc.SetPlacement(ge::kPlacementHost);
    x_desc.SetFormat(FORMAT_ND);
    Tensor x_tensor;
    ret = GenBoolData(x_shape, x_tensor, x_desc, {true, true, false, true, true, true});
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());
        return FAILED;
    }
    x.update_input_desc_x(x_desc);
    x.update_output_desc_y(x_desc);
    input.push_back(x_tensor);
    graph.AddOp(x);
    reduce_all.set_input_x(x);
    inputs.push_back(x);

    auto axes = op::Data("placeholder1").set_attr_index(1);
    TensorDesc axes_desc(ge::Shape(axes_shape), FORMAT_ND, DT_INT32);
    axes_desc.SetPlacement(ge::kPlacementHost);
    axes_desc.SetFormat(FORMAT_ND);
    Tensor axes_tensor;
    ret = GenInt32Data(axes_shape, axes_tensor, axes_desc, {1});
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate axes data failed\n", GetTime().c_str());
        return FAILED;
    }
    axes.update_input_desc_x(axes_desc);
    axes.update_output_desc_y(axes_desc);
    input.push_back(axes_tensor);
    graph.AddOp(axes);
    reduce_all.set_input_axes(axes);
    reduce_all.update_input_desc_axes(axes_desc);
    inputs.push_back(axes);

    TensorDesc y_desc(ge::Shape(y_shape), FORMAT_ND, DT_BOOL);
    reduce_all.update_output_desc_y(y_desc);
    reduce_all.set_attr_keep_dims(false);
    outputs.push_back(reduce_all);
    return SUCCESS;
}

int SaveInputOutput(const vector<ge::Tensor> &input, const vector<ge::Tensor> &output)
{
    for (size_t i = 0; i < input.size(); ++i) {
        string input_file = "./tc_ge_irrun_reduce_all_npu_input_" + std::to_string(i) + ".bin";
        const uint8_t *input_data = input[i].GetData();
        int64_t input_shape = input[i].GetTensorDesc().GetShape().GetShapeSize();
        uint32_t data_size = input_shape * GetDataTypeSize(input[i].GetTensorDesc().GetDataType());
        if (WriteDataToFile(input_file, data_size, input_data) != SUCCESS) {
            return FAILED;
        }
    }

    const bool expected[] = {false, true};
    for (size_t i = 0; i < output.size(); ++i) {
        string output_file = "./tc_ge_irrun_reduce_all_npu_output_" + std::to_string(i) + ".bin";
        const uint8_t *output_data = output[i].GetData();
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        uint32_t data_size = output_shape * GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        if (WriteDataToFile(output_file, data_size, output_data) != SUCCESS) {
            return FAILED;
        }
        const bool *result_data = reinterpret_cast<const bool *>(output_data);
        for (int64_t j = 0; j < output_shape; ++j) {
            LOG_PRINT("result[%ld] is: %u\n", j, static_cast<unsigned int>(result_data[j]));
            if (result_data[j] != expected[j]) {
                printf("ReduceAll output mismatch at index %ld\n", j);
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

int main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;

    const char *graph_name = "tc_ge_irrun_reduce_all";
    Graph graph(graph_name);
    vector<ge::Tensor> input;

    printf("%s - INFO - [XIR]: Start to initialize ge using ge global options\n", GetTime().c_str());
    map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Initialize ge using ge global options failed\n", GetTime().c_str());
        return FAILED;
    }

    vector<Operator> inputs;
    vector<Operator> outputs;
    ret = CreateOppInGraph(input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        ge::GEFinalize();
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    map<AscendString, AscendString> build_options;
    ge::Session *session = new (std::nothrow) Session(build_options);
    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create ir session failed\n", GetTime().c_str());
        ge::GEFinalize();
        return FAILED;
    }

    map<AscendString, AscendString> graph_options = {{"ge.exec.precision_mode", "allow_mix_precision"}};
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Add graph failed\n", GetTime().c_str());
        delete session;
        ge::GEFinalize();
        return FAILED;
    }

    vector<ge::Tensor> output;
    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Run graph failed\n", GetTime().c_str());
        delete session;
        ge::GEFinalize();
        return FAILED;
    }

    ret = SaveInputOutput(input, output);
    delete session;
    if (ret != SUCCESS) {
        ge::GEFinalize();
        return FAILED;
    }

    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: GE Finalize failed\n", GetTime().c_str());
        return FAILED;
    }

    return SUCCESS;
}