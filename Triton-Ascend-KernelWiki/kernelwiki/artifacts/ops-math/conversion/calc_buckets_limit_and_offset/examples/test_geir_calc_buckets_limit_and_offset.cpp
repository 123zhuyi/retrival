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
 * \file test_geir_calc_buckets_limit_and_offset.cpp
 * \brief GE IR test for CalcBucketsLimitAndOffset operator
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

#include "../op_graph/calc_buckets_limit_and_offset_proto.h"

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
        case ge::DT_INT32:
            return 4U;
        case ge::DT_INT64:
            return 8U;
        default:
            return 0U;
    }
}

int32_t GenIndexData(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, DataType data_type)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }

    uint32_t type_size = GetDataTypeSize(data_type);
    if (type_size == 0U) {
        printf("%s - ERROR - [XIR]: GenIndexData: unsupported data type\n", GetTime().c_str());
        return FAILED;
    }
    uint32_t data_len = size * type_size;
    uint8_t* pData = new (std::nothrow) uint8_t[data_len];
    if (pData == nullptr) {
        return FAILED;
    }

    if (data_type == ge::DT_INT32) {
        int32_t* data = reinterpret_cast<int32_t*>(pData);
        for (size_t i = 0; i < size; ++i) {
            data[i] = static_cast<int32_t>(i);
        }
    } else if (data_type == ge::DT_INT64) {
        int64_t* data = reinterpret_cast<int64_t*>(pData);
        for (size_t i = 0; i < size; ++i) {
            data[i] = static_cast<int64_t>(i);
        }
    }

    input_tensor = Tensor(input_tensor_desc, pData, data_len);
    delete[] pData;
    return SUCCESS;
}

int32_t GenOnesData(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, DataType data_type,
                    double value)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }

    uint32_t type_size = GetDataTypeSize(data_type);
    if (type_size == 0U) {
        printf("%s - ERROR - [XIR]: GenOnesData: unsupported data type\n", GetTime().c_str());
        return FAILED;
    }
    uint32_t data_len = size * type_size;
    uint8_t* pData = new (std::nothrow) uint8_t[data_len];
    if (pData == nullptr) {
        return FAILED;
    }

    if (data_type == ge::DT_INT32) {
        int32_t* data = reinterpret_cast<int32_t*>(pData);
        for (size_t i = 0; i < size; ++i) {
            data[i] = static_cast<int32_t>(value);
        }
    } else if (data_type == ge::DT_INT64) {
        int64_t* data = reinterpret_cast<int64_t*>(pData);
        for (size_t i = 0; i < size; ++i) {
            data[i] = static_cast<int64_t>(value);
        }
    }

    input_tensor = Tensor(input_tensor_desc, pData, data_len);
    delete[] pData;
    return SUCCESS;
}

int32_t WriteDataToFile(string bin_file, uint64_t data_size, uint8_t* inputData)
{
    FILE* fp = fopen(bin_file.c_str(), "wb");
    if (fp == nullptr) {
        return FAILED;
    }
    size_t write_size = fwrite(inputData, sizeof(uint8_t), data_size, fp);
    fclose(fp);
    return write_size == data_size ? SUCCESS : FAILED;
}

bool InitEnv()
{
    printf("%s - INFO - [XIR]: Start to initialize ge using ge global options\n", GetTime().c_str());
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Initialize ge using ge global options failed\n", GetTime().c_str());
        return false;
    }
    printf("%s - INFO - [XIR]: Initialize ge using ge global options success\n", GetTime().c_str());
    return true;
}

int CreateOppInGraph(DataType inDtype, std::vector<ge::Tensor>& input, std::vector<Operator>& inputs,
                     std::vector<Operator>& outputs, Graph& graph)
{
    Status ret = SUCCESS;
    const int32_t total_limit = 20;
    const int64_t bucket_count = 5;
    const int64_t ivf_size = 20;

    auto calc_op = op::CalcBucketsLimitAndOffset("calc_buckets_limit_and_offset1");

    std::string name0 = "placeholder0";
    auto placeholder0 = op::Data(name0.c_str()).set_attr_index(0);
    TensorDesc placeholder0_desc = TensorDesc(ge::Shape({bucket_count}), ge::FORMAT_ND, ge::DT_INT32);
    placeholder0_desc.SetPlacement(ge::kPlacementHost);
    placeholder0_desc.SetFormat(ge::FORMAT_ND);
    Tensor tensor_placeholder0;
    ret = GenIndexData({bucket_count}, tensor_placeholder0, placeholder0_desc, ge::DT_INT32);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate bucket_list data failed\n", GetTime().c_str());
        return FAILED;
    }
    placeholder0.update_input_desc_x(placeholder0_desc);
    graph.AddOp(placeholder0);
    input.push_back(tensor_placeholder0);
    calc_op.set_input_bucket_list(placeholder0);
    inputs.push_back(placeholder0);

    std::string name1 = "placeholder1";
    auto placeholder1 = op::Data(name1.c_str()).set_attr_index(1);
    TensorDesc placeholder1_desc = TensorDesc(ge::Shape({ivf_size}), ge::FORMAT_ND, ge::DT_INT32);
    placeholder1_desc.SetPlacement(ge::kPlacementHost);
    placeholder1_desc.SetFormat(ge::FORMAT_ND);
    Tensor tensor_placeholder1;
    ret = GenOnesData({ivf_size}, tensor_placeholder1, placeholder1_desc, ge::DT_INT32, 1.0);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate ivf_counts data failed\n", GetTime().c_str());
        return FAILED;
    }
    placeholder1.update_input_desc_x(placeholder1_desc);
    graph.AddOp(placeholder1);
    input.push_back(tensor_placeholder1);
    calc_op.set_input_ivf_counts(placeholder1);
    inputs.push_back(placeholder1);

    std::string name2 = "placeholder2";
    auto placeholder2 = op::Data(name2.c_str()).set_attr_index(2);
    TensorDesc placeholder2_desc = TensorDesc(ge::Shape({ivf_size}), ge::FORMAT_ND, inDtype);
    placeholder2_desc.SetPlacement(ge::kPlacementHost);
    placeholder2_desc.SetFormat(ge::FORMAT_ND);
    Tensor tensor_placeholder2;
    ret = GenOnesData({ivf_size}, tensor_placeholder2, placeholder2_desc, inDtype, 1.0);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate ivf_offset data failed\n", GetTime().c_str());
        return FAILED;
    }
    placeholder2.update_input_desc_x(placeholder2_desc);
    graph.AddOp(placeholder2);
    input.push_back(tensor_placeholder2);
    calc_op.set_input_ivf_offset(placeholder2);
    inputs.push_back(placeholder2);

    calc_op.set_attr_total_limit(total_limit);

    TensorDesc buckets_limit_desc = TensorDesc(ge::Shape({bucket_count}), ge::FORMAT_ND, ge::DT_INT32);
    TensorDesc buckets_offset_desc = TensorDesc(ge::Shape({bucket_count}), ge::FORMAT_ND, inDtype);
    calc_op.update_output_desc_buckets_limit(buckets_limit_desc);
    calc_op.update_output_desc_buckets_offset(buckets_offset_desc);

    outputs.push_back(calc_op);
    return SUCCESS;
}

void ProcessInputData(std::vector<ge::Tensor>& input)
{
    int input_num = input.size();
    for (int i = 0; i < input_num; i++) {
        string input_file = "./tc_ge_irrun_calc_buckets_input_" + std::to_string(i) + ".bin";
        uint8_t* input_data_i = input[i].GetData();
        int64_t input_shape = input[i].GetTensorDesc().GetShape().GetShapeSize();
        uint32_t type_size = GetDataTypeSize(input[i].GetTensorDesc().GetDataType());
        if (type_size == 0U) {
            printf("ERROR: input %d has unsupported dtype\n", i);
            continue;
        }
        uint32_t data_size = input_shape * type_size;
        WriteDataToFile(input_file, data_size, input_data_i);
    }
}

void ProcessOutputData(std::vector<ge::Tensor>& output)
{
    int output_num = output.size();
    for (int i = 0; i < output_num; i++) {
        string output_file = "./tc_ge_irrun_calc_buckets_output_" + std::to_string(i) + ".bin";
        uint8_t* output_data_i = output[i].GetData();
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        uint32_t type_size = GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        if (type_size == 0U) {
            printf("ERROR: output %d has unsupported dtype\n", i);
            continue;
        }
        uint32_t data_size = output_shape * type_size;
        WriteDataToFile(output_file, data_size, output_data_i);
    }
}

int main(int argc, char* argv[])
{
    if (!InitEnv()) {
        return FAILED;
    }

    DataType inDtype = DT_INT32;

    const char* graph_name = "tc_ge_irrun_test";
    Graph graph(graph_name);
    std::vector<ge::Tensor> input;
    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};

    Status ret = CreateOppInGraph(inDtype, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create ir graph failed\n", GetTime().c_str());
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    std::map<AscendString, AscendString> build_options = {};
    printf("%s - INFO - [XIR]: Start to create ir session using build options\n", GetTime().c_str());
    ge::Session* session = new Session(build_options);
    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create ir session failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Create ir session using build options success\n", GetTime().c_str());

    printf("%s - INFO - [XIR]: Start to add compute graph to ir session\n", GetTime().c_str());
    std::map<AscendString, AscendString> graph_options = {{"ge.exec.exclude_engines", "AiCore"}};
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Add graph failed\n", GetTime().c_str());
        delete session;
        ge::GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Session add ir compute graph to ir session success\n", GetTime().c_str());

    printf("%s - INFO - [XIR]: dump graph to txt\n", GetTime().c_str());
    std::string file_path = "./dump";
    aclgrphDumpGraph(graph, file_path.c_str(), file_path.length());

    printf("%s - INFO - [XIR]: Start to run ir compute graph\n", GetTime().c_str());
    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Run graph failed\n", GetTime().c_str());
        delete session;
        ge::GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Session run ir compute graph success\n", GetTime().c_str());

    ProcessInputData(input);
    ProcessOutputData(output);

    ge::AscendString error_msg = ge::GEGetErrorMsgV2();
    std::string error_str(error_msg.GetString());
    std::cout << "Error message: " << error_str << std::endl;
    ge::AscendString warning_msg = ge::GEGetWarningMsgV2();
    std::string warning_str(warning_msg.GetString());
    std::cout << "Warning message: " << warning_str << std::endl;

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
