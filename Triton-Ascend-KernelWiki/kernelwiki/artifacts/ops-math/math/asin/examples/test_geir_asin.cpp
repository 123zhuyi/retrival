/**
* This program is free software, you can redistribute it and/or modify it.
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
* BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
* the software repository for the full text of the License.
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

#include "nn_other.h"
#include "../op_graph/asin_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

#define ADD_INPUT(inputIndex, inputName, inputDtype, inputShape, inputValues)                                        \
    do {                                                                                                             \
        vector<int64_t> placeholder##inputIndex##_shape = inputShape;                                                \
        auto placeholder##inputIndex = op::Data("placeholder" #inputIndex).set_attr_index(0);                        \
        TensorDesc placeholder##inputIndex##_desc =                                                                  \
            TensorDesc(ge::Shape(placeholder##inputIndex##_shape), FORMAT_ND, inputDtype);                           \
        placeholder##inputIndex##_desc.SetPlacement(ge::kPlacementHost);                                             \
        placeholder##inputIndex##_desc.SetFormat(FORMAT_ND);                                                         \
        Tensor tensor_placeholder##inputIndex;                                                                       \
        ret = GenSampleData(placeholder##inputIndex##_shape, tensor_placeholder##inputIndex,                         \
            placeholder##inputIndex##_desc, inputDtype, inputValues);                                                \
        if (ret != SUCCESS) {                                                                                        \
            printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());                          \
            return FAILED;                                                                                           \
        }                                                                                                            \
        placeholder##inputIndex.update_input_desc_x(placeholder##inputIndex##_desc);                                 \
        placeholder##inputIndex.update_output_desc_y(placeholder##inputIndex##_desc);                                \
        input.push_back(tensor_placeholder##inputIndex);                                                             \
        graph.AddOp(placeholder##inputIndex);                                                                        \
        asin1.set_input_##inputName(placeholder##inputIndex);                                                        \
        inputs.push_back(placeholder##inputIndex);                                                                   \
    } while (0)

#define LOG_PRINT(message, ...)     \
    do {                            \
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
    int32_t dtype_size = ge::GetSizeByDataType(dt);
    return dtype_size > 0 ? static_cast<uint32_t>(dtype_size) : 1U;
}

void PrintOutputData(const uint8_t *output_data, int64_t output_shape, DataType data_type)
{
    if (data_type == DT_DOUBLE) {
        const double *result = reinterpret_cast<const double *>(output_data);
        for (int64_t j = 0; j < output_shape; j++) {
            LOG_PRINT("result[%ld] is: %f\n", j, result[j]);
        }
        return;
    }
    if (data_type == DT_FLOAT) {
        const float *result = reinterpret_cast<const float *>(output_data);
        for (int64_t j = 0; j < output_shape; j++) {
            LOG_PRINT("result[%ld] is: %f\n", j, result[j]);
        }
        return;
    }
    if (data_type == DT_FLOAT16) {
        const uint16_t *result = reinterpret_cast<const uint16_t *>(output_data);
        for (int64_t j = 0; j < output_shape; j++) {
            LOG_PRINT("result[%ld] fp16 bits is: 0x%04x\n", j, result[j]);
        }
    }
}

int32_t GenSampleData(vector<int64_t> shapes, Tensor &input_tensor, TensorDesc &input_tensor_desc,
    DataType data_type, const vector<double> &values)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= static_cast<size_t>(shapes[i]);
    }
    if (values.size() != size) {
        return FAILED;
    }
    uint32_t data_len = static_cast<uint32_t>(size * GetDataTypeSize(data_type));
    std::vector<uint8_t> data(data_len, 0);
    if (data_type == DT_FLOAT) {
        float *data_ptr = reinterpret_cast<float *>(data.data());
        for (size_t i = 0; i < size; ++i) {
            data_ptr[i] = static_cast<float>(values[i]);
        }
    } else if (data_type == DT_DOUBLE) {
        double *data_ptr = reinterpret_cast<double *>(data.data());
        for (size_t i = 0; i < size; ++i) {
            data_ptr[i] = values[i];
        }
    } else {
        return FAILED;
    }
    input_tensor = Tensor(input_tensor_desc, data);
    return SUCCESS;
}

int32_t WriteDataToFile(string bin_file, uint64_t data_size, uint8_t *inputData)
{
    FILE *fp = fopen(bin_file.c_str(), "wb");
    if (fp == nullptr) {
        return FAILED;
    }
    size_t write_size = fwrite(inputData, sizeof(uint8_t), data_size, fp);
    fclose(fp);
    return write_size == data_size ? SUCCESS : FAILED;
}

int CreateOppInGraph(DataType inDtype, std::vector<ge::Tensor> &input, std::vector<Operator> &inputs,
    std::vector<Operator> &outputs, Graph &graph)
{
    Status ret = SUCCESS;
    auto asin1 = op::Asin("asin1");
    std::vector<int64_t> xShape = {2, 3};
    std::vector<double> xData = {-0.75, -0.5, -0.25, 0.25, 0.5, 0.75};
    ADD_INPUT(1, x, inDtype, xShape, xData);

    outputs.push_back(asin1);
    return SUCCESS;
}

int main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;
    const char *graph_name = "tc_ge_irrun_test";
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

    DataType inDtype = DT_DOUBLE;

    ret = CreateOppInGraph(inDtype, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    std::map<AscendString, AscendString> build_options = {

    };
    printf("%s - INFO - [XIR]: Start to create ir session using build options\n", GetTime().c_str());
    ge::Session *session = new Session(build_options);

    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Create ir session using build options success\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: Start to add compute graph to ir session\n", GetTime().c_str());

    std::map<AscendString, AscendString> graph_options = {

    };
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);

    printf("%s - INFO - [XIR]: Session add ir compute graph to ir session success\n", GetTime().c_str());
    std::string file_path = "./dump";
    aclgrphDumpGraph(graph, file_path.c_str(), file_path.length());
    printf("%s - INFO - [XIR]: Start to run ir compute graph\n", GetTime().c_str());

    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Run graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Session run ir compute graph success\n", GetTime().c_str());

    int input_num = static_cast<int>(input.size());
    for (int i = 0; i < input_num; i++) {
        string input_file = "./tc_ge_irrun_test_asin_npu_input_" + std::to_string(i) + ".bin";
        uint8_t *input_data_i = input[i].GetData();
        int64_t input_shape = input[i].GetTensorDesc().GetShape().GetShapeSize();
        uint32_t data_size = static_cast<uint32_t>(input_shape) * GetDataTypeSize(input[i].GetTensorDesc().GetDataType());
        (void)WriteDataToFile((const char *)input_file.c_str(), data_size, input_data_i);
    }

    int output_num = static_cast<int>(output.size());
    for (int i = 0; i < output_num; i++) {
        string output_file = "./tc_ge_irrun_test_asin_npu_output_" + std::to_string(i) + ".bin";
        uint8_t *output_data_i = output[i].GetData();
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        uint32_t data_size = static_cast<uint32_t>(output_shape) * GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        (void)WriteDataToFile((const char *)output_file.c_str(), data_size, output_data_i);
        PrintOutputData(output_data_i, output_shape, output[i].GetTensorDesc().GetDataType());
    }

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
