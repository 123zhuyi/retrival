/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_geir_adjacent_difference.cpp
 * \brief Graph mode example for AdjacentDifference operator.
 *
 * This example demonstrates how to use the AdjacentDifference operator in graph mode.
 * The operator compares adjacent elements: y[0]=0, y[i]=x[i]==x[i-1]?0:1
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
#include "../op_graph/adjacent_difference_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

/**
 * @brief Macro to add input tensor to the graph
 */
#define ADD_INPUT(inputIndex, inputName, inputDtype, inputShape)                                             \
    do {                                                                                                      \
        std::string name##inputIndex = "placeholder" + std::to_string(inputIndex);                            \
        auto placeholder##inputIndex = op::Data(name##inputIndex.c_str()).set_attr_index(0);                  \
        TensorDesc placeholder##inputIndex##_desc = TensorDesc(ge::Shape(inputShape), FORMAT_ND, inputDtype); \
        placeholder##inputIndex##_desc.SetPlacement(ge::kPlacementHost);                                      \
        placeholder##inputIndex##_desc.SetFormat(FORMAT_ND);                                                  \
        Tensor tensor_placeholder##inputIndex;                                                                \
        ret = GenInputDataFloat32(inputShape, tensor_placeholder##inputIndex, placeholder##inputIndex##_desc); \
        if (ret != SUCCESS) {                                                                                 \
            printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());                    \
            return FAILED;                                                                                    \
        }                                                                                                     \
        placeholder##inputIndex.update_input_desc_x(placeholder##inputIndex##_desc);                          \
        graph.AddOp(placeholder##inputIndex);                                                                 \
        input.push_back(tensor_placeholder##inputIndex);                                                      \
        adjDiff1.set_input_##inputName(placeholder##inputIndex);                                              \
        inputs.push_back(placeholder##inputIndex);                                                            \
    } while (0)

/**
 * @brief Macro to set output tensor descriptor
 */
#define ADD_OUTPUT(outputIndex, outputName, outputDtype, outputShape)                                        \
    do {                                                                                                      \
        TensorDesc outputName##outputIndex##_desc = TensorDesc(ge::Shape(outputShape), FORMAT_ND, outputDtype); \
        adjDiff1.update_output_desc_##outputName(outputName##outputIndex##_desc);                             \
    } while (0)

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
    if (dt == ge::DT_FLOAT) {
        return 4;
    }
    if (dt == ge::DT_FLOAT16) {
        return 2;
    }
    if (dt == ge::DT_BF16) {
        return 2;
    }
    if (dt == ge::DT_INT32) {
        return 4;
    }
    if (dt == ge::DT_INT64) {
        return 8;
    }
    return 4;
}

/**
 * @brief Generate input data with meaningful values for AdjacentDifference testing
 *
 * The operator computes: y[0]=0, y[i]=x[i]==x[i-1]?0:1
 * Test data: [1, 1, 2, 3, 3, 4, 6, 6, 6, 8]
 * Expected:  [0, 0, 1, 1, 0, 1, 1, 0, 0, 1]
 */
int32_t GenInputDataFloat32(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    uint32_t data_len = size * 4;
    float* pData = new (std::nothrow) float[size];

    // Generate test data that will show the adjacent difference behavior
    // [1, 1, 2, 3, 3, 4, 6, 6, 6, 8] -> [0, 0, 1, 1, 0, 1, 1, 0, 0, 1]
    float testData[] = {1.0f, 1.0f, 2.0f, 3.0f, 3.0f, 4.0f, 6.0f, 6.0f, 6.0f, 8.0f};
    size_t testDataSize = sizeof(testData) / sizeof(testData[0]);

    for (size_t i = 0; i < size; ++i) {
        pData[i] = testData[i % testDataSize];
    }
    input_tensor = Tensor(input_tensor_desc, (uint8_t*)pData, data_len);
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

/**
 * @brief Create the AdjacentDifference operator in the graph
 */
int CreateOppInGraph(
    DataType inDtype, std::vector<ge::Tensor>& input, std::vector<Operator>& inputs, std::vector<Operator>& outputs,
    Graph& graph)
{
    Status ret = SUCCESS;

    // Create AdjacentDifference operator
    auto adjDiff1 = op::AdjacentDifference("adjDiff1");

    // Set output dtype attribute (DT_INT32 or DT_INT64)
    adjDiff1.set_attr_y_dtype(DT_INT32);

    // Input shape: {10} - test data has 10 elements
    std::vector<int64_t> xShape = {10};

    ADD_INPUT(1, x, inDtype, xShape);

    // Output shape is same as input, dtype is int32
    ADD_OUTPUT(1, y, DT_INT32, xShape);
    outputs.push_back(adjDiff1);

    return SUCCESS;
}

int main(int argc, char* argv[])
{
    const char* graph_name = "tc_ge_irrun_test";
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

    std::cout << argv[1] << std::endl;

    // Input dtype: float32
    DataType inDtype = DT_FLOAT;
    std::cout << "Input dtype: " << inDtype << std::endl;

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
    ge::Session* session = new Session(build_options);

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
    printf("%s - INFO - [XIR]: dump graph to txt\n", GetTime().c_str());
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

    // Print input data
    printf("\n=== Input Data ===\n");
    int input_num = input.size();
    for (int i = 0; i < input_num; i++) {
        std::cout << "input " << i << " dtype :  " << input[i].GetTensorDesc().GetDataType() << std::endl;
        string input_file = "./tc_ge_irrun_test_0008_npu_input_" + std::to_string(i) + ".bin";
        uint8_t* input_data_i = input[i].GetData();
        int64_t input_shape = input[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th input, input shape size = " << input_shape << std::endl;
        uint32_t data_size = input_shape * GetDataTypeSize(input[i].GetTensorDesc().GetDataType());
        WriteDataToFile((const char*)input_file.c_str(), data_size, input_data_i);
        float* inputData = reinterpret_cast<float*>(input_data_i);
        printf("Input values: [");
        for (int64_t j = 0; j < input_shape; j++) {
            printf("%.1f", inputData[j]);
            if (j < input_shape - 1) {
                printf(", ");
            }
        }
        printf("]\n");
        printf("Expected output: [0, 0, 1, 1, 0, 1, 1, 0, 0, 1]\n");
    }

    // Print output data
    printf("\n=== Output Data ===\n");
    int output_num = output.size();
    for (int i = 0; i < output_num; i++) {
        std::cout << "output " << i << " dtype :  " << output[i].GetTensorDesc().GetDataType() << std::endl;
        string output_file = "./tc_ge_irrun_test_0008_npu_output_" + std::to_string(i) + ".bin";
        uint8_t* output_data_i = output[i].GetData();
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th output, output shape size = " << output_shape << std::endl;
        uint32_t data_size = output_shape * GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        WriteDataToFile((const char*)output_file.c_str(), data_size, output_data_i);
        int32_t* resultData = reinterpret_cast<int32_t*>(output_data_i);
        printf("Output values: [");
        for (int64_t j = 0; j < output_shape; j++) {
            printf("%d", resultData[j]);
            if (j < output_shape - 1) {
                printf(", ");
            }
        }
        printf("]\n");
    }

    ge::AscendString error_msg = ge::GEGetErrorMsgV2();
    std::string error_str(error_msg.GetString());
    std::cout << "Error message: " << error_str << std::endl;
    ge::AscendString warning_msg = ge::GEGetWarningMsgV2();
    std::string warning_str(warning_msg.GetString());
    std::cout << "Warning message: " << warning_str << std::endl;
    printf("%s - INFO - [XIR]: Start to finalize ir graph session\n", GetTime().c_str());
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Finalize ir graph session failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize ir graph session success\n", GetTime().c_str());
    return SUCCESS;
}
