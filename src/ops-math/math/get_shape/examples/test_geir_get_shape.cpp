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
 * \file test_geir_get_shape.cpp
 * \brief GEIR test for TBE-implemented GetShape operator.
 *
 * GetShape (TBE) reads each input as a 128-element int64 shape descriptor:
 *   descriptor[0..2] = reserved (0)
 *   descriptor[3]    = ndim (number of dimensions)
 *   descriptor[4..4+ndim-1] = dim values
 *   descriptor[rest] = 0
 *
 * InferShape (array_ops.cc) computes output size = sum of each input's dim count.
 * So the TensorDesc shape of each input must be the shape being queried
 * (e.g. {2,3,4} → 3 dims → contributes 3 to output), while the actual data
 * buffer must contain the descriptor format above.
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

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

#define DESCRIPTOR_SIZE 128

#define ADD_DYNAMIC_INPUT(inputIndex, inputDtype, inputShape, dataName)                    \
    do {                                                                                   \
        vector<int64_t> shape##dataName = inputShape;                                      \
        auto placeholder##dataName = op::Data("placeholder" #dataName).set_attr_index(0);  \
        TensorDesc desc##dataName(ge::Shape(shape##dataName), FORMAT_ND, inputDtype);      \
        desc##dataName.SetFormat(FORMAT_ND);                                               \
        placeholder##dataName.update_input_desc_x(desc##dataName);                         \
        placeholder##dataName.update_output_desc_y(desc##dataName);                        \
        Tensor tensor##dataName;                                                           \
        ret = GenShapeDescriptor(shape##dataName, tensor##dataName, desc##dataName);       \
        if (ret != SUCCESS) {                                                              \
            printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str()); \
            return FAILED;                                                                 \
        }                                                                                  \
        getShapeOp.UpdateDynamicInputDesc("x", inputIndex, desc##dataName);                \
        getShapeOp.set_dynamic_input_x(inputIndex, placeholder##dataName);                 \
        input.push_back(tensor##dataName);                                                 \
        graph.AddOp(placeholder##dataName);                                                \
        inputs.push_back(placeholder##dataName);                                           \
    } while (0)

#define ADD_OUTPUT(outputIndex, outputName, outputDtype, outputShape)                                           \
    do {                                                                                                        \
        TensorDesc outputName##outputIndex##_desc = TensorDesc(ge::Shape(outputShape), FORMAT_ND, outputDtype); \
        getShapeOp.update_output_desc_##outputName(outputName##outputIndex##_desc);                             \
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
    if (dt == ge::DT_FLOAT || dt == ge::DT_INT32 || dt == ge::DT_UINT32) {
        return 4;
    }
    if (dt == ge::DT_FLOAT16 || dt == ge::DT_BF16 || dt == ge::DT_INT16 || dt == ge::DT_UINT16) {
        return 2;
    }
    if (dt == ge::DT_INT64 || dt == ge::DT_UINT64 || dt == ge::DT_DOUBLE) {
        return 8;
    }
    return 1;
}

/*!
 * \brief Generate a 128-element int64 shape descriptor as input tensor data.
 *
 * The TensorDesc shape (e.g. {2,3,4}) determines:
 *   - GM allocation size (product(shape) * sizeof(int64) bytes)
 *   - InferShape output contribution (shape.size() dims)
 *
 * The actual data buffer is filled with the descriptor format:
 *   [0, 0, 0, ndim, dim0, dim1, ..., 0, 0, ...]
 *
 * The TBE kernel reads 128 int64 values (1024 bytes) via data_move from GM,
 * but only accesses index 3 (ndim) and index 4..4+ndim-1 (dim values).
 * As long as these indices fall within the allocated GM region, the kernel
 * produces correct results. For shape {2,3,4}: 24 int64 = 192 bytes,
 * index 3 and 4-6 are within range.
 */
int32_t GenShapeDescriptor(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    input_tensor_desc.SetShape(ge::Shape(shapes));

    size_t elemCount = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        elemCount *= static_cast<size_t>(shapes[i]);
    }

    size_t dataLen = elemCount * sizeof(int64_t);
    int64_t* pData = new (std::nothrow) int64_t[elemCount]();

    int64_t ndim = static_cast<int64_t>(shapes.size());
    if (elemCount >= 4) {
        pData[3] = ndim;
    }
    for (size_t i = 0; i < static_cast<size_t>(ndim) && (4 + i) < elemCount; ++i) {
        pData[4 + i] = shapes[i];
    }

    printf("GenShapeDescriptor: TensorDesc shape = {");
    for (size_t i = 0; i < shapes.size(); ++i) {
        printf("%ld%s", shapes[i], (i + 1 < shapes.size()) ? ", " : "");
    }
    printf("}, ndim = %ld, elemCount = %zu, dataLen = %zu\n", ndim, elemCount, dataLen);
    printf("  descriptor[3] = %ld (ndim)\n", pData[3]);
    for (size_t i = 0; i < static_cast<size_t>(ndim); ++i) {
        printf("  descriptor[%zu] = %ld (dim%zu)\n", 4 + i, pData[4 + i], i);
    }

    input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t*>(pData), dataLen);
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

int CreateOppInGraph(DataType inDtype, std::vector<ge::Tensor>& input, std::vector<Operator>& inputs,
                     std::vector<Operator>& outputs, Graph& graph)
{
    Status ret = SUCCESS;
    auto getShapeOp = op::GetShape("get_shape_1").create_dynamic_input_x(2, false);

    // Input 1: query shape {2, 3, 4} → contributes 3 to output
    // Data = descriptor [0,0,0,3,2,3,4,0,...] (24 int64 elements)
    std::vector<int64_t> x1Shape = {2, 3, 4};
    ADD_DYNAMIC_INPUT(0, ge::DT_INT64, x1Shape, X1);

    // Input 2: query shape {5, 6} → contributes 2 to output
    // Data = descriptor [0,0,0,2,5,6,0,0,0,0] (10 int64 elements)
    std::vector<int64_t> x2Shape = {5, 6};
    ADD_DYNAMIC_INPUT(1, ge::DT_INT64, x2Shape, X2);

    // Output: [2, 3, 4, 5, 6] → 5 elements, dtype int32
    vector<int64_t> yShape = {5};
    ADD_OUTPUT(1, y, ge::DT_INT32, yShape);

    outputs.push_back(getShapeOp);
    return SUCCESS;
}

void SaveInputOutput(std::vector<ge::Tensor>& input, std::vector<ge::Tensor>& output)
{
    int input_num = input.size();
    for (int i = 0; i < input_num; i++) {
        std::cout << "input " << i << " dtype :  " << input[i].GetTensorDesc().GetDataType() << std::endl;
        string input_file = "./tc_ge_irrun_test_0008_npu_input_" + std::to_string(i) + ".bin";
        uint8_t* input_data_i = input[i].GetData();
        int64_t input_shape = input[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th input, input shape size =" << input_shape << std::endl;
        uint32_t data_size = input_shape * GetDataTypeSize(input[i].GetTensorDesc().GetDataType());
        WriteDataToFile((const char*)input_file.c_str(), data_size, input_data_i);

        int64_t* descData = (int64_t*)input_data_i;
        printf("  input[%d] descriptor dump (first 8 int64): ", i);
        for (int j = 0; j < 8 && j < input_shape; ++j) {
            printf("[%d]=%ld ", j, descData[j]);
        }
        printf("\n");
    }

    int output_num = output.size();
    for (int i = 0; i < output_num; i++) {
        std::cout << "output " << i << " dtype :  " << output[i].GetTensorDesc().GetDataType() << std::endl;
        string output_file = "./tc_ge_irrun_test_0008_npu_output_" + std::to_string(i) + ".bin";
        uint8_t* output_data_i = output[i].GetData();
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th output, output shape size =" << output_shape << std::endl;
        uint32_t data_size = output_shape * GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        WriteDataToFile((const char*)output_file.c_str(), data_size, output_data_i);

        int32_t* resultData = (int32_t*)output_data_i;
        printf("  Expected output: [2, 3, 4, 5, 6]\n");
        printf("  Actual output:   [");
        for (int64_t j = 0; j < output_shape; ++j) {
            printf("%d%s", resultData[j], (j + 1 < output_shape) ? ", " : "");
        }
        printf("]\n");

        bool pass = true;
        int32_t expected[] = {2, 3, 4, 5, 6};
        for (int64_t j = 0; j < output_shape && j < 5; ++j) {
            if (resultData[j] != expected[j]) {
                pass = false;
                break;
            }
        }
        printf("  Result: %s\n", pass ? "PASS" : "FAIL");
    }
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
        printf("%s - ERROR - [XIR]: Initialize ge using ge global options failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Initialize ge using ge global options success\n", GetTime().c_str());

    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};

    if (argc > 1) {
        std::cout << argv[1] << std::endl;
    }

    DataType inDtype = DT_INT64;
    std::cout << "Input dtype: " << inDtype << std::endl;

    ret = CreateOppInGraph(inDtype, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create graph failed\n", GetTime().c_str());
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    std::map<AscendString, AscendString> build_options = {};
    printf("%s - INFO - [XIR]: Start to create ir session using build options\n", GetTime().c_str());
    ge::Session* session = new Session(build_options);

    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Create ir session using build options success\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: Start to add compute graph to ir session\n", GetTime().c_str());

    std::map<AscendString, AscendString> graph_options = {};
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
        printf("%s - ERROR - [XIR]: Run graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Session run ir compute graph success\n", GetTime().c_str());

    SaveInputOutput(input, output);

    ge::AscendString error_msg = ge::GEGetErrorMsgV2();
    std::string error_str(error_msg.GetString());
    std::cout << "Error message: " << error_str << std::endl;
    ge::AscendString warning_msg = ge::GEGetWarningMsgV2();
    std::string warning_str(warning_msg.GetString());
    std::cout << "Warning message: " << warning_str << std::endl;
    printf("%s - INFO - [XIR]: Start to finalize ir graph session\n", GetTime().c_str());
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Finalize ir graph session failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize ir graph session success\n", GetTime().c_str());
    return SUCCESS;
}
