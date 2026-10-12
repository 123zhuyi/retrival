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
 * \file test_geir_slice_write.cpp
 * \brief GE IR test for SliceWrite operator
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

#include "../op_graph/slice_write_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

#define ADD_INPUT(inputIndex, inputName, inputDtype, inputShape)                                                    \
    do {                                                                                                            \
        std::string name##inputIndex = "placeholder" + std::to_string(inputIndex);                                  \
        auto placeholder##inputIndex = op::Data(name##inputIndex.c_str()).set_attr_index(0);                        \
        TensorDesc placeholder##inputIndex##_desc = TensorDesc(ge::Shape(inputShape), ge::FORMAT_ND, inputDtype);   \
        placeholder##inputIndex##_desc.SetPlacement(ge::kPlacementHost);                                            \
        placeholder##inputIndex##_desc.SetFormat(ge::FORMAT_ND);                                                    \
        Tensor tensor_placeholder##inputIndex;                                                                      \
        ret = GenOnesData(inputShape, tensor_placeholder##inputIndex, placeholder##inputIndex##_desc, inputDtype, 1.0); \
        if (ret != SUCCESS) {                                                                                       \
            printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());                          \
            return FAILED;                                                                                          \
        }                                                                                                           \
        placeholder##inputIndex.update_input_desc_x(placeholder##inputIndex##_desc);                                \
        graph.AddOp(placeholder##inputIndex);                                                                       \
        input.push_back(tensor_placeholder##inputIndex);                                                            \
        slice_write1.set_input_##inputName(placeholder##inputIndex);                                                \
        inputs.push_back(placeholder##inputIndex);                                                                  \
    } while (0)

#define ADD_OUTPUT(outputIndex, outputName, outputDtype, outputShape)                                               \
    do {                                                                                                            \
        TensorDesc outputName##outputIndex##_desc = TensorDesc(ge::Shape(outputShape), ge::FORMAT_ND, outputDtype); \
        slice_write1.update_output_desc_##outputName(outputName##outputIndex##_desc);                               \
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

int32_t GenOnesData(
    vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, DataType data_type, double value)
{
    uint32_t type_size = GetDataTypeSize(data_type);
    if (type_size == 0U) {
        printf("ERROR: data_type %d is not supported by GenOnesData (no standard C++ type mapping)\n",
               static_cast<int>(data_type));
        return FAILED;
    }

    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) { size *= shapes[i]; }

    uint32_t data_len = size * type_size;
    uint8_t* pData = new (std::nothrow) uint8_t[data_len];
    if (pData == nullptr) { return FAILED; }

    switch (data_type) {
        case ge::DT_BOOL: {
            bool* data = reinterpret_cast<bool*>(pData);
            for (size_t i = 0; i < size; ++i) data[i] = (value != 0);
            break;
        }
        case ge::DT_INT8: {
            int8_t* data = reinterpret_cast<int8_t*>(pData);
            for (size_t i = 0; i < size; ++i) data[i] = static_cast<int8_t>(value);
            break;
        }
        case ge::DT_UINT8: {
            uint8_t* data = pData;
            for (size_t i = 0; i < size; ++i) data[i] = static_cast<uint8_t>(value);
            break;
        }
        case ge::DT_INT16: {
            int16_t* data = reinterpret_cast<int16_t*>(pData);
            for (size_t i = 0; i < size; ++i) data[i] = static_cast<int16_t>(value);
            break;
        }
        case ge::DT_UINT16: {
            uint16_t* data = reinterpret_cast<uint16_t*>(pData);
            for (size_t i = 0; i < size; ++i) data[i] = static_cast<uint16_t>(value);
            break;
        }
        case ge::DT_INT32: {
            int32_t* data = reinterpret_cast<int32_t*>(pData);
            for (size_t i = 0; i < size; ++i) data[i] = static_cast<int32_t>(value);
            break;
        }
        case ge::DT_UINT32: {
            uint32_t* data = reinterpret_cast<uint32_t*>(pData);
            for (size_t i = 0; i < size; ++i) data[i] = static_cast<uint32_t>(value);
            break;
        }
        case ge::DT_INT64: {
            int64_t* data = reinterpret_cast<int64_t*>(pData);
            for (size_t i = 0; i < size; ++i) data[i] = static_cast<int64_t>(value);
            break;
        }
        case ge::DT_UINT64: {
            uint64_t* data = reinterpret_cast<uint64_t*>(pData);
            for (size_t i = 0; i < size; ++i) data[i] = static_cast<uint64_t>(value);
            break;
        }
        case ge::DT_FLOAT: {
            float* data = reinterpret_cast<float*>(pData);
            for (size_t i = 0; i < size; ++i) data[i] = static_cast<float>(value);
            break;
        }
        case ge::DT_DOUBLE: {
            double* data = reinterpret_cast<double*>(pData);
            for (size_t i = 0; i < size; ++i) data[i] = value;
            break;
        }
        // FLOAT16/BF16/COMPLEX blocked by type_size==0 check above
        default:
            delete[] pData;
            return FAILED;
    }

    input_tensor = Tensor(input_tensor_desc, pData, data_len);
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

int CreateOppInGraph(DataType inDtype, DataType beginDtype, std::vector<ge::Tensor>& input,
    std::vector<Operator>& inputs, std::vector<Operator>& outputs, Graph& graph)
{
    Status ret = SUCCESS;
    auto slice_write1 = op::SliceWrite("slice_write1");

    std::vector<int64_t> xShape = {2, 2};
    std::vector<int64_t> beginShape = {2};
    std::vector<int64_t> valueShape = {1, 2};

    ADD_INPUT(1, x, inDtype, xShape);

    // begin input: generate [1, 0] using GenOnesData then manually set element 1 to 0
    std::string name2 = "placeholder2";
    auto placeholder2 = op::Data(name2.c_str()).set_attr_index(1);
    TensorDesc placeholder2_desc = TensorDesc(ge::Shape(beginShape), ge::FORMAT_ND, beginDtype);
    placeholder2_desc.SetPlacement(ge::kPlacementHost);
    placeholder2_desc.SetFormat(ge::FORMAT_ND);
    Tensor tensor_placeholder2;
    ret = GenOnesData(beginShape, tensor_placeholder2, placeholder2_desc, beginDtype, 1.0);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate begin data failed\n", GetTime().c_str());
        return FAILED;
    }
    // Set begin data: [1, 0]
    uint8_t* begin_raw = tensor_placeholder2.GetData();
    switch (beginDtype) {
        case ge::DT_INT32: {
            int32_t* b = reinterpret_cast<int32_t*>(begin_raw);
            b[0] = 1;
            b[1] = 0;
            break;
        }
        case ge::DT_INT64: {
            int64_t* b = reinterpret_cast<int64_t*>(begin_raw);
            b[0] = 1;
            b[1] = 0;
            break;
        }
        default:
            printf("%s - ERROR - [XIR]: begin dtype %d not supported for special values\n",
                   GetTime().c_str(), static_cast<int>(beginDtype));
            return FAILED;
    }
    placeholder2.update_input_desc_x(placeholder2_desc);
    graph.AddOp(placeholder2);
    input.push_back(tensor_placeholder2);
    slice_write1.set_input_begin(placeholder2);
    inputs.push_back(placeholder2);

    ADD_INPUT(3, value, inDtype, valueShape);

    ADD_OUTPUT(1, x, inDtype, xShape);

    outputs.push_back(slice_write1);
    return SUCCESS;
}

void ProcessInputData(std::vector<ge::Tensor>& input)
{
    int input_num = input.size();
    for (int i = 0; i < input_num; i++) {
        string input_file = "./tc_ge_irrun_test_slice_write_npu_input_" + std::to_string(i) + ".bin";
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
        string output_file = "./tc_ge_irrun_test_slice_write_npu_output_" + std::to_string(i) + ".bin";
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

    DataType inDtype = DT_FLOAT;
    DataType beginDtype = DT_INT32;

    const char* graph_name = "tc_ge_irrun_test_slice_write";
    Graph graph(graph_name);
    std::vector<ge::Tensor> input;
    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};

    Status ret = CreateOppInGraph(inDtype, beginDtype, input, inputs, outputs, graph);
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
    std::map<AscendString, AscendString> graph_options = {};
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