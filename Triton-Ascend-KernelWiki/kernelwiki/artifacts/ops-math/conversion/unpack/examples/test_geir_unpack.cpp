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
 * \file test_geir_unpack.cpp
 * \brief GE IR test for Unpack operator
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

#include "nn_other.h"
#include "../op_graph/unpack_proto.h"

#define FAILED -1
#define SUCCESS 0
constexpr double DEFAULT_TEST_VALUE = 2.0;

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
        // FLOAT16/BF16 blocked by type_size==0 check above
        default:
            delete[] pData;
            return FAILED;
    }

    input_tensor = Tensor(input_tensor_desc, pData, data_len);
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
    const int64_t num = 2;
    const int64_t axis = 0;
    auto add1 = op::Unpack("unpack").create_dynamic_output_y(num);

    vector<int64_t> input_shape = {2, 4};
    auto data_op = op::Data("data").set_attr_index(0);
    TensorDesc data_desc = TensorDesc(ge::Shape(input_shape), ge::FORMAT_ND, inDtype);
    data_desc.SetPlacement(ge::kPlacementHost);
    data_desc.SetFormat(ge::FORMAT_ND);

    Tensor input_tensor;
    ret = GenOnesData(input_shape, input_tensor, data_desc, inDtype, DEFAULT_TEST_VALUE);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());
        return FAILED;
    }

    data_op.update_input_desc_x(data_desc);
    data_op.update_output_desc_y(data_desc);
    input.push_back(input_tensor);
    graph.AddOp(data_op);
    inputs.push_back(data_op);

    add1.set_input_x(data_op);
    add1.set_attr_num(num);
    add1.set_attr_axis(axis);

    outputs.push_back(add1);
    return SUCCESS;
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

void ProcessInputData(std::vector<ge::Tensor>& input)
{
    int input_num = input.size();
    for (int i = 0; i < input_num; i++) {
        std::cout << "input " << i << " dtype :  " << input[i].GetTensorDesc().GetDataType() << std::endl;
        string input_file = "./tc_ge_irrun_test_0008_npu_input_" + std::to_string(i) + ".bin";
        uint8_t* input_data_i = input[i].GetData();
        int64_t input_shape = input[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th input, input shape size =" << input_shape << std::endl;
        uint32_t type_size = GetDataTypeSize(input[i].GetTensorDesc().GetDataType());
        if (type_size == 0U) {
            printf("ERROR: input %d has unsupported dtype\n", i);
            continue;
        }
        uint32_t data_size = input_shape * type_size;
        WriteDataToFile((const char*)input_file.c_str(), data_size, input_data_i);
    }
}

void ProcessOutputData(std::vector<ge::Tensor>& output)
{
    int output_num = output.size();
    for (int i = 0; i < output_num; i++) {
        string output_file = "./tc_ge_irrun_test_0008_npu_output_" + std::to_string(i) + ".bin";
        uint8_t* output_data_i = output[i].GetData();
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        uint32_t type_size = GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        if (type_size == 0U) {
            printf("ERROR: output %d has unsupported dtype\n", i);
            continue;
        }
        uint32_t data_size = output_shape * type_size;
        WriteDataToFile((const char*)output_file.c_str(), data_size, output_data_i);
    }
}

int FinalizeRes()
{
    ge::AscendString error_msg = ge::GEGetErrorMsgV2();
    std::string error_str(error_msg.GetString());
    std::cout << "Error message: " << error_str << std::endl;
    ge::AscendString warning_msg = ge::GEGetWarningMsgV2();
    std::string warning_str(warning_msg.GetString());
    std::cout << "Warning message: " << warning_str << std::endl;
    printf("%s - INFO - [XIR]: Precision is ok\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: Start to finalize ir graph session\n", GetTime().c_str());
    Status ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Finalize ir graph session failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize ir graph session success\n", GetTime().c_str());
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    if (!InitEnv()) {
        return FAILED;
    }

    DataType inDtype = DT_DOUBLE;

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

    delete session;
    return FinalizeRes();
}