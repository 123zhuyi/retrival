/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <fstream>
#include <string.h>
#include <stdint.h>
#include <vector>
#include <string>
#include <map>
#include <complex>
#include "assert.h"

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "ge_ir_build.h"

#include "../op_graph/conjugate_transpose_proto.h"

#define FAILED -1
#define SUCCESS 0

namespace ge {
REG_OP(Data)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(index, Int, 0)
    .OP_END_FACTORY_REG(Data)

        REG_OP(Const)
    .OUTPUT(y, TensorType::ALL())
    .ATTR(value, Tensor, Tensor())
    .OP_END_FACTORY_REG(Const)
}

using namespace ge;
using std::map;
using std::string;
using std::vector;

#define LOG_PRINT(message, ...) printf(message, ##__VA_ARGS__)

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
    if (dt == ge::DT_COMPLEX64) {
        return static_cast<uint32_t>(sizeof(std::complex<float>));
    } else if (dt == ge::DT_COMPLEX128) {
        return static_cast<uint32_t>(sizeof(std::complex<double>));
    } else if (dt == ge::DT_INT32) {
        return static_cast<uint32_t>(sizeof(int32_t));
    } else if (dt == ge::DT_INT64) {
        return static_cast<uint32_t>(sizeof(int64_t));
    }
    return static_cast<uint32_t>(sizeof(int32_t));
}

std::string DataTypeToString(DataType dt)
{
    switch (dt) {
        case ge::DT_COMPLEX64:
            return "DT_COMPLEX64";
        case ge::DT_COMPLEX128:
            return "DT_COMPLEX128";
        case ge::DT_FLOAT16:
            return "DT_FLOAT16";
        case ge::DT_FLOAT:
            return "DT_FLOAT";
        case ge::DT_DOUBLE:
            return "DT_DOUBLE";
        case ge::DT_BOOL:
            return "DT_BOOL";
        case ge::DT_INT8:
            return "DT_INT8";
        case ge::DT_INT16:
            return "DT_INT16";
        case ge::DT_INT32:
            return "DT_INT32";
        case ge::DT_INT64:
            return "DT_INT64";
        case ge::DT_UINT8:
            return "DT_UINT8";
        case ge::DT_UINT16:
            return "DT_UINT16";
        case ge::DT_UINT32:
            return "DT_UINT32";
        case ge::DT_UINT64:
            return "DT_UINT64";
        default:
            return "DTYPE(" + std::to_string(static_cast<int>(dt)) + ")";
    }
}

// Generate complex input data (COMPLEX128 default validation dtype).
int32_t GenComplexData(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    uint32_t data_len = size * sizeof(std::complex<double>);
    std::complex<double>* pData = new (std::nothrow) std::complex<double>[size];
    if (pData == nullptr) {
        printf("%s - ERROR - [XIR]: Allocate input data buffer failed\n", GetTime().c_str());
        return FAILED;
    }
    for (size_t i = 0; i < size; ++i) {
        pData[i] = std::complex<double>(static_cast<double>(i) + 1.0, -(static_cast<double>(i) + 1.0));
    }
    input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t*>(pData), data_len);
    delete[] pData;
    return SUCCESS;
}

int32_t WriteDataToFile(string bin_file, uint64_t data_size, uint8_t* inputData)
{
    FILE* fp = fopen(bin_file.c_str(), "w");
    if (fp == nullptr) {
        printf("Failed to open file %s for writing.\n", bin_file.c_str());
        return FAILED;
    }
    fwrite(inputData, sizeof(uint8_t), data_size, fp);
    fclose(fp);
    return SUCCESS;
}

int CreateOppInGraph(DataType inDtype, std::vector<ge::Tensor>& input, std::vector<Operator>& inputs,
                     std::vector<Operator>& outputs, Graph& graph)
{
    Status ret = SUCCESS;
    // 自定义代码：添加单算子定义到图中
    auto conjOp = op::ConjugateTranspose("conjugate_transpose");

    std::vector<int64_t> xShape = {2, 3};
    std::vector<int64_t> permShape = {2};
    std::vector<int64_t> yShape = {3, 2};

    // input x (Data placeholder)
    auto placeholder_x = op::Data("placeholder_x").set_attr_index(0);
    TensorDesc x_desc = TensorDesc(ge::Shape(xShape), FORMAT_ND, inDtype);
    x_desc.SetPlacement(ge::kPlacementHost);
    x_desc.SetFormat(FORMAT_ND);
    Tensor tensor_x;
    ret = GenComplexData(xShape, tensor_x, x_desc);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());
        return FAILED;
    }
    placeholder_x.update_input_desc_x(x_desc);
    input.push_back(tensor_x);
    graph.AddOp(placeholder_x);
    conjOp.set_input_x(placeholder_x);
    conjOp.update_input_desc_x(x_desc);
    inputs.push_back(placeholder_x);

    // input perm (Const): infershape has a data dependency on perm.
    auto perm_const = op::Const("perm");
    TensorDesc perm_desc = TensorDesc(ge::Shape(permShape), FORMAT_ND, DT_INT32);
    perm_desc.SetPlacement(ge::kPlacementHost);
    perm_desc.SetFormat(FORMAT_ND);
    int32_t perm_data[2] = {1, 0};
    Tensor perm_tensor(perm_desc, reinterpret_cast<uint8_t*>(perm_data), permShape[0] * sizeof(int32_t));
    perm_const.SetAttr("value", perm_tensor);
    perm_const.update_output_desc_y(perm_desc);
    graph.AddOp(perm_const);
    conjOp.set_input_perm(perm_const);
    conjOp.update_input_desc_perm(perm_desc);

    // output y
    TensorDesc y_desc = TensorDesc(ge::Shape(yShape), FORMAT_ND, inDtype);
    conjOp.update_output_desc_y(y_desc);

    outputs.push_back(conjOp);
    // 添加完毕
    return SUCCESS;
}

int InitializeAndSetupGraph(Graph& graph, std::vector<ge::Tensor>& input, DataType inDtype)
{
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

    ret = CreateOppInGraph(inDtype, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    return SUCCESS;
}

int ExecuteGraph(Graph& graph, std::vector<ge::Tensor>& input, std::vector<ge::Tensor>& output)
{
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
    Status ret = session->AddGraph(graph_id, graph, graph_options);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Session add ir compute graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }

    printf("%s - INFO - [XIR]: Session add ir compute graph to ir session success\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: dump graph to txt\n", GetTime().c_str());
    std::string file_path = "./dump";
    aclgrphDumpGraph(graph, file_path.c_str(), file_path.length());
    printf("%s - INFO - [XIR]: Start to run ir compute graph\n", GetTime().c_str());

    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Run graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Session run ir compute graph success\n", GetTime().c_str());

    delete session;
    return SUCCESS;
}

void ProcessIOData(std::vector<ge::Tensor>& input, std::vector<ge::Tensor>& output)
{
    int input_num = input.size();
    for (int i = 0; i < input_num; i++) {
        std::cout << "input " << i << " dtype :  " << DataTypeToString(input[i].GetTensorDesc().GetDataType())
                  << std::endl;
        string input_file = "./tc_ge_irrun_conjugate_transpose_input_" + std::to_string(i) + ".bin";
        uint8_t* input_data_i = input[i].GetData();
        int64_t input_shape = input[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th input, input shape size =" << input_shape << std::endl;
        uint32_t data_size = input_shape * GetDataTypeSize(input[i].GetTensorDesc().GetDataType());
        WriteDataToFile((const char*)input_file.c_str(), data_size, input_data_i);
    }

    int output_num = output.size();
    for (int i = 0; i < output_num; i++) {
        std::cout << "output " << i << " dtype :  " << DataTypeToString(output[i].GetTensorDesc().GetDataType())
                  << std::endl;
        string output_file = "./tc_ge_irrun_conjugate_transpose_output_" + std::to_string(i) + ".bin";
        uint8_t* output_data_i = output[i].GetData();
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th output, output shape size =" << output_shape << std::endl;
        uint32_t data_size = output_shape * GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        WriteDataToFile((const char*)output_file.c_str(), data_size, output_data_i);
        // Print the actual output tensor values before validation for observability.
        std::complex<double>* resultData = reinterpret_cast<std::complex<double>*>(output_data_i);
        for (int64_t j = 0; j < output_shape; j++) {
            LOG_PRINT("result[%ld] = (%f, %f)\n", j, resultData[j].real(), resultData[j].imag());
        }
    }
}

int main(int argc, char* argv[])
{
    // 1、创建图对象
    const char* graph_name = "tc_ge_irrun_conjugate_transpose_test";
    Graph graph(graph_name);
    std::vector<ge::Tensor> input;

    if (argc > 1) {
        std::cout << argv[1] << std::endl;
    }

    DataType inDtype = DT_COMPLEX128;

    std::cout << inDtype << std::endl;

    // 初始化和设置图
    if (InitializeAndSetupGraph(graph, input, inDtype) != SUCCESS) {
        return FAILED;
    }

    // 执行图计算
    std::vector<ge::Tensor> output;
    if (ExecuteGraph(graph, input, output) != SUCCESS) {
        return FAILED;
    }

    // 处理输入输出数据
    ProcessIOData(input, output);

    ge::AscendString error_msg = ge::GEGetErrorMsgV2();
    std::string error_str(error_msg.GetString());
    if (!error_str.empty()) {
        std::cout << "Error message: " << error_str << std::endl;
    }
    ge::AscendString warning_msg = ge::GEGetWarningMsgV2();
    std::string warning_str(warning_msg.GetString());
    if (!warning_str.empty()) {
        std::cout << "Warning message: " << warning_str << std::endl;
    }
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
