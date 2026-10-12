/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "../op_graph/assign_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;
#define ADD_INPUT(inputIndex, inputName, inputDtype, inputShape)                                                       \
    vector<int64_t> placeholder##inputIndex##_shape = inputShape;                                                      \
    auto placeholder##inputIndex = op::Data("placeholder" + inputIndex).set_attr_index(0);                             \
    TensorDesc placeholder##inputIndex##_desc = TensorDesc(ge::Shape(placeholder##inputIndex##_shape), ge::FORMAT_ND,  \
                                                           inputDtype);                                                \
    placeholder##inputIndex##_desc.SetPlacement(ge::kPlacementHost);                                                   \
    placeholder##inputIndex##_desc.SetFormat(ge::FORMAT_ND);                                                           \
    Tensor tensor_placeholder##inputIndex;                                                                             \
    ret = GenOnesData(placeholder##inputIndex##_shape, tensor_placeholder##inputIndex, placeholder##inputIndex##_desc, \
                      inputDtype, 2);                                                                                  \
    if (ret != SUCCESS) {                                                                                              \
        LOG_PRINT("%s - ERROR - [ASSIGN_GE_IR]: Generate input data failed\n", GetTime().c_str());                     \
        return FAILED;                                                                                                 \
    }                                                                                                                  \
    placeholder##inputIndex.update_input_desc_x(placeholder##inputIndex##_desc);                                       \
    input.push_back(tensor_placeholder##inputIndex);                                                                   \
    graph.AddOp(placeholder##inputIndex);                                                                              \
    add1.set_input_##inputName(placeholder##inputIndex);                                                               \
    inputs.push_back(placeholder##inputIndex)

#define ADD_CONST_INPUT(inputIndex, inputName, inputDtype, inputShape)                                                 \
    vector<int64_t> placeholder##inputIndex##_shape = inputShape;                                                      \
    auto placeholder##inputIndex = op::Const("placeholder" + inputIndex);                                              \
    TensorDesc placeholder##inputIndex##_desc = TensorDesc(ge::Shape(placeholder##inputIndex##_shape), FORMAT_ND,      \
                                                           inputDtype);                                                \
    placeholder##inputIndex##_desc.SetPlacement(ge::kPlacementHost);                                                   \
    placeholder##inputIndex##_desc.SetFormat(FORMAT_ND);                                                               \
    Tensor tensor_placeholder##inputIndex;                                                                             \
    ret = GenOnesData(placeholder##inputIndex##_shape, tensor_placeholder##inputIndex, placeholder##inputIndex##_desc, \
                      inputDtype, 2);                                                                                  \
    if (ret != SUCCESS) {                                                                                              \
        LOG_PRINT("%s - ERROR - [ASSIGN_GE_IR]: Generate input data failed\n", GetTime().c_str());                     \
        return FAILED;                                                                                                 \
    }                                                                                                                  \
    placeholder##inputIndex.SetAttr("value", tensor_placeholder##inputIndex);                                          \
    placeholder##inputIndex.update_output_desc_y(placeholder##inputIndex##_desc);                                      \
    graph.AddOp(placeholder##inputIndex);                                                                              \
    add1.set_input_##inputName(placeholder##inputIndex);                                                               \
    add1.update_input_desc_##inputName(placeholder##inputIndex##_desc);                                                \
    inputs.push_back(placeholder##inputIndex)

#define ADD_OUTPUT(outputIndex, outputName, outputDtype, outputShape)                                           \
    TensorDesc outputName##outputIndex##_desc = TensorDesc(ge::Shape(outputShape), ge::FORMAT_ND, outputDtype); \
    add1.update_output_desc_##outputName(outputName##outputIndex##_desc)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

#define ADD_INPUT_ATTR(attrName, attrValue) add1.set_attr_##attrName(attrValue)

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
    }
    return dilation;
}

int32_t GenOnesDataFloat32(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, float value)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    uint32_t byteSizeFloat32 = 4;
    uint32_t data_len = size * byteSizeFloat32;
    float* pData = new (std::nothrow) float[size];

    for (size_t i = 0; i < size; ++i) {
        *(pData + i) = value;
    }
    input_tensor = Tensor(input_tensor_desc, (uint8_t*)pData, data_len);
    return SUCCESS;
}

int32_t GenOnesData(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, DataType data_type,
                    int value)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    uint32_t data_len = size * GetDataTypeSize(data_type);
    int32_t* pData = new (std::nothrow) int32_t[data_len];
    for (uint32_t i = 0; i < size; ++i) {
        *(pData + i) = value;
    }
    input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t*>(pData), data_len);
    return SUCCESS;
}

int32_t WriteDataToFile(string bin_file, uint64_t data_size, uint8_t* inputData)
{
    FILE* fp = fopen(bin_file.c_str(), "w");
    fwrite(inputData, sizeof(uint8_t), data_size, fp);
    fclose(fp);
    return SUCCESS;
}

int CreateOppInGraph(DataType inDtype, std::vector<ge::Tensor>& input, std::vector<Operator>& inputs,
                     std::vector<Operator>& outputs, Graph& graph)
{
    Status ret = SUCCESS;
    // 自定义代码：添加单算子定义到图中
    auto add1 = op::Assign("assign");
    vector<vector<int64_t>> shape = {{3, 4}, {3, 4}};

    ADD_INPUT(1, ref, inDtype, shape[0]);
    ADD_INPUT(2, value, inDtype, shape[1]);
    ADD_OUTPUT(1, ref, inDtype, shape[0]);

    outputs.push_back(add1);
    // 添加完毕
    return SUCCESS;
}

bool InitEnv()
{
    LOG_PRINT("%s - INFO - [ASSIGN_GE_IR]: Start to initialize ge using ge global options\n", GetTime().c_str());
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        LOG_PRINT("%s - ERROR - [ASSIGN_GE_IR]: Initialize ge using ge global options failed\n", GetTime().c_str());
        return false;
    }
    LOG_PRINT("%s - INFO - [ASSIGN_GE_IR]: Initialize ge using ge global options success\n", GetTime().c_str());
    return true;
}

bool CreateAndConfigGraph(Graph& graph, std::vector<ge::Tensor>& input)
{
    LOG_PRINT("%s - INFO - [ASSIGN_GE_IR]: Start to CreateAndConfigGraph\n", GetTime().c_str());
    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};

    DataType inDtype = DT_FLOAT;
    LOG_PRINT("inDtype: %d\n", static_cast<int>(inDtype));

    Status ret = CreateOppInGraph(inDtype, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        LOG_PRINT("%s - ERROR - [ASSIGN_GE_IR]: Create ir session using build options failed\n", GetTime().c_str());
        return false;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }
    return true;
}

bool AddGraphToSession(ge::Session* session, Graph& graph, uint32_t graph_id)
{
    LOG_PRINT("%s - INFO - [ASSIGN_GE_IR]: Create ir session using build options success\n", GetTime().c_str());

    LOG_PRINT("%s - INFO - [ASSIGN_GE_IR]: Start to add compute graph to ir session\n", GetTime().c_str());

    std::map<AscendString, AscendString> graph_options = {

    };

    Status ret = session->AddGraph(graph_id, graph, graph_options);
    if (ret != SUCCESS) {
        LOG_PRINT("%s - ERROR - [ASSIGN_GE_IR]: Add graph failed\n", GetTime().c_str());
        delete session;
        ge::GEFinalize();
        return false;
    }
    LOG_PRINT("%s - INFO - [ASSIGN_GE_IR]: Session add ir compute graph to ir session success\n", GetTime().c_str());

    return true;
}

bool DumpAndRunGraph(ge::Session* session, Graph& graph, std::vector<ge::Tensor>& input,
                     std::vector<ge::Tensor>& output, uint32_t graph_id)
{
    LOG_PRINT("%s - INFO - [ASSIGN_GE_IR]: dump graph to txt\n", GetTime().c_str());
    std::string file_path = "./dump";
    aclgrphDumpGraph(graph, file_path.c_str(), file_path.length());

    LOG_PRINT("%s - INFO - [ASSIGN_GE_IR]: Start to run ir compute graph\n", GetTime().c_str());

    Status ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        LOG_PRINT("%s - ERROR - [ASSIGN_GE_IR]: Run graph failed\n", GetTime().c_str());
        delete session;
        ge::GEFinalize();
        return false;
    }
    LOG_PRINT("%s - INFO - [ASSIGN_GE_IR]: Session run ir compute graph success\n", GetTime().c_str());
    return true;
}

void ProcessInputData(std::vector<ge::Tensor>& input)
{
    int input_num = input.size();
    for (int i = 0; i < input_num; i++) {
        LOG_PRINT("input %d dtype: %d\n", i, static_cast<int>(input[i].GetTensorDesc().GetDataType()));
        string input_file = "./tc_ge_irrun_test_0008_npu_input_" + std::to_string(i) + ".bin";
        uint8_t* input_data_i = input[i].GetData();
        int64_t input_shape = input[i].GetTensorDesc().GetShape().GetShapeSize();
        LOG_PRINT("input %d shape size: %ld\n", i, input_shape);
        uint32_t data_size = input_shape * GetDataTypeSize(input[i].GetTensorDesc().GetDataType());
        WriteDataToFile((const char*)input_file.c_str(), data_size, input_data_i);
    }
}

void ProcessOutputData(std::vector<ge::Tensor>& output)
{
    int output_num = output.size();
    for (int i = 0; i < output_num; i++) {
        LOG_PRINT("output %d dtype: %d\n", i, static_cast<int>(output[i].GetTensorDesc().GetDataType()));
        string output_file = "./tc_ge_irrun_test_0008_npu_output_" + std::to_string(i) + ".bin";
        uint8_t* output_data_i = output[i].GetData();
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        LOG_PRINT("output %d shape size: %ld\n", i, output_shape);
        uint32_t data_size = output_shape * GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        WriteDataToFile((const char*)output_file.c_str(), data_size, output_data_i);
        for (int64_t j = 0; j < output_shape; j++) {
            LOG_PRINT("result[%ld] is: %u\n", j, output_data_i[j]);
        }
    }
}

int FinalizeRes()
{
    ge::AscendString error_msg = ge::GEGetErrorMsgV2();
    std::string error_str(error_msg.GetString());
    LOG_PRINT("Error message: %s\n", error_str.c_str());
    ge::AscendString warning_msg = ge::GEGetWarningMsgV2();
    std::string warning_str(warning_msg.GetString());
    LOG_PRINT("Warning message: %s\n", warning_str.c_str());
    LOG_PRINT("%s - INFO - [ASSIGN_GE_IR]: Precision is ok\n", GetTime().c_str());
    LOG_PRINT("%s - INFO - [ASSIGN_GE_IR]: Start to finalize ir graph session\n", GetTime().c_str());
    Status ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        LOG_PRINT("%s - ERROR - [ASSIGN_GE_IR]: Finalize ir graph session failed\n", GetTime().c_str());
        return FAILED;
    }
    LOG_PRINT("%s - INFO - [ASSIGN_GE_IR]: Finalize ir graph session success\n", GetTime().c_str());
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    // 初始化环境
    if (!InitEnv()) {
        return FAILED;
    }

    // 创建计算图
    const char* graph_name = "tc_ge_irrun_test";
    Graph graph(graph_name);
    std::vector<ge::Tensor> input;
    if (!CreateAndConfigGraph(graph, input)) {
        return FAILED;
    }

    // 创建会话并添加图
    std::map<AscendString, AscendString> build_options = {

    };
    LOG_PRINT("%s - INFO - [ASSIGN_GE_IR]: Start to create ir session using build options\n", GetTime().c_str());
    ge::Session* session = new Session(build_options);

    uint32_t graph_id = 0;
    if (!AddGraphToSession(session, graph, graph_id)) {
        return FAILED;
    }

    // 执行图
    std::vector<ge::Tensor> output;
    if (!DumpAndRunGraph(session, graph, input, output, graph_id)) {
        return FAILED;
    }
    // 处理输入输出数据
    ProcessInputData(input);
    ProcessOutputData(output);

    // 清理资源
    return FinalizeRes();
}
