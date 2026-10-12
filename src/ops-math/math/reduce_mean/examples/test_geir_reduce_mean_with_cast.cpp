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
#include <cstring>
#include <stdint.h>
#include <vector>
#include <string>
#include <map>
#include <new>

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "array_ops.h"
#include "ge_ir_build.h"

#include "../op_graph/reduce_mean_with_cast_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

#define ADD_INPUT(inputIndex, inputName, inputDtype, inputShape)                          \
    vector<int64_t> placeholder##inputIndex##_shape = inputShape;                           \
    auto placeholder##inputIndex = op::Data("placeholder" + inputIndex).set_attr_index(0); \
    TensorDesc placeholder##inputIndex##_desc =                                             \
        TensorDesc(ge::Shape(placeholder##inputIndex##_shape), FORMAT_ND, inputDtype);     \
    placeholder##inputIndex##_desc.SetPlacement(ge::kPlacementHost);                        \
    placeholder##inputIndex##_desc.SetFormat(FORMAT_ND);                                    \
    Tensor tensor_placeholder##inputIndex;                                                  \
    ret = GenOnesData(placeholder##inputIndex##_shape,                                      \
        tensor_placeholder##inputIndex,                                                     \
        placeholder##inputIndex##_desc,                                                     \
        inputDtype,                                                                         \
        2);                                                                                  \
    if (ret != SUCCESS) {                                                                    \
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());       \
        return FAILED;                                                                       \
    }                                                                                        \
    placeholder##inputIndex.update_input_desc_x(placeholder##inputIndex##_desc);           \
    input.push_back(tensor_placeholder##inputIndex);                                        \
    graph.AddOp(placeholder##inputIndex);                                                   \
    add1.set_input_##inputName(placeholder##inputIndex);                                   \
    inputs.push_back(placeholder##inputIndex)

#define ADD_CONST_INPUT(inputIndex, inputName, inputDtype, inputShape)                    \
    vector<int64_t> placeholder##inputIndex##_shape = inputShape;                           \
    auto placeholder##inputIndex = op::Const("placeholder" + inputIndex);                  \
    TensorDesc placeholder##inputIndex##_desc =                                             \
        TensorDesc(ge::Shape(placeholder##inputIndex##_shape), FORMAT_ND, inputDtype);     \
    placeholder##inputIndex##_desc.SetPlacement(ge::kPlacementHost);                        \
    placeholder##inputIndex##_desc.SetFormat(FORMAT_ND);                                    \
    Tensor tensor_placeholder##inputIndex;                                                  \
    ret = GenOnesData(placeholder##inputIndex##_shape,                                      \
        tensor_placeholder##inputIndex,                                                     \
        placeholder##inputIndex##_desc,                                                     \
        inputDtype,                                                                         \
        2);                                                                                  \
    if (ret != SUCCESS) {                                                                    \
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());       \
        return FAILED;                                                                       \
    }                                                                                        \
    placeholder##inputIndex.SetAttr("value", tensor_placeholder##inputIndex);              \
    placeholder##inputIndex.update_output_desc_y(placeholder##inputIndex##_desc);          \
    graph.AddOp(placeholder##inputIndex);                                                   \
    add1.set_input_##inputName(placeholder##inputIndex);                                   \
    add1.update_input_desc_##inputName(placeholder##inputIndex##_desc);                    \
    inputs.push_back(placeholder##inputIndex)

#define ADD_OUTPUT(outputIndex, outputName, outputDtype, outputShape)                        \
    TensorDesc outputName##outputIndex##_desc =                                              \
        TensorDesc(ge::Shape(outputShape), FORMAT_ND, outputDtype);                          \
    add1.update_output_desc_##outputName(outputName##outputIndex##_desc)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

#define ADD_INPUT_ATTR(attrName, attrValue)                                                  \
    add1.set_attr_##attrName(attrValue)

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
    if (dt == ge::DT_FLOAT)        { dilation = 4; }
    else if (dt == ge::DT_FLOAT16) { dilation = 2; }
    else if (dt == ge::DT_BF16)    { dilation = 2; }
    else if (dt == ge::DT_INT16)   { dilation = 2; }
    else if (dt == ge::DT_UINT16)  { dilation = 2; }
    else if (dt == ge::DT_INT32)   { dilation = 4; }
    else if (dt == ge::DT_UINT32)  { dilation = 4; }
    else if (dt == ge::DT_INT64)   { dilation = 8; }
    else if (dt == ge::DT_UINT64)  { dilation = 8; }
    else if (dt == ge::DT_INT8)    { dilation = 1; }
    return dilation;
}

int32_t GenOnesData(
    vector<int64_t> shapes, Tensor &input_tensor, TensorDesc &input_tensor_desc, DataType data_type, int value)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    uint32_t data_len = size * GetDataTypeSize(data_type);
    int32_t *pData = new (std::nothrow) int32_t[data_len];
    for (uint32_t i = 0; i < size; ++i) {
        *(pData + i) = value;
    }
    input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t *>(pData), data_len);
    return SUCCESS;
}

int32_t WriteDataToFile(string bin_file, uint64_t data_size, uint8_t *inputData)
{
    FILE *fp = fopen(bin_file.c_str(), "w");
    fwrite(inputData, sizeof(uint8_t), data_size, fp);
    fclose(fp);
    return SUCCESS;
}

// 构造 ReduceMeanWithCast 算子图
// 场景：x(FP16) -> ReduceMeanWithCast(dtype=DT_FLOAT) -> y(FP32)
// 融合规则会将其展开为 Cast(FP16->FP32) + ReduceMean
int CreateOppInGraph(DataType inDtype1, DataType inDtype2, DataType outDtype,
    std::vector<ge::Tensor> &input, std::vector<Operator> &inputs,
    std::vector<Operator> &outputs, Graph &graph)
{
    Status ret = SUCCESS;
    auto add1 = op::ReduceMeanWithCast("reduce_mean_with_cast");
    // x: [4, 3, 6], axes: [1] (reduce dim 0), y: [3, 6]
    std::vector<std::vector<int64_t>> shapes = {{4, 3, 6}, {1}, {3, 6}};

    ADD_INPUT(1, x, inDtype1, shapes[0]);
    ADD_CONST_INPUT(2, axes, inDtype2, shapes[1]);
    ADD_OUTPUT(3, y, outDtype, shapes[2]);
    ADD_INPUT_ATTR(keep_dims, false);
    ADD_INPUT_ATTR(noop_with_empty_axes, true);
    // dtype 属性触发融合规则中的 Cast 路径：将 x 从 inDtype1 cast 到 outDtype
    ADD_INPUT_ATTR(dtype, outDtype);

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

bool CreateAndConfigGraph(Graph &graph, std::vector<ge::Tensor> &input)
{
    printf("%s - INFO - [XIR]: Start to CreateAndConfigGraph\n", GetTime().c_str());
    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};

    // x: FP16 输入，dtype=DT_FLOAT 触发融合规则插入 Cast(FP16->FP32)，y: FP32 输出
    DataType inDtype1 = DT_FLOAT16;
    DataType inDtype2 = DT_INT32;
    DataType outDtype = DT_FLOAT;

    std::cout << "inDtype1(x): " << inDtype1 << std::endl;
    std::cout << "inDtype2(axes): " << inDtype2 << std::endl;
    std::cout << "outDtype(y/dtype): " << outDtype << std::endl;

    Status ret = CreateOppInGraph(inDtype1, inDtype2, outDtype, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create op in graph failed\n", GetTime().c_str());
        return false;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }
    return true;
}

bool AddGraphToSession(ge::Session *session, Graph &graph, uint32_t graph_id)
{
    printf("%s - INFO - [XIR]: Create ir session using build options success\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: Start to add compute graph to ir session\n", GetTime().c_str());

    std::map<AscendString, AscendString> graph_options = {};
    Status ret = session->AddGraph(graph_id, graph, graph_options);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Add graph failed\n", GetTime().c_str());
        delete session;
        ge::GEFinalize();
        return false;
    }
    printf("%s - INFO - [XIR]: Session add ir compute graph to ir session success\n", GetTime().c_str());
    return true;
}

bool DumpAndRunGraph(
    ge::Session *session, Graph &graph, std::vector<ge::Tensor> &input, std::vector<ge::Tensor> &output,
    uint32_t graph_id)
{
    printf("%s - INFO - [XIR]: dump graph to txt\n", GetTime().c_str());
    std::string file_path = "./dump";
    aclgrphDumpGraph(graph, file_path.c_str(), file_path.length());

    printf("%s - INFO - [XIR]: Start to run ir compute graph\n", GetTime().c_str());
    Status ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Run graph failed\n", GetTime().c_str());
        delete session;
        ge::GEFinalize();
        return false;
    }
    printf("%s - INFO - [XIR]: Session run ir compute graph success\n", GetTime().c_str());
    return true;
}

void ProcessInputData(std::vector<ge::Tensor> &input)
{
    int input_num = input.size();
    for (int i = 0; i < input_num; i++) {
        std::cout << "input " << i << " dtype :  " << input[i].GetTensorDesc().GetDataType() << std::endl;
        string input_file = "./tc_ge_irrun_reduce_mean_with_cast_input_" + std::to_string(i) + ".bin";
        uint8_t *input_data_i = input[i].GetData();
        int64_t input_shape = input[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th input, input shape size =" << input_shape << std::endl;
        uint32_t data_size = input_shape * GetDataTypeSize(input[i].GetTensorDesc().GetDataType());
        WriteDataToFile(input_file, data_size, input_data_i);
    }
}

void ProcessOutputData(std::vector<ge::Tensor> &output)
{
    int output_num = output.size();
    for (int i = 0; i < output_num; i++) {
        std::cout << "output " << i << " dtype :  " << output[i].GetTensorDesc().GetDataType() << std::endl;
        string output_file = "./tc_ge_irrun_reduce_mean_with_cast_output_" + std::to_string(i) + ".bin";
        uint8_t *output_data_i = output[i].GetData();
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th output, output shape size =" << output_shape << std::endl;
        uint32_t data_size = output_shape * GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        WriteDataToFile(output_file, data_size, output_data_i);
        for (int64_t j = 0; j < output_shape; j++) {
            LOG_PRINT("result[%ld] is: %u\n", j, output_data_i[j]);
        }
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

int main(int argc, char *argv[])
{
    if (!InitEnv()) {
        return FAILED;
    }

    const char *graph_name = "tc_ge_irrun_reduce_mean_with_cast";
    Graph graph(graph_name);
    std::vector<ge::Tensor> input;
    if (!CreateAndConfigGraph(graph, input)) {
        return FAILED;
    }

    std::map<AscendString, AscendString> build_options = {};
    printf("%s - INFO - [XIR]: Start to create ir session using build options\n", GetTime().c_str());
    ge::Session *session = new Session(build_options);

    uint32_t graph_id = 0;
    if (!AddGraphToSession(session, graph, graph_id)) {
        return FAILED;
    }

    std::vector<ge::Tensor> output;
    if (!DumpAndRunGraph(session, graph, input, output, graph_id)) {
        return FAILED;
    }

    ProcessInputData(input);
    ProcessOutputData(output);

    return FinalizeRes();
}