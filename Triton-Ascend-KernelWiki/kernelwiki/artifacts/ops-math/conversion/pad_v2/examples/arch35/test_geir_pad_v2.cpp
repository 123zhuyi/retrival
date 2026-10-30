/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file test_geir_pad_v2.cpp
 * @brief PadV2算子的GE IR示例代码
 *
 * 本示例演示如何使用GE IR接口构建和执行PadV2算子：
 * 1. 创建输入tensor（2D tensor）
 * 2. 设置paddings参数（例如：[[1,1], [2,2]]）
 * 3. 设置constant_values（例如：0.0）
 * 4. 执行PadV2算子
 * 5. 验证输出结果
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
#include "../op_graph/pad_v2_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

// 宏：添加输入tensor
#define ADD_INPUT(inputIndex, inputName, inputDtype, inputShape)                                              \
    do {                                                                                                      \
        std::string name##inputIndex = "placeholder" + std::to_string(inputIndex);                            \
        auto placeholder##inputIndex = op::Data(name##inputIndex.c_str()).set_attr_index(0);                  \
        TensorDesc placeholder##inputIndex##_desc = TensorDesc(ge::Shape(inputShape), FORMAT_ND, inputDtype); \
        placeholder##inputIndex##_desc.SetPlacement(ge::kPlacementHost);                                      \
        placeholder##inputIndex##_desc.SetFormat(FORMAT_ND);                                                  \
        Tensor tensor_placeholder##inputIndex;                                                                \
        ret = GenDataFloat32(inputShape, tensor_placeholder##inputIndex, placeholder##inputIndex##_desc);     \
        if (ret != SUCCESS) {                                                                                 \
            printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());                    \
            return FAILED;                                                                                    \
        }                                                                                                     \
        placeholder##inputIndex.update_input_desc_x(placeholder##inputIndex##_desc);                          \
        graph.AddOp(placeholder##inputIndex);                                                                 \
        input.push_back(tensor_placeholder##inputIndex);                                                      \
        padv2.set_input_##inputName(placeholder##inputIndex);                                                 \
        inputs.push_back(placeholder##inputIndex);                                                            \
    } while (0)

// 宏：添加int32类型的输入tensor（用于paddings）
#define ADD_INT32_INPUT(inputIndex, inputName, inputShape, dataVec)                                              \
    do {                                                                                                         \
        std::string name##inputIndex = "placeholder" + std::to_string(inputIndex);                               \
        auto placeholder##inputIndex = op::Data(name##inputIndex.c_str()).set_attr_index(0);                     \
        TensorDesc placeholder##inputIndex##_desc = TensorDesc(ge::Shape(inputShape), FORMAT_ND, DT_INT32);      \
        placeholder##inputIndex##_desc.SetPlacement(ge::kPlacementHost);                                         \
        placeholder##inputIndex##_desc.SetFormat(FORMAT_ND);                                                     \
        Tensor tensor_placeholder##inputIndex;                                                                   \
        ret = GenInt32Data(inputShape, tensor_placeholder##inputIndex, placeholder##inputIndex##_desc, dataVec); \
        if (ret != SUCCESS) {                                                                                    \
            printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());                       \
            return FAILED;                                                                                       \
        }                                                                                                        \
        placeholder##inputIndex.update_input_desc_x(placeholder##inputIndex##_desc);                             \
        graph.AddOp(placeholder##inputIndex);                                                                    \
        input.push_back(tensor_placeholder##inputIndex);                                                         \
        padv2.set_input_##inputName(placeholder##inputIndex);                                                    \
        inputs.push_back(placeholder##inputIndex);                                                               \
    } while (0)

// 宏：添加scalar输入tensor（用于constant_values）
#define ADD_SCALAR_INPUT(inputIndex, inputName, inputDtype, value)                                              \
    do {                                                                                                        \
        std::string name##inputIndex = "placeholder" + std::to_string(inputIndex);                              \
        auto placeholder##inputIndex = op::Data(name##inputIndex.c_str()).set_attr_index(0);                    \
        TensorDesc placeholder##inputIndex##_desc = TensorDesc(ge::Shape({1}), FORMAT_ND, inputDtype);          \
        placeholder##inputIndex##_desc.SetPlacement(ge::kPlacementHost);                                        \
        placeholder##inputIndex##_desc.SetFormat(FORMAT_ND);                                                    \
        Tensor tensor_placeholder##inputIndex;                                                                  \
        ret = GenScalarData(inputDtype, tensor_placeholder##inputIndex, placeholder##inputIndex##_desc, value); \
        if (ret != SUCCESS) {                                                                                   \
            printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());                      \
            return FAILED;                                                                                      \
        }                                                                                                       \
        placeholder##inputIndex.update_input_desc_x(placeholder##inputIndex##_desc);                            \
        graph.AddOp(placeholder##inputIndex);                                                                   \
        input.push_back(tensor_placeholder##inputIndex);                                                        \
        padv2.set_input_##inputName(placeholder##inputIndex);                                                   \
        inputs.push_back(placeholder##inputIndex);                                                              \
    } while (0)

// 宏：添加输出tensor
#define ADD_OUTPUT(outputIndex, outputName, outputDtype, outputShape)                                           \
    do {                                                                                                        \
        TensorDesc outputName##outputIndex##_desc = TensorDesc(ge::Shape(outputShape), FORMAT_ND, outputDtype); \
        padv2.update_output_desc_##outputName(outputName##outputIndex##_desc);                                  \
    } while (0)

// 宏：添加属性
#define ADD_ATTR(attrName, attrValue) padv2.set_attr_##attrName(attrValue)

// 宏：打印日志
#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

// 获取当前时间字符串
string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime(&timep));
    return tmp;
}

// 获取数据类型的大小（字节数）
uint32_t GetDataTypeSize(DataType dt)
{
    if (dt == ge::DT_FLOAT)
        return 4;
    if (dt == ge::DT_FLOAT16)
        return 2;
    if (dt == ge::DT_BF16)
        return 2;
    if (dt == ge::DT_INT32)
        return 4;
    if (dt == ge::DT_INT64)
        return 8;
    return 4;
}

// 生成float32数据
int32_t GenDataFloat32(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    uint32_t data_len = size * 4;
    float* pData = new (std::nothrow) float[size];

    // 生成测试数据：0, 1, 2, 3, ...
    for (size_t i = 0; i < size; ++i) {
        pData[i] = static_cast<float>(i);
    }
    input_tensor = Tensor(input_tensor_desc, (uint8_t*)pData, data_len);
    return SUCCESS;
}

// 生成int32数据（用于paddings）
int32_t GenInt32Data(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc,
                     const std::vector<int32_t>& dataVec)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    uint32_t data_len = size * 4;
    int32_t* pData = new (std::nothrow) int32_t[size];

    // 使用传入的数据
    for (size_t i = 0; i < size && i < dataVec.size(); ++i) {
        pData[i] = dataVec[i];
    }
    input_tensor = Tensor(input_tensor_desc, (uint8_t*)pData, data_len);
    return SUCCESS;
}

// 生成scalar数据（用于constant_values）
int32_t GenScalarData(DataType dtype, Tensor& input_tensor, TensorDesc& input_tensor_desc, float value)
{
    input_tensor_desc.SetRealDimCnt(1);
    uint32_t data_len = GetDataTypeSize(dtype);

    if (dtype == ge::DT_FLOAT) {
        float* pData = new (std::nothrow) float[1];
        pData[0] = value;
        input_tensor = Tensor(input_tensor_desc, (uint8_t*)pData, data_len);
    } else {
        // 其他类型暂不支持
        return FAILED;
    }
    return SUCCESS;
}

// 写数据到文件
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
 * @brief 创建包含PadV2算子的计算图
 *
 * 示例场景：
 * - 输入x: shape=[3, 3]，数据为[0,1,2,3,4,5,6,7,8]
 * - paddings: [[1,1], [2,2]]，表示在第0维前后各填充1，在第1维前后各填充2
 * - constant_values: 0.0
 * - 输出y: shape=[5, 7]，填充后的结果
 */
int CreateOppInGraph(DataType inDtype, std::vector<ge::Tensor>& input, std::vector<Operator>& inputs,
                     std::vector<Operator>& outputs, Graph& graph)
{
    Status ret = SUCCESS;

    // 创建PadV2算子
    auto padv2 = op::PadV2("test_geir_pad_v2");

    // 定义输入shape
    std::vector<int64_t> xShape = {3, 3};           // 输入tensor的shape
    std::vector<int64_t> paddingsShape = {2, 2};    // paddings的shape: [N, 2]，N为x的rank
    std::vector<int64_t> constantValuesShape = {1}; // constant_values的shape（scalar）

    // 定义输出shape（根据paddings计算）
    // x.shape = [3, 3], paddings = [[1,1], [2,2]]
    // y.shape = [3+1+1, 3+2+2] = [5, 7]
    std::vector<int64_t> yShape = {5, 7};

    // 添加输入x
    ADD_INPUT(1, x, inDtype, xShape);

    // 添加输入paddings（int32类型）
    // paddings = [[1,1], [2,2]]，表示在第0维前后各填充1，在第1维前后各填充2
    std::vector<int32_t> paddingsData = {1, 1, 2, 2};
    ADD_INT32_INPUT(2, paddings, paddingsShape, paddingsData);

    // 添加输入constant_values（可选输入）
    ADD_SCALAR_INPUT(3, constant_values, inDtype, 0.0f);

    // 添加属性mode
    ADD_ATTR(mode, "constant");

    // 添加输出y
    ADD_OUTPUT(1, y, inDtype, yShape);

    outputs.push_back(padv2);

    return SUCCESS;
}

int main(int argc, char* argv[])
{
    const char* graph_name = "tc_ge_irrun_test";
    Graph graph(graph_name);
    std::vector<ge::Tensor> input;

    printf("%s - INFO - [XIR]: Start to initialize ge using ge global options\n", GetTime().c_str());

    // 初始化GE环境
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Initialize ge using ge global options failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Initialize ge using ge global options success\n", GetTime().c_str());

    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};

    // 设置输入数据类型
    DataType inDtype = DT_FLOAT;
    printf("Input dtype: %d\n", inDtype);

    // 创建计算图
    ret = CreateOppInGraph(inDtype, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create graph failed\n", GetTime().c_str());
        return FAILED;
    }

    // 设置图的输入输出
    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    // 创建会话
    std::map<AscendString, AscendString> build_options = {};
    printf("%s - INFO - [XIR]: Start to create ir session using build options\n", GetTime().c_str());
    ge::Session* session = new Session(build_options);

    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Create ir session using build options success\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: Start to add compute graph to ir session\n", GetTime().c_str());

    // 添加计算图到会话
    std::map<AscendString, AscendString> graph_options = {};
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);

    printf("%s - INFO - [XIR]: Session add ir compute graph to ir session success\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: dump graph to txt\n", GetTime().c_str());

    // dump图结构到文件
    std::string file_path = "./dump";
    aclgrphDumpGraph(graph, file_path.c_str(), file_path.length());

    // 执行图
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

    // 保存输入数据到文件
    int input_num = input.size();
    for (int i = 0; i < input_num; i++) {
        std::cout << "input " << i << " dtype :  " << input[i].GetTensorDesc().GetDataType() << std::endl;
        string input_file = "./tc_ge_irrun_test_0008_npu_input_" + std::to_string(i) + ".bin";
        uint8_t* input_data_i = input[i].GetData();
        int64_t input_shape = input[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th input, input shape size =" << input_shape << std::endl;
        uint32_t data_size = input_shape * GetDataTypeSize(input[i].GetTensorDesc().GetDataType());
        WriteDataToFile((const char*)input_file.c_str(), data_size, input_data_i);

        // 打印输入数据（仅float类型）
        if (input[i].GetTensorDesc().GetDataType() == ge::DT_FLOAT) {
            float* inputData = (float*)input_data_i;
            std::cout << "Input " << i << " data:" << std::endl;
            for (int64_t j = 0; j < std::min(input_shape, (int64_t)10); j++) {
                LOG_PRINT("  input[%ld] = %f\n", j, inputData[j]);
            }
        }
    }

    // 保存输出数据到文件
    int output_num = output.size();
    for (int i = 0; i < output_num; i++) {
        std::cout << "output " << i << " dtype :  " << output[i].GetTensorDesc().GetDataType() << std::endl;
        string output_file = "./tc_ge_irrun_test_0008_npu_output_" + std::to_string(i) + ".bin";
        uint8_t* output_data_i = output[i].GetData();
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th output, output shape size =" << output_shape << std::endl;
        uint32_t data_size = output_shape * GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        WriteDataToFile((const char*)output_file.c_str(), data_size, output_data_i);

        // 打印输出数据（仅float类型）
        if (output[i].GetTensorDesc().GetDataType() == ge::DT_FLOAT) {
            float* resultData = (float*)output_data_i;
            std::cout << "Output " << i << " data:" << std::endl;
            // 打印前10个和后10个元素
            for (int64_t j = 0; j < std::min(output_shape, (int64_t)10); j++) {
                LOG_PRINT("  result[%ld] = %f\n", j, resultData[j]);
            }
            if (output_shape > 20) {
                LOG_PRINT("  ... (showing first 10 elements)\n");
                for (int64_t j = output_shape - 10; j < output_shape; j++) {
                    LOG_PRINT("  result[%ld] = %f\n", j, resultData[j]);
                }
            }
        }
    }

    // 获取错误和警告信息
    ge::AscendString error_msg = ge::GEGetErrorMsgV2();
    std::string error_str(error_msg.GetString());
    std::cout << "Error message: " << error_str << std::endl;
    ge::AscendString warning_msg = ge::GEGetWarningMsgV2();
    std::string warning_str(warning_msg.GetString());
    std::cout << "Warning message: " << warning_str << std::endl;

    // 清理资源
    printf("%s - INFO - [XIR]: Start to finalize ir graph session\n", GetTime().c_str());
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Finalize ir graph session failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize ir graph session success\n", GetTime().c_str());

    return SUCCESS;
}
