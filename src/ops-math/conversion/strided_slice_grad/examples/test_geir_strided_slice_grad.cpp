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
#include "../op_graph/strided_slice_grad_proto.h"

#define FAILED -1
#define SUCCESS 0
#define DEFAULT_DATA_SIZE 4
#define GEN_ONES_DATA_FLOAT32_GENERATOR(ORIG_VAL, IDX) ORIG_VAL + (IDX % 3) * 0.4f

using namespace ge;
using std::map;
using std::string;
using std::vector;

#define ADD_INPUT(inputIndex, inputName, inputDtype, inputShape)                                                    \
    do {                                                                                                            \
        std::string name##inputIndex = "placeholder" + std::to_string(inputIndex);                                  \
        auto placeholder##inputIndex = op::Data(name##inputIndex.c_str()).set_attr_index(0);                        \
        TensorDesc placeholder##inputIndex##_desc = TensorDesc(ge::Shape(inputShape), FORMAT_ND, inputDtype);       \
        placeholder##inputIndex##_desc.SetPlacement(ge::kPlacementHost);                                            \
        placeholder##inputIndex##_desc.SetFormat(FORMAT_ND);                                                        \
        Tensor tensor_placeholder##inputIndex;                                                                      \
        ret = GenOnesDataFloat32(inputShape, tensor_placeholder##inputIndex, placeholder##inputIndex##_desc, 2.3f); \
        if (ret != SUCCESS) {                                                                                       \
            LOG_PRINT("%s - ERROR - [STRIDED_SLICE_GRAD_GE_IR]: Generate input data failed\n", GetTime().c_str());  \
            return FAILED;                                                                                          \
        }                                                                                                           \
        placeholder##inputIndex.update_input_desc_x(placeholder##inputIndex##_desc);                                \
        graph.AddOp(placeholder##inputIndex);                                                                       \
        input.push_back(tensor_placeholder##inputIndex);                                                            \
        stridedSliceGradOps.set_input_##inputName(placeholder##inputIndex);                                         \
        inputs.push_back(placeholder##inputIndex);                                                                  \
    } while (0)

#define ADD_CONST_INPUT(inputIndex, inputName, inputDtype, inputShape, value)                                         \
    do {                                                                                                              \
        vector<int64_t> placeholder##inputIndex##_shape = inputShape;                                                 \
        auto placeholder##inputIndex = op::Const("placeholder" + inputIndex);                                         \
        TensorDesc placeholder##inputIndex##_desc = TensorDesc(ge::Shape(placeholder##inputIndex##_shape), FORMAT_ND, \
                                                               inputDtype);                                           \
        placeholder##inputIndex##_desc.SetPlacement(ge::kPlacementHost);                                              \
        placeholder##inputIndex##_desc.SetFormat(FORMAT_ND);                                                          \
        Tensor tensor_placeholder##inputIndex;                                                                        \
        ret = GenOnesData(placeholder##inputIndex##_shape, tensor_placeholder##inputIndex,                            \
                          placeholder##inputIndex##_desc, inputDtype, value);                                         \
        if (ret != SUCCESS) {                                                                                         \
            LOG_PRINT("%s - ERROR - [STRIDED_SLICE_GRAD_GE_IR]: Generate const input data failed\n",                  \
                      GetTime().c_str());                                                                             \
            return FAILED;                                                                                            \
        }                                                                                                             \
        placeholder##inputIndex.SetAttr("value", tensor_placeholder##inputIndex);                                     \
        placeholder##inputIndex.update_output_desc_y(placeholder##inputIndex##_desc);                                 \
        graph.AddOp(placeholder##inputIndex);                                                                         \
        stridedSliceGradOps.set_input_##inputName(placeholder##inputIndex);                                           \
        stridedSliceGradOps.update_input_desc_##inputName(placeholder##inputIndex##_desc);                            \
        inputs.push_back(placeholder##inputIndex);                                                                    \
    } while (0)

#define ADD_OUTPUT(outputIndex, outputName, outputDtype, outputShape)                                           \
    do {                                                                                                        \
        TensorDesc outputName##outputIndex##_desc = TensorDesc(ge::Shape(outputShape), FORMAT_ND, outputDtype); \
        stridedSliceGradOps.update_output_desc_##outputName(outputName##outputIndex##_desc);                    \
    } while (0)

#define ADD_INPUT_ATTR(attrName, attrValue) stridedSliceGradOps.set_attr_##attrName(attrValue)

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
    if (dt == ge::DT_FLOAT)
        return 4;
    if (dt == ge::DT_FLOAT16)
        return 2;
    if (dt == ge::DT_BF16)
        return 2;
    return 4;
}

int32_t GenOnesDataFloat32(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, float value)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    uint32_t data_len = size * 4;
    float* pData = new (std::nothrow) float[size];
    for (size_t i = 0; i < size; ++i) {
        // make data meaningful
        pData[i] = GEN_ONES_DATA_FLOAT32_GENERATOR(value, i);
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
    // 自定义代码：添加单算子定义到图中
    auto stridedSliceGradOps = op::StridedSliceGrad("test_geir_strided_slice_grad");
    // 输入shape
    std::vector<int64_t> constShape = {1};
    std::vector<int64_t> dyShape = {79};
    // 输出shape
    std::vector<int64_t> outShape = {79};
    DataType constDtype = DT_INT32;

    ADD_CONST_INPUT(1, shape, constDtype, constShape, 79);
    ADD_CONST_INPUT(2, begin, constDtype, constShape, 0);
    ADD_CONST_INPUT(3, end, constDtype, constShape, 58);
    ADD_CONST_INPUT(4, strides, constDtype, constShape, 1);
    ADD_INPUT(5, dy, inDtype, dyShape);

    // 添加必选属性
    ADD_INPUT_ATTR(begin_mask, 0);
    ADD_INPUT_ATTR(end_mask, 1);
    ADD_INPUT_ATTR(ellipsis_mask, 1);
    ADD_INPUT_ATTR(new_axis_mask, 0);
    ADD_INPUT_ATTR(shrink_axis_mask, 0);
    // 添加输出
    ADD_OUTPUT(1, output, inDtype, outShape);

    outputs.push_back(stridedSliceGradOps);
    // 添加完毕
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    const char* graph_name = "tc_ge_irrun_test";
    Graph graph(graph_name);
    std::vector<ge::Tensor> input;

    LOG_PRINT("%s - INFO - [STRIDED_SLICE_GRAD_GE_IR]: Start to initialize ge using ge global options\n",
              GetTime().c_str());
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        LOG_PRINT("%s - ERROR - [STRIDED_SLICE_GRAD_GE_IR]: Initialize ge using ge global options failed\n",
                  GetTime().c_str());
        return FAILED;
    }
    LOG_PRINT("%s - INFO - [STRIDED_SLICE_GRAD_GE_IR]: Initialize ge using ge global options success\n",
              GetTime().c_str());

    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};

    LOG_PRINT("argv[1] = %s\n", argv[1]);
    DataType inDtype = DT_FLOAT;
    ret = CreateOppInGraph(inDtype, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        LOG_PRINT("%s - ERROR - [STRIDED_SLICE_GRAD_GE_IR]: Create ir session using build options failed\n",
                  GetTime().c_str());
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    std::map<AscendString, AscendString> build_options = {

    };
    LOG_PRINT("%s - INFO - [STRIDED_SLICE_GRAD_GE_IR]: Start to create ir session using build options\n",
              GetTime().c_str());
    ge::Session* session = new Session(build_options);

    if (session == nullptr) {
        LOG_PRINT("%s - ERROR - [STRIDED_SLICE_GRAD_GE_IR]: Create ir session using build options failed\n",
                  GetTime().c_str());
        return FAILED;
    }
    LOG_PRINT("%s - INFO - [STRIDED_SLICE_GRAD_GE_IR]: Create ir session using build options success\n",
              GetTime().c_str());
    LOG_PRINT("%s - INFO - [STRIDED_SLICE_GRAD_GE_IR]: Start to add compute graph to ir session\n", GetTime().c_str());

    std::map<AscendString, AscendString> graph_options = {

    };
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);

    LOG_PRINT("%s - INFO - [STRIDED_SLICE_GRAD_GE_IR]: Session add ir compute graph to ir session success\n",
              GetTime().c_str());
    LOG_PRINT("%s - INFO - [STRIDED_SLICE_GRAD_GE_IR]: dump graph to txt\n", GetTime().c_str());
    std::string file_path = "./dump";
    aclgrphDumpGraph(graph, file_path.c_str(), file_path.length());
    LOG_PRINT("%s - INFO - [STRIDED_SLICE_GRAD_GE_IR]: Start to run ir compute graph\n", GetTime().c_str());
    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        LOG_PRINT("%s - ERROR - [STRIDED_SLICE_GRAD_GE_IR]: Run graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }
    LOG_PRINT("%s - INFO - [STRIDED_SLICE_GRAD_GE_IR]: Session run ir compute graph success\n", GetTime().c_str());

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

    int output_num = output.size();
    for (int i = 0; i < output_num; i++) {
        LOG_PRINT("output %d dtype: %d\n", i, static_cast<int>(output[i].GetTensorDesc().GetDataType()));
        string output_file = "./tc_ge_irrun_test_0008_npu_output_" + std::to_string(i) + ".bin";
        uint8_t* output_data_i = output[i].GetData();
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        LOG_PRINT("output %d shape size: %ld\n", i, output_shape);
        uint32_t data_size = output_shape * GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        WriteDataToFile((const char*)output_file.c_str(), data_size, output_data_i);
        float* resultData = (float*)output_data_i;
        for (int64_t j = 0; j < output_shape; j++) {
            LOG_PRINT("result[%ld] is: %f\n", j, resultData[j]);
        }
    }

    ge::AscendString error_msg = ge::GEGetErrorMsgV2();
    std::string error_str(error_msg.GetString());
    LOG_PRINT("Error message: %s\n", error_str.c_str());
    ge::AscendString warning_msg = ge::GEGetWarningMsgV2();
    std::string warning_str(warning_msg.GetString());
    LOG_PRINT("Warning message: %s\n", warning_str.c_str());
    LOG_PRINT("%s - INFO - [STRIDED_SLICE_GRAD_GE_IR]: Start to finalize ir graph session\n", GetTime().c_str());
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        LOG_PRINT("%s - ERROR - [STRIDED_SLICE_GRAD_GE_IR]: Finalize ir graph session failed\n", GetTime().c_str());
        return FAILED;
    }
    LOG_PRINT("%s - INFO - [STRIDED_SLICE_GRAD_GE_IR]: Finalize ir graph session success\n", GetTime().c_str());
    return SUCCESS;
}
