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
 * @file test_geir_add_mat_mat_elements_plus.cpp
 * @brief GE IR 图模式调用示例 - AddMatMatElementsPlus
 *
 * 数学公式: c = c * beta + alpha * a * b
 *
 * 通过 GE Session 构建计算图，编译并执行 AddMatMatElementsPlus 算子。
 */

#include <iostream>
#include <fstream>
#include <string.h>
#include <stdint.h>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include "assert.h"

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "array_ops.h"
#include "ge_ir_build.h"

#include "../../op_graph/add_mat_mat_elements_plus_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

#define ADD_INPUT(inputIndex, inputName, inputDtype, inputShape)                                                  \
    vector<int64_t> placeholder##inputIndex##_shape = inputShape;                                                 \
    auto placeholder##inputIndex = op::Data("placeholder" #inputIndex).set_attr_index(inputIndex - 1);            \
    TensorDesc placeholder##inputIndex##_desc = TensorDesc(ge::Shape(placeholder##inputIndex##_shape), FORMAT_ND, \
                                                           inputDtype);                                           \
    placeholder##inputIndex##_desc.SetPlacement(ge::kPlacementHost);                                              \
    placeholder##inputIndex##_desc.SetFormat(FORMAT_ND);                                                          \
    Tensor tensor_placeholder##inputIndex;                                                                        \
    ret = GenInputData(placeholder##inputIndex##_shape, tensor_placeholder##inputIndex,                           \
                       placeholder##inputIndex##_desc, inputDtype, inputIndex);                                   \
    if (ret != SUCCESS) {                                                                                         \
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());                            \
        return FAILED;                                                                                            \
    }                                                                                                             \
    placeholder##inputIndex.update_input_desc_x(placeholder##inputIndex##_desc);                                  \
    input.push_back(tensor_placeholder##inputIndex);                                                              \
    graph.AddOp(placeholder##inputIndex);                                                                         \
    addMatOp.set_input_##inputName(placeholder##inputIndex);                                                      \
    inputs.push_back(placeholder##inputIndex);

#define ADD_OUTPUT(outputIndex, outputName, outputDtype, outputShape)                                       \
    TensorDesc outputName##outputIndex##_desc = TensorDesc(ge::Shape(outputShape), FORMAT_ND, outputDtype); \
    addMatOp.update_output_desc_##outputName(outputName##outputIndex##_desc);

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
    } else if (dt == ge::DT_FLOAT16 || dt == ge::DT_BF16) {
        return 2;
    }
    return 4;
}

static void FloatToBf16Bytes(float val, uint8_t* out)
{
    uint32_t bits;
    memcpy(&bits, &val, 4);
    uint32_t lsb = (bits >> 16) & 1, mag = bits & 0x7FFFFFFF;
    uint32_t rounded = mag + 0x7FFF + lsb;
    if (rounded < mag)
        rounded = 0x7F800000; // overflow -> inf
    uint16_t bf16 = (uint16_t)(((bits & 0x80000000) | rounded) >> 16);
    memcpy(out, &bf16, 2);
}
static void FloatToFp16Bytes(float val, uint8_t* out)
{
    uint32_t bits;
    memcpy(&bits, &val, 4);
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFF;
    uint16_t h;
    if (exp <= 0) {
        if (exp < -10) {
            h = (uint16_t)sign;
        } else {
            uint32_t sh = (uint32_t)(14 - exp), m = (mant | 0x800000) >> sh, rem = mant & ((1u << sh) - 1),
                     hlsb = 1u << (sh - 1);
            if (rem > hlsb || (rem == hlsb && (m & 1)))
                m++;
            h = (uint16_t)(sign | m);
        }
    } else if (exp >= 0x1F) {
        h = (uint16_t)(sign | 0x7C00 | (mant ? 1 : 0));
    } else {
        uint32_t m = mant >> 13, rem = mant & 0x1FFF, hlsb = 0x1000;
        if (rem > hlsb || (rem == hlsb && (m & 1)))
            m++;
        if (m > 0x3FF) {
            m = 0;
            exp++;
        }
        if (exp >= 0x1F)
            h = (uint16_t)(sign | 0x7C00);
        else
            h = (uint16_t)(sign | ((uint32_t)exp << 10) | m);
    }
    memcpy(out, &h, 2);
}
static void FloatToBytes(float val, uint8_t* out, DataType dt)
{
    if (dt == ge::DT_FLOAT) {
        memcpy(out, &val, 4);
    } else if (dt == ge::DT_FLOAT16) {
        FloatToFp16Bytes(val, out);
    } else {
        FloatToBf16Bytes(val, out);
    } // DT_BF16
}

int32_t GenInputData(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, DataType data_type,
                     int inputIndex)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    uint32_t elem_bytes = GetDataTypeSize(data_type);
    uint32_t data_len = size * elem_bytes;
    uint8_t* pData = new (std::nothrow) uint8_t[data_len];
    if (pData == nullptr) {
        return FAILED;
    }

    // inputIndex: 1=c, 2=a, 3=b, 4=beta(scalar), 5=alpha(scalar)
    float fillVal;
    if (inputIndex == 1)
        fillVal = 1.0f;
    else if (inputIndex == 2)
        fillVal = 2.0f;
    else if (inputIndex == 3)
        fillVal = 3.0f;
    else if (inputIndex == 4)
        fillVal = 0.5f;
    else
        fillVal = 1.5f;
    FloatToBytes(fillVal, pData, data_type); // 填首元素（目标 dtype 二进制）
    for (size_t i = 1; i < size; ++i) {
        memcpy(pData + i * elem_bytes, pData, elem_bytes); // 复制到所有元素
    }
    input_tensor = Tensor(input_tensor_desc, pData, data_len);
    delete[] pData;
    return SUCCESS;
}

int CreateOppInGraph(DataType inDtype, std::vector<ge::Tensor>& input, std::vector<Operator>& inputs,
                     std::vector<Operator>& outputs, Graph& graph)
{
    Status ret = SUCCESS;
    auto addMatOp = op::AddMatMatElementsPlus("add_mat_mat_elements_plus_1");
    std::vector<int64_t> xShape = {4, 8};
    std::vector<int64_t> scalarShape = {1};

    ADD_INPUT(1, c, inDtype, xShape);
    ADD_INPUT(2, a, inDtype, xShape);
    ADD_INPUT(3, b, inDtype, xShape);
    ADD_INPUT(4, beta, inDtype, scalarShape);
    ADD_INPUT(5, alpha, inDtype, scalarShape);
    ADD_OUTPUT(1, c, inDtype, xShape);

    outputs.push_back(addMatOp);
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    const char* graph_name = "add_mat_mat_elements_plus_ge_ir_test";
    Graph graph(graph_name);
    std::vector<ge::Tensor> input;

    printf("%s - INFO - [XIR]: Start to initialize ge using ge global options\n", GetTime().c_str());
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Initialize ge using ge global options failed\n", GetTime().c_str());
        return FAILED;
    }

    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};

    DataType inDtype = DT_FLOAT;
    if (argc > 1) {
        std::string dtypeArg(argv[1]);
        if (dtypeArg == "fp16") {
            inDtype = DT_FLOAT16;
        } else if (dtypeArg == "bf16") {
            inDtype = DT_BF16;
        }
    }
    printf("%s - INFO - [XIR]: Using dtype = %d\n", GetTime().c_str(), static_cast<int>(inDtype));

    ret = CreateOppInGraph(inDtype, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create op in graph failed\n", GetTime().c_str());
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    std::map<AscendString, AscendString> build_options = {};
    ge::Session* session = new Session(build_options);
    if (session == nullptr) {
        return FAILED;
    }

    std::map<AscendString, AscendString> graph_options = {};
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);
    if (ret != SUCCESS) {
        delete session;
        GEFinalize();
        return FAILED;
    }

    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        delete session;
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Session run ir compute graph success\n", GetTime().c_str());

    int output_num = output.size();
    for (int i = 0; i < output_num; i++) {
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        printf("%s - INFO - [XIR]: output %d shape size = %ld, dtype = %d\n", GetTime().c_str(), i, output_shape,
               static_cast<int>(output[i].GetTensorDesc().GetDataType()));

        if (output[i].GetTensorDesc().GetDataType() == DT_FLOAT && output_shape > 0) {
            float* outData = reinterpret_cast<float*>(output[i].GetData());
            // expected: c*beta + alpha*a*b = 1*0.5 + 1.5*2*3 = 9.5
            float expected = 1.0f * 0.5f + 1.5f * 2.0f * 3.0f;
            int64_t printNum = (output_shape < 10) ? output_shape : 10;
            for (int64_t j = 0; j < printNum; j++) {
                printf("  output[%ld] = %.6f (expected = %.6f)\n", j, outData[j], expected);
            }
        }
    }

    ge::AscendString error_msg = ge::GEGetErrorMsgV2();
    std::string error_str(error_msg.GetString());
    if (!error_str.empty()) {
        std::cout << "Error message: " << error_str << std::endl;
    }

    delete session;
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        return FAILED;
    }
    return SUCCESS;
}
