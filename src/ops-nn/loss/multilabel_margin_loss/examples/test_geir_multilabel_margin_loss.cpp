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
 * \file test_geir_multilabel_margin_loss.cpp
 * \brief MultilabelMarginLoss GE IR (graph-mode) construction example.
 */

#include <ctime>
#include <cstdint>
#include <iostream>
#include <map>
#include <new>
#include <string>
#include <vector>

#include "ge_api.h"
#include "ge_api_types.h"
#include "ge_error_codes.h"
#include "ge_ir_build.h"
#include "graph.h"
#include "array_ops.h"
#include "tensor.h"
#include "types.h"
#include "../op_graph/multilabel_margin_loss_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::string;
using std::vector;

namespace {
constexpr uint32_t INT32_BYTE_SIZE = 4;
constexpr uint32_t FP16_BYTE_SIZE = 2;
constexpr uint32_t FP32_BYTE_SIZE = 4;

string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime(&timep));
    return tmp;
}

int64_t GetShapeSize(const vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto dim : shape) {
        shapeSize *= dim;
    }
    return shapeSize;
}

uint32_t GetDataTypeSize(DataType dt)
{
    if (dt == ge::DT_FLOAT16 || dt == ge::DT_BF16) {
        return FP16_BYTE_SIZE;
    }
    return FP32_BYTE_SIZE; // DT_FLOAT / DT_INT32
}

int32_t GenInputData(const vector<int64_t>& shape, Tensor& tensor, TensorDesc& tensorDesc, DataType dataType)
{
    tensorDesc.SetRealDimCnt(shape.size());
    size_t elementNum = static_cast<size_t>(GetShapeSize(shape));
    size_t dataLen = elementNum * GetDataTypeSize(dataType);
    uint8_t* data = new (std::nothrow) uint8_t[dataLen];
    if (data == nullptr) {
        printf("%s - ERROR - [XIR]: Alloc input data failed\n", GetTime().c_str());
        return FAILED;
    }
    for (size_t i = 0; i < dataLen; ++i) {
        data[i] = static_cast<uint8_t>(i % 23);
    }
    tensor = Tensor(tensorDesc, data, dataLen);
    delete[] data;
    return SUCCESS;
}

struct MultilabelMarginLossCase {
    const char* name;
    DataType xDtype;       // float32 / float16 / bfloat16
    int64_t n;             // rows
    int64_t c;             // classes
    const char* reduction; // none / mean / sum
};

int CreateMultilabelMarginLossGraph(const MultilabelMarginLossCase& tc, Graph& graph, vector<Tensor>& input,
                                    vector<Operator>& inputs, vector<Operator>& outputs)
{
    auto op = ge::op::MultilabelMarginLoss("multilabel_margin_loss");

    vector<int64_t> xShape = {tc.n, tc.c};

    // x
    TensorDesc xDesc(ge::Shape(xShape), FORMAT_ND, tc.xDtype);
    xDesc.SetPlacement(ge::kPlacementHost);
    Tensor xTensor;
    if (GenInputData(xShape, xTensor, xDesc, tc.xDtype) != SUCCESS) {
        return FAILED;
    }
    auto xData = op::Data("x").set_attr_index(0);
    xData.update_input_desc_x(xDesc);
    xData.update_output_desc_y(xDesc);
    graph.AddOp(xData);
    op.set_input_x(xData);
    input.push_back(xTensor);
    inputs.push_back(xData);

    // target (int32, same shape as x)
    TensorDesc targetDesc(ge::Shape(xShape), FORMAT_ND, ge::DT_INT32);
    targetDesc.SetPlacement(ge::kPlacementHost);
    Tensor targetTensor;
    if (GenInputData(xShape, targetTensor, targetDesc, ge::DT_INT32) != SUCCESS) {
        return FAILED;
    }
    auto targetData = op::Data("target").set_attr_index(1);
    targetData.update_input_desc_x(targetDesc);
    targetData.update_output_desc_y(targetDesc);
    graph.AddOp(targetData);
    op.set_input_target(targetData);
    input.push_back(targetTensor);
    inputs.push_back(targetData);

    // outputs: y (none+2D -> [N], else scalar [1]); is_target = target shape
    const bool isNone = (string(tc.reduction) == "none");
    vector<int64_t> yShape = isNone ? vector<int64_t>{tc.n} : vector<int64_t>{1};
    TensorDesc yDesc(ge::Shape(yShape), FORMAT_ND, tc.xDtype);
    TensorDesc isTargetDesc(ge::Shape(xShape), FORMAT_ND, ge::DT_INT32);
    op.update_output_desc_y(yDesc);
    op.update_output_desc_is_target(isTargetDesc);

    op.set_attr_reduction(tc.reduction);

    outputs.push_back(op);
    return SUCCESS;
}

int RunMultilabelMarginLossCase(const MultilabelMarginLossCase& tc)
{
    printf("%s - INFO - [XIR]: Run %s\n", GetTime().c_str(), tc.name);

    Graph graph(tc.name);
    vector<Tensor> input;
    vector<Operator> inputs;
    vector<Operator> outputs;
    if (CreateMultilabelMarginLossGraph(tc, graph, input, inputs, outputs) != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create graph failed\n", GetTime().c_str());
        return FAILED;
    }
    graph.SetInputs(inputs).SetOutputs(outputs);

    std::map<AscendString, AscendString> buildOptions = {};
    Session* session = new (std::nothrow) Session(buildOptions);
    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create ir session failed\n", GetTime().c_str());
        return FAILED;
    }

    uint32_t graphId = 0;
    std::map<AscendString, AscendString> graphOptions = {};
    if (session->AddGraph(graphId, graph, graphOptions) != SUCCESS) {
        printf("%s - ERROR - [XIR]: Add graph failed\n", GetTime().c_str());
        delete session;
        return FAILED;
    }

    vector<Tensor> output;
    if (session->RunGraph(graphId, input, output) != SUCCESS) {
        printf("%s - ERROR - [XIR]: Run %s graph failed\n", GetTime().c_str(), tc.name);
        delete session;
        return FAILED;
    }

    for (size_t i = 0; i < output.size(); ++i) {
        const TensorDesc& desc = output[i].GetTensorDesc();
        std::cout << tc.name << " output " << i << " dtype: " << desc.GetDataType()
                  << " shape size = " << desc.GetShape().GetShapeSize() << std::endl;
    }
    printf("%s - INFO - [XIR]: Run %s graph success\n", GetTime().c_str(), tc.name);
    delete session;
    return SUCCESS;
}
} // namespace

int main(int argc, char* argv[])
{
    if (argc > 1) {
        std::cout << argv[1] << std::endl;
    }

    printf("%s - INFO - [XIR]: Start to initialize ge\n", GetTime().c_str());
    std::map<AscendString, AscendString> globalOptions = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    if (ge::GEInitialize(globalOptions) != SUCCESS) {
        printf("%s - ERROR - [XIR]: Initialize ge failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Initialize ge success\n", GetTime().c_str());

    vector<MultilabelMarginLossCase> testCases = {
        {"fp32_none_4x8", ge::DT_FLOAT, 4, 8, "none"},
        {"fp16_mean_4x8", ge::DT_FLOAT16, 4, 8, "mean"},
        {"bf16_sum_4x8", ge::DT_BF16, 4, 8, "sum"},
    };

    for (const auto& tc : testCases) {
        if (RunMultilabelMarginLossCase(tc) != SUCCESS) {
            (void)ge::GEFinalize();
            return FAILED;
        }
    }

    if (ge::GEFinalize() != SUCCESS) {
        printf("%s - ERROR - [XIR]: Finalize ge failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize ge success\n", GetTime().c_str());
    return SUCCESS;
}
