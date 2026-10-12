/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "array_ops.h"
#include "ge_api.h"
#include "ge_api_types.h"
#include "ge_error_codes.h"
#include "ge_ir_build.h"
#include "graph.h"
#include "tensor.h"
#include "types.h"

#include "cumulative_logsumexp_proto.h"

#define FAILED -1
#define SUCCESS 0

using ge::AscendString;
using ge::DataType;
using ge::DT_FLOAT;
using ge::DT_INT32;
using ge::FORMAT_ND;
using ge::Graph;
using ge::Operator;
using ge::Session;
using ge::Status;
using ge::Tensor;
using ge::TensorDesc;
using std::map;
using std::string;
using std::vector;

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

static string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime(&timep));
    return tmp;
}

static uint32_t GetDataTypeSize(DataType dt)
{
    int32_t dtypeSize = ge::GetSizeByDataType(dt);
    return dtypeSize > 0 ? static_cast<uint32_t>(dtypeSize) : 1U;
}

template <typename T>
static int32_t GenTensor(const vector<int64_t>& shape, const vector<T>& values, DataType dataType, Tensor& tensor)
{
    size_t elementCount = 1;
    for (int64_t dim : shape) {
        elementCount *= static_cast<size_t>(dim);
    }
    if (values.size() != elementCount) {
        return FAILED;
    }
    TensorDesc desc(ge::Shape(shape), FORMAT_ND, dataType);
    desc.SetPlacement(ge::kPlacementHost);
    desc.SetFormat(FORMAT_ND);
    desc.SetRealDimCnt(shape.size());
    tensor = Tensor(desc, reinterpret_cast<const uint8_t*>(values.data()), values.size() * sizeof(T));
    return SUCCESS;
}

static float LogAddExp(float a, float b)
{
    if (std::isinf(a) && a < 0.0f) {
        return b;
    }
    if (std::isinf(b) && b < 0.0f) {
        return a;
    }
    float m = std::max(a, b);
    return m + std::log1pf(std::exp(-std::fabs(a - b)));
}

static vector<float> Golden(const vector<float>& x, const vector<int64_t>& shape, int32_t axis, bool exclusive,
                            bool reverse)
{
    int64_t rank = static_cast<int64_t>(shape.size());
    int64_t normAxis = axis < 0 ? axis + rank : axis;
    int64_t inner = 1;
    for (int64_t i = normAxis + 1; i < rank; ++i) {
        inner *= shape[i];
    }
    int64_t axisLen = shape[normAxis];
    int64_t outer = static_cast<int64_t>(x.size()) / (axisLen * inner);
    vector<float> y(x.size(), 0.0f);
    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t in = 0; in < inner; ++in) {
            float acc = -INFINITY;
            for (int64_t step = 0; step < axisLen; ++step) {
                int64_t a = reverse ? axisLen - 1 - step : step;
                int64_t idx = (o * axisLen + a) * inner + in;
                if (!exclusive) {
                    acc = LogAddExp(acc, x[idx]);
                    y[idx] = acc;
                } else {
                    y[idx] = acc;
                    acc = LogAddExp(acc, x[idx]);
                }
            }
        }
    }
    return y;
}

static int CreateGraph(vector<Tensor>& input, vector<Operator>& inputs, vector<Operator>& outputs, Graph& graph)
{
    Status ret = SUCCESS;
    auto op = ge::op::CumulativeLogsumexp("cumulative_logsumexp1");
    vector<int64_t> xShape = {2, 4};
    vector<int64_t> axisShape = {1};
    vector<float> xData = {-3.0f, -2.0f, -1.0f, -0.5f, 0.25f, -0.75f, 1.0f, -4.0f};
    vector<int32_t> axisData = {1};

    auto x = ge::op::Data("x").set_attr_index(0);
    TensorDesc xDesc(ge::Shape(xShape), FORMAT_ND, DT_FLOAT);
    xDesc.SetPlacement(ge::kPlacementHost);
    xDesc.SetFormat(FORMAT_ND);
    Tensor xTensor;
    ret = GenTensor(xShape, xData, DT_FLOAT, xTensor);
    CHECK_RET(ret == SUCCESS, return FAILED);
    x.update_input_desc_x(xDesc);
    x.update_output_desc_y(xDesc);
    graph.AddOp(x);
    input.push_back(xTensor);
    inputs.push_back(x);
    op.set_input_x(x);

    auto axis = ge::op::Data("axis").set_attr_index(1);
    TensorDesc axisDesc(ge::Shape(axisShape), FORMAT_ND, DT_INT32);
    axisDesc.SetPlacement(ge::kPlacementHost);
    axisDesc.SetFormat(FORMAT_ND);
    Tensor axisTensor;
    ret = GenTensor(axisShape, axisData, DT_INT32, axisTensor);
    CHECK_RET(ret == SUCCESS, return FAILED);
    axis.update_input_desc_x(axisDesc);
    axis.update_output_desc_y(axisDesc);
    graph.AddOp(axis);
    input.push_back(axisTensor);
    inputs.push_back(axis);
    op.set_input_axis(axis);

    op.set_attr_exclusive(false);
    op.set_attr_reverse(false);
    TensorDesc yDesc(ge::Shape(xShape), FORMAT_ND, DT_FLOAT);
    op.update_output_desc_y(yDesc);
    outputs.push_back(op);
    return SUCCESS;
}

int main()
{
    const char* graphName = "cumulative_logsumexp_ge_path";
    Graph graph(graphName);
    vector<Tensor> input;
    vector<Operator> inputs;
    vector<Operator> outputs;

    printf("%s - INFO - GE initialize\n", GetTime().c_str());
    map<AscendString, AscendString> globalOptions = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(globalOptions);
    CHECK_RET(ret == SUCCESS, printf("%s - ERROR - GEInitialize failed: %u\n", GetTime().c_str(), ret); return FAILED);

    ret = CreateGraph(input, inputs, outputs, graph);
    CHECK_RET(ret == SUCCESS, ge::GEFinalize(); return FAILED);
    graph.SetInputs(inputs).SetOutputs(outputs);

    map<AscendString, AscendString> buildOptions;
    Session* session = new Session(buildOptions);
    CHECK_RET(session != nullptr, ge::GEFinalize(); return FAILED);

    uint32_t graphId = 0;
    map<AscendString, AscendString> graphOptions;
    ret = session->AddGraph(graphId, graph, graphOptions);
    CHECK_RET(ret == SUCCESS, printf("%s - ERROR - AddGraph failed: %u\n", GetTime().c_str(), ret); delete session;
              ge::GEFinalize(); return FAILED);

    vector<Tensor> output;
    ret = session->RunGraph(graphId, input, output);
    CHECK_RET(ret == SUCCESS, printf("%s - ERROR - RunGraph failed: %u\n", GetTime().c_str(), ret); delete session;
              ge::GEFinalize(); return FAILED);
    CHECK_RET(output.size() == 1, delete session; ge::GEFinalize(); return FAILED);

    const uint8_t* data = output[0].GetData();
    int64_t elementCount = output[0].GetTensorDesc().GetShape().GetShapeSize();
    const float* result = reinterpret_cast<const float*>(data);
    vector<float> xData = {-3.0f, -2.0f, -1.0f, -0.5f, 0.25f, -0.75f, 1.0f, -4.0f};
    vector<float> gold = Golden(xData, {2, 4}, 1, false, false);

    int failCount = 0;
    for (int64_t i = 0; i < elementCount; ++i) {
        float diff = std::fabs(result[i] - gold[i]);
        bool ok = diff <= (1e-4f + 1e-4f * std::fabs(gold[i]));
        printf("[%ld] out=%f gold=%f diff=%e %s\n", i, result[i], gold[i], diff, ok ? "OK" : "FAIL");
        if (!ok) {
            ++failCount;
        }
    }

    delete session;
    ge::GEFinalize();
    if (failCount == 0) {
        printf("GE cumulative_logsumexp PASS\n");
        return SUCCESS;
    }
    printf("GE cumulative_logsumexp FAIL: %d / %ld\n", failCount, elementCount);
    return FAILED;
}
