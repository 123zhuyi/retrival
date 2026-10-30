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
 * @file test_geir_fused_mul_add_nl2loss.cpp
 * @brief FusedMulAddNL2loss 图模式（GE IR）构图调用示例（ascend950 真机）
 *
 * 算子功能：y1 = x1 * x3[0] + x2；y2 = sum(x1^2 / 2)（0 维标量）
 *
 * 本示例构造一张仅含 op::FusedMulAddNL2loss 节点的计算图并在 ascend950 上执行：
 *   x1 = x2 = x3 = 2（shape {2,2} / {2,2} / {1}）
 *   => y1 全为 2*2+2 = 6；y2 = 4 * (2^2) / 2 = 8
 * 校验图模式全链路：proto 注册 / OpDef(ascend950) / infershape+inferDataType / tiling / kernel。
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "array_ops.h"
#include "ge_ir_build.h"

#include "../../op_graph/fused_mul_add_nl2loss_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

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

int32_t GenConstDataFloat32(const vector<int64_t>& shape, Tensor& tensor, TensorDesc& desc, float value)
{
    desc.SetRealDimCnt(shape.size());
    size_t size = 1;
    for (auto d : shape) {
        size *= d;
    }
    uint32_t dataLen = size * sizeof(float);
    float* data = new (std::nothrow) float[size];
    for (size_t i = 0; i < size; ++i) {
        data[i] = value;
    }
    tensor = Tensor(desc, (uint8_t*)data, dataLen);
    return SUCCESS;
}

int main()
{
    const char* graphName = "fused_mul_add_nl2loss_geir";
    Graph graph(graphName);
    std::vector<ge::Tensor> input;
    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};
    Status ret = SUCCESS;

    LOG_PRINT("%s - INFO - [XIR]: GEInitialize\n", GetTime().c_str());
    std::map<AscendString, AscendString> globalOptions = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    ret = ge::GEInitialize(globalOptions);
    if (ret != SUCCESS) {
        LOG_PRINT("%s - ERROR - [XIR]: GEInitialize failed\n", GetTime().c_str());
        return FAILED;
    }

    // 算子节点：x1/x2 shape {2,2}，x3 shape {1}，均为 fp32、常量 2.0
    auto fusedOp = op::FusedMulAddNL2loss("fused_mul_add_nl2loss");
    vector<int64_t> x1Shape = {2, 2};
    vector<int64_t> x2Shape = {2, 2};
    vector<int64_t> x3Shape = {1};
    vector<int64_t> y1Shape = {2, 2};
    vector<int64_t> y2Shape = {}; // 0 维标量（与 infershape 一致）

    // x1/x2/x3：Data 占位 + 常量数据 + 接入算子输入
    vector<vector<int64_t>*> inShapes = {&x1Shape, &x2Shape, &x3Shape};
    const char* inNames[3] = {"x1", "x2", "x3"};
    for (int i = 0; i < 3; i++) {
        auto data = op::Data("placeholder" + std::to_string(i)).set_attr_index(0);
        TensorDesc desc = TensorDesc(ge::Shape(*inShapes[i]), FORMAT_ND, DT_FLOAT);
        desc.SetPlacement(ge::kPlacementHost);
        desc.SetFormat(FORMAT_ND);
        Tensor tensor;
        ret = GenConstDataFloat32(*inShapes[i], tensor, desc, 2.0f);
        if (ret != SUCCESS) {
            LOG_PRINT("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());
            return FAILED;
        }
        data.update_input_desc_x(desc);
        input.push_back(tensor);
        graph.AddOp(data);
        switch (i) {
            case 0:
                fusedOp.set_input_x1(data);
                break;
            case 1:
                fusedOp.set_input_x2(data);
                break;
            default:
                fusedOp.set_input_x3(data);
                break;
        }
        inputs.push_back(data);
    }
    // y1/y2：声明输出 desc
    TensorDesc y1Desc = TensorDesc(ge::Shape(y1Shape), FORMAT_ND, DT_FLOAT);
    fusedOp.update_output_desc_y1(y1Desc);
    TensorDesc y2Desc = TensorDesc(ge::Shape(y2Shape), FORMAT_ND, DT_FLOAT);
    fusedOp.update_output_desc_y2(y2Desc);
    outputs.push_back(fusedOp);

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    LOG_PRINT("%s - INFO - [XIR]: AddGraph + RunGraph\n", GetTime().c_str());
    std::map<AscendString, AscendString> buildOptions = {};
    ge::Session* session = new Session(buildOptions);
    if (session == nullptr) {
        LOG_PRINT("%s - ERROR - [XIR]: Create session failed\n", GetTime().c_str());
        GEFinalize();
        return FAILED;
    }
    std::map<AscendString, AscendString> graphOptions = {};
    uint32_t graphId = 0;
    ret = session->AddGraph(graphId, graph, graphOptions);
    if (ret != SUCCESS) {
        LOG_PRINT("%s - ERROR - [XIR]: AddGraph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }

    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graphId, input, output);
    if (ret != SUCCESS) {
        LOG_PRINT("%s - ERROR - [XIR]: RunGraph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }
    LOG_PRINT("%s - INFO - [XIR]: RunGraph success, outputs=%zu\n", GetTime().c_str(), output.size());

    // 校验：y1 全为 6.0，y2 == 8.0
    int failCnt = 0;
    if (output.size() != 2) {
        LOG_PRINT("[CHECK] output num %zu != 2\n", output.size());
        failCnt++;
    } else {
        float* y1Data = reinterpret_cast<float*>(output[0].GetData());
        int64_t y1Size = output[0].GetTensorDesc().GetShape().GetShapeSize();
        for (int64_t i = 0; i < y1Size; i++) {
            LOG_PRINT("y1[%ld] = %f\n", i, y1Data[i]);
            if (fabsf(y1Data[i] - 6.0f) > 1e-6f) {
                failCnt++;
            }
        }
        float* y2Data = reinterpret_cast<float*>(output[1].GetData());
        int64_t y2Size = output[1].GetTensorDesc().GetShape().GetShapeSize();
        // 0 维标量在 GE 中 GetShapeSize() 返回 0，数据区仍为 1 个元素
        LOG_PRINT("y2 (0-dim scalar, shapeSize=%ld) = %f\n", y2Size, y2Data[0]);
        if (y2Size > 1 || fabsf(y2Data[0] - 8.0f) > 1e-6f) {
            failCnt++;
        }
    }
    LOG_PRINT("[CHECK] %s\n", failCnt == 0 ? "PASS" : "FAIL");

    delete session;
    ge::GEFinalize();
    return failCnt == 0 ? SUCCESS : FAILED;
}
