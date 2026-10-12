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
 * @file test_geir_add_mat_mat_elements_plus_perf.cpp
 * @brief AddMatMatElementsPlus 性能测试（图编译一次，RunGraph 多次，测纯 kernel 执行时间）
 */

#include <iostream>
#include <string.h>
#include <stdint.h>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <cstdlib>
#include <chrono>

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "array_ops.h"
#include "ge_ir_build.h"

#include "../op_graph/add_mat_mat_elements_plus_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;

static uint32_t GetDataTypeSize(DataType dt)
{
    if (dt == ge::DT_FLOAT)
        return 4;
    if (dt == ge::DT_FLOAT16 || dt == ge::DT_BF16)
        return 2;
    return 4;
}

static const char* DtypeName(ge::DataType dt)
{
    switch (dt) {
        case DT_FLOAT16:
            return "fp16";
        case DT_BF16:
            return "bf16";
        case DT_FLOAT:
            return "fp32";
        default:
            return "unknown";
    }
}

static void FillRandom(float* data, size_t n, unsigned int seed)
{
    srand(seed);
    for (size_t i = 0; i < n; i++)
        data[i] = (static_cast<float>(rand()) / RAND_MAX) * 4.0f - 2.0f;
}

static void FloatToBytes(const float* src, uint8_t* dst, size_t n, ge::DataType dt)
{
    if (dt == DT_FLOAT) {
        memcpy(dst, src, n * 4);
    } else if (dt == DT_FLOAT16) {
        for (size_t i = 0; i < n; i++)
            reinterpret_cast<__fp16*>(dst)[i] = static_cast<__fp16>(src[i]);
    } else {
        for (size_t i = 0; i < n; i++) {
            uint32_t bits;
            memcpy(&bits, &src[i], 4);
            uint16_t bf = static_cast<uint16_t>(bits >> 16);
            memcpy(&dst[i * 2], &bf, 2);
        }
    }
}

static ge::Session* g_session = nullptr;

static int InitGE()
{
    std::map<AscendString, AscendString> opts = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    if (ge::GEInitialize(opts) != SUCCESS)
        return FAILED;
    std::map<AscendString, AscendString> sessOpts = {};
    g_session = new Session(sessOpts);
    return g_session ? SUCCESS : FAILED;
}

static void FiniGE()
{
    delete g_session;
    g_session = nullptr;
    ge::GEFinalize();
}

static int BuildAndAddGraph(uint32_t graphId, DataType dt, const vector<int64_t>& shape,
                            vector<ge::Tensor>& inputTensors)
{
    size_t elemCount = 1;
    for (auto d : shape)
        elemCount *= d;
    size_t elemBytes = GetDataTypeSize(dt);
    size_t dataBytes = elemCount * elemBytes;

    vector<float> cF(elemCount), aF(elemCount), bF(elemCount);
    FillRandom(cF.data(), elemCount, 42);
    FillRandom(aF.data(), elemCount, 43);
    FillRandom(bF.data(), elemCount, 44);
    float betaVal = 0.5f, alphaVal = 1.5f;

    vector<uint8_t> cData(dataBytes), aData(dataBytes), bData(dataBytes);
    FloatToBytes(cF.data(), cData.data(), elemCount, dt);
    FloatToBytes(aF.data(), aData.data(), elemCount, dt);
    FloatToBytes(bF.data(), bData.data(), elemCount, dt);

    vector<uint8_t> betaData(elemBytes), alphaData(elemBytes);
    FloatToBytes(&betaVal, betaData.data(), 1, dt);
    FloatToBytes(&alphaVal, alphaData.data(), 1, dt);

    Graph graph("amme_perf");
    auto op_obj = op::AddMatMatElementsPlus("amme_perf_1");
    auto dC = op::Data("dC").set_attr_index(0);
    auto dA = op::Data("dA").set_attr_index(1);
    auto dB = op::Data("dB").set_attr_index(2);
    auto dBeta = op::Data("dBeta").set_attr_index(3);
    auto dAlpha = op::Data("dAlpha").set_attr_index(4);

    TensorDesc descMain(ge::Shape(shape), FORMAT_ND, dt);
    descMain.SetPlacement(ge::kPlacementHost);
    TensorDesc descScalar(ge::Shape({1}), FORMAT_ND, dt);
    descScalar.SetPlacement(ge::kPlacementHost);

    dC.update_input_desc_x(descMain);
    dA.update_input_desc_x(descMain);
    dB.update_input_desc_x(descMain);
    dBeta.update_input_desc_x(descScalar);
    dAlpha.update_input_desc_x(descScalar);

    op_obj.set_input_c(dC);
    op_obj.set_input_a(dA);
    op_obj.set_input_b(dB);
    op_obj.set_input_beta(dBeta);
    op_obj.set_input_alpha(dAlpha);
    op_obj.update_output_desc_c(descMain);

    graph.AddOp(dC);
    graph.AddOp(dA);
    graph.AddOp(dB);
    graph.AddOp(dBeta);
    graph.AddOp(dAlpha);
    vector<Operator> inputs{dC, dA, dB, dBeta, dAlpha};
    vector<Operator> outputs{op_obj};
    graph.SetInputs(inputs).SetOutputs(outputs);

    Tensor tc(descMain, const_cast<uint8_t*>(cData.data()), dataBytes);
    Tensor ta(descMain, const_cast<uint8_t*>(aData.data()), dataBytes);
    Tensor tb(descMain, const_cast<uint8_t*>(bData.data()), dataBytes);
    Tensor tbeta(descScalar, const_cast<uint8_t*>(betaData.data()), elemBytes);
    Tensor talpha(descScalar, const_cast<uint8_t*>(alphaData.data()), elemBytes);
    inputTensors.push_back(tc);
    inputTensors.push_back(ta);
    inputTensors.push_back(tb);
    inputTensors.push_back(tbeta);
    inputTensors.push_back(talpha);

    std::map<AscendString, AscendString> gopts = {};
    return g_session->AddGraph(graphId, graph, gopts);
}

static double MeasurePerf(DataType dt, size_t elemCount, int iterations, uint32_t graphId)
{
    const char* dtName = DtypeName(dt);
    vector<int64_t> shape = {static_cast<int64_t>(elemCount)};

    // 图编译一次
    vector<ge::Tensor> inputTensors;
    if (BuildAndAddGraph(graphId, dt, shape, inputTensors) != SUCCESS) {
        printf("| %-6s | 编译失败 |\n", dtName);
        return -1;
    }

    // warmup（多次，确保图编译透彻）
    vector<ge::Tensor> output;
    for (int w = 0; w < 5; w++) {
        g_session->RunGraph(graphId, inputTensors, output);
    }

    // 打印前5次逐次耗时（排查编译开销）
    printf("  [%s] 逐次耗时:", dtName);
    for (int i = 0; i < 5; i++) {
        auto s = std::chrono::high_resolution_clock::now();
        g_session->RunGraph(graphId, inputTensors, output);
        auto e = std::chrono::high_resolution_clock::now();
        printf(" %.2fms", std::chrono::duration<double, std::milli>(e - s).count());
    }
    printf("\n");

    // 计时（复用已编译的图）
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        g_session->RunGraph(graphId, inputTensors, output);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double totalMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double avgMs = totalMs / iterations;
    double meps = elemCount / (avgMs / 1000.0) / 1e6;

    printf("| %-6s | %8zu | %8d | %10.2f | %12.2f |\n", dtName, elemCount, iterations, avgMs, meps);
    return avgMs;
}

int main()
{
    if (InitGE() != SUCCESS) {
        printf("GE初始化失败\n");
        return FAILED;
    }

    size_t elemCount = 1000000;
    int iterations = 100;

    printf("============================================================\n");
    printf(" AddMatMatElementsPlus 性能测试 (Ascend910B, 图复用)\n");
    printf("============================================================\n");
    printf("| dtype  | 元素数   | 迭代次数 | 平均耗时(ms) | 吞吐(M/s) |\n");
    printf("|--------|----------|----------|--------------|-----------|\n");

    double fp32Ms = MeasurePerf(DT_FLOAT, elemCount, iterations, 0);
    double fp16Ms = MeasurePerf(DT_FLOAT16, elemCount, iterations, 1);
    double bf16Ms = MeasurePerf(DT_BF16, elemCount, iterations, 2);

    printf("\n=== 总结 ===\n");
    printf("自定义算子(geir): fp32=%.2fms, fp16=%.2fms, bf16=%.2fms\n", fp32Ms, fp16Ms, bf16Ms);
    printf("（对比 torch_npu 标杆见 tests/perf_baseline_torchnpu.py 输出）\n");

    FiniGE();
    return SUCCESS;
}
