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
 * @file test_geir_add_mat_mat_elements_plus_precision.cpp
 * @brief AddMatMatElementsPlus 精度+性能测试（Ascend910B）
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

static float GetThreshold(ge::DataType dt)
{
    switch (dt) {
        case DT_FLOAT16:
            return 0.000977f;
        case DT_BF16:
            return 0.007812f;
        case DT_FLOAT:
            return 0.000122f;
        default:
            return 0.000122f;
    }
}

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

// Golden: c_out = c * beta + alpha * a * b (upcast to float for fp16/bf16)
static void ComputeGoldenFp32(const float* c, const float* a, const float* b, float betaVal, float alphaVal, float* out,
                              size_t n)
{
    for (size_t i = 0; i < n; i++)
        out[i] = c[i] * betaVal + alphaVal * a[i] * b[i];
}

struct PrecResult {
    bool pass;
    float mere;
    float mare;
    float threshold;
};

static PrecResult CheckPrecisionFp32(const float* actual, const float* golden, size_t n, ge::DataType dt)
{
    float threshold = GetThreshold(dt);
    float mereSum = 0.0f;
    float mareMax = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float diff = std::abs(actual[i] - golden[i]);
        float denom = std::abs(golden[i]) + 1e-7f;
        float relErr = diff / denom;
        mereSum += relErr;
        if (relErr > mareMax)
            mareMax = relErr;
    }
    float mere = mereSum / static_cast<float>(n);
    bool pass = (mere < threshold) && (mareMax < 10.0f * threshold);
    return {pass, mere, mareMax, threshold};
}

// dtype 转换辅助
static void FloatToBytes(const float* src, uint8_t* dst, size_t n, ge::DataType dt)
{
    if (dt == DT_FLOAT) {
        memcpy(dst, src, n * 4);
    } else if (dt == DT_FLOAT16) {
        for (size_t i = 0; i < n; i++)
            reinterpret_cast<__fp16*>(dst)[i] = static_cast<__fp16>(src[i]);
    } else { // BF16
        for (size_t i = 0; i < n; i++) {
            uint32_t bits;
            memcpy(&bits, &src[i], 4);
            uint16_t bf = static_cast<uint16_t>(bits >> 16);
            memcpy(&dst[i * 2], &bf, 2);
        }
    }
}

static void BytesToFloat(const uint8_t* src, float* dst, size_t n, ge::DataType dt)
{
    if (dt == DT_FLOAT) {
        memcpy(dst, src, n * 4);
    } else if (dt == DT_FLOAT16) {
        const __fp16* p = reinterpret_cast<const __fp16*>(src);
        for (size_t i = 0; i < n; i++)
            dst[i] = static_cast<float>(p[i]);
    } else { // BF16
        for (size_t i = 0; i < n; i++) {
            uint16_t bits;
            memcpy(&bits, &src[i * 2], 2);
            uint32_t fbits = static_cast<uint32_t>(bits) << 16;
            memcpy(&dst[i], &fbits, 4);
        }
    }
}

// 全局 GE session
static ge::Session* g_session = nullptr;
static uint32_t g_graphId = 0;

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

static int RunGraph(DataType dt, const vector<int64_t>& shape, const vector<uint8_t>& cData,
                    const vector<uint8_t>& aData, const vector<uint8_t>& bData, const vector<uint8_t>& betaData,
                    const vector<uint8_t>& alphaData, vector<uint8_t>& outData)
{
    Graph graph("amme_test");
    auto op_obj = op::AddMatMatElementsPlus("amme_1");

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

    size_t elemCount = 1;
    for (auto d : shape)
        elemCount *= d;
    size_t dataBytes = elemCount * GetDataTypeSize(dt);
    size_t scalarBytes = GetDataTypeSize(dt);

    vector<ge::Tensor> inputTensors;
    Tensor tc(descMain, const_cast<uint8_t*>(cData.data()), dataBytes);
    Tensor ta(descMain, const_cast<uint8_t*>(aData.data()), dataBytes);
    Tensor tb(descMain, const_cast<uint8_t*>(bData.data()), dataBytes);
    Tensor tbeta(descScalar, const_cast<uint8_t*>(betaData.data()), scalarBytes);
    Tensor talpha(descScalar, const_cast<uint8_t*>(alphaData.data()), scalarBytes);
    inputTensors.push_back(tc);
    inputTensors.push_back(ta);
    inputTensors.push_back(tb);
    inputTensors.push_back(tbeta);
    inputTensors.push_back(talpha);

    uint32_t gid = g_graphId++;
    std::map<AscendString, AscendString> gopts = {};
    if (g_session->AddGraph(gid, graph, gopts) != SUCCESS)
        return FAILED;

    vector<ge::Tensor> output;
    if (g_session->RunGraph(gid, inputTensors, output) != SUCCESS)
        return FAILED;

    if (!output.empty()) {
        outData.resize(output[0].GetSize());
        memcpy(outData.data(), output[0].GetData(), output[0].GetSize());
    }
    return SUCCESS;
}

struct TestCase {
    const char* name;
    vector<int64_t> shape;
    unsigned int seed;
};

static int RunPrecisionTests(DataType dt)
{
    const char* dtName = DtypeName(dt);
    size_t elemBytes = GetDataTypeSize(dt);
    int passCount = 0;
    int totalCount = 0;

    vector<TestCase> cases = {
        {"1d_128", {128}, 100},          {"1d_4096", {4096}, 101},
        {"1d_100000", {100000}, 102},    {"2d_4x8", {4, 8}, 103},
        {"2d_256x512", {256, 512}, 104}, {"3d_2x3x4", {2, 3, 4}, 105},
        {"3d_4x8x16", {4, 8, 16}, 106},  {"4d_2x4x8x16", {2, 4, 8, 16}, 107},
        {"edge_31", {31}, 108},          {"edge_33", {33}, 109},
    };

    float betaVal = 0.5f, alphaVal = 1.5f;

    printf("\n=== 精度测试 [%s] (threshold=%.6e) ===\n", dtName, GetThreshold(dt));
    printf("%-16s %8s %14s %14s %8s\n", "用例", "元素数", "MERE", "MARE", "结果");

    for (const auto& tc : cases) {
        size_t elemCount = 1;
        for (auto d : tc.shape)
            elemCount *= d;
        size_t dataBytes = elemCount * elemBytes;
        size_t scalarBytes = elemBytes;

        vector<float> cF(elemCount), aF(elemCount), bF(elemCount), gF(elemCount);
        FillRandom(cF.data(), elemCount, tc.seed);
        FillRandom(aF.data(), elemCount, tc.seed + 1);
        FillRandom(bF.data(), elemCount, tc.seed + 2);

        // 先量化输入到目标 dtype，再反量化回 float，匹配 kernel 的数据流：
        // kernel: input(T) → Cast → float → compute → Cast → T
        vector<uint8_t> cData(dataBytes), aData(dataBytes), bData(dataBytes);
        FloatToBytes(cF.data(), cData.data(), elemCount, dt);
        FloatToBytes(aF.data(), aData.data(), elemCount, dt);
        FloatToBytes(bF.data(), bData.data(), elemCount, dt);

        // 用量化后的输入计算 golden
        vector<float> cQ(elemCount), aQ(elemCount), bQ(elemCount);
        BytesToFloat(cData.data(), cQ.data(), elemCount, dt);
        BytesToFloat(aData.data(), aQ.data(), elemCount, dt);
        BytesToFloat(bData.data(), bQ.data(), elemCount, dt);
        ComputeGoldenFp32(cQ.data(), aQ.data(), bQ.data(), betaVal, alphaVal, gF.data(), elemCount);

        // golden 输出也量化到目标 dtype（匹配 kernel 的 Cast 输出）
        if (dt != DT_FLOAT) {
            vector<uint8_t> gQ(dataBytes);
            FloatToBytes(gF.data(), gQ.data(), elemCount, dt);
            BytesToFloat(gQ.data(), gF.data(), elemCount, dt);
        }

        vector<uint8_t> betaData(scalarBytes), alphaData(scalarBytes);
        FloatToBytes(&betaVal, betaData.data(), 1, dt);
        FloatToBytes(&alphaVal, alphaData.data(), 1, dt);

        vector<uint8_t> outData;
        int ret = RunGraph(dt, tc.shape, cData, aData, bData, betaData, alphaData, outData);
        totalCount++;

        if (ret != SUCCESS || outData.empty()) {
            printf("%-16s %8zu %14s %14s %8s\n", tc.name, elemCount, "N/A", "N/A", "FAIL");
            continue;
        }

        vector<float> actualF(elemCount);
        BytesToFloat(outData.data(), actualF.data(), elemCount, dt);

        PrecResult r = CheckPrecisionFp32(actualF.data(), gF.data(), elemCount, dt);
        if (r.pass)
            passCount++;
        printf("%-16s %8zu %14.6e %14.6e %8s\n", tc.name, elemCount, r.mere, r.mare, r.pass ? "PASS" : "FAIL");
    }

    printf("[%s] %d/%d 通过\n", dtName, passCount, totalCount);
    return passCount == totalCount ? SUCCESS : FAILED;
}

static int RunPerfTest(DataType dt, size_t elemCount, int iterations)
{
    const char* dtName = DtypeName(dt);
    size_t elemBytes = GetDataTypeSize(dt);
    size_t dataBytes = elemCount * elemBytes;
    vector<int64_t> shape = {static_cast<int64_t>(elemCount)};

    vector<float> cF(elemCount), aF(elemCount), bF(elemCount);
    FillRandom(cF.data(), elemCount, 42);
    FillRandom(aF.data(), elemCount, 43);
    FillRandom(bF.data(), elemCount, 44);

    vector<uint8_t> cData(dataBytes), aData(dataBytes), bData(dataBytes);
    FloatToBytes(cF.data(), cData.data(), elemCount, dt);
    FloatToBytes(aF.data(), aData.data(), elemCount, dt);
    FloatToBytes(bF.data(), bData.data(), elemCount, dt);

    float betaVal = 0.5f, alphaVal = 1.5f;
    vector<uint8_t> betaData(elemBytes), alphaData(elemBytes);
    FloatToBytes(&betaVal, betaData.data(), 1, dt);
    FloatToBytes(&alphaVal, alphaData.data(), 1, dt);

    vector<uint8_t> outData;
    RunGraph(dt, shape, cData, aData, bData, betaData, alphaData, outData); // warmup

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++)
        RunGraph(dt, shape, cData, aData, bData, betaData, alphaData, outData);
    auto t1 = std::chrono::high_resolution_clock::now();

    double totalMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double avgMs = totalMs / iterations;
    double meps = elemCount / (avgMs / 1000.0) / 1e6;

    printf("| %-6s | %8zu | %8d | %10.2f | %12.2f |\n", dtName, elemCount, iterations, avgMs, meps);
    return SUCCESS;
}

int main()
{
    if (InitGE() != SUCCESS) {
        printf("GE初始化失败\n");
        return FAILED;
    }

    printf("========================================\n");
    printf(" AddMatMatElementsPlus 精度测试 (Ascend910B)\n");
    printf("========================================\n");

    int allPass = SUCCESS;
    for (auto dt : {DT_FLOAT, DT_FLOAT16, DT_BF16}) {
        if (RunPrecisionTests(dt) != SUCCESS)
            allPass = FAILED;
    }

    printf("\n========================================\n");
    printf(" AddMatMatElementsPlus 性能测试 (Ascend910B)\n");
    printf("========================================\n");
    printf("| dtype  | 元素数   | 迭代次数 | 平均耗时(ms) | 吞吐(M/s) |\n");
    printf("|--------|----------|----------|--------------|-----------|\n");
    for (auto dt : {DT_FLOAT, DT_FLOAT16, DT_BF16})
        RunPerfTest(dt, 1000000, 30);

    printf("\n=== 总结: 精度 %s ===\n", allPass == SUCCESS ? "ALL PASS" : "SOME FAILED");

    FiniGE();
    return allPass;
}
