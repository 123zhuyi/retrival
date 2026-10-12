/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// End-to-end GE-IR example for FusedMulAdd on Ascend 950.
//
// Each test case builds a one-op graph, runs it on device, and checks the
// output against a pre-computed expected value. The example exits with
// status 0 only if every case passes.
//
// Cases:
//   1. fp32 same shape [2,2]     :  x1=2,  x2=3, x3=4  -> y=10
//   2. fp16 same shape [4]       :  x1=1.5,x2=2, x3=0.5 -> y=3.5
//   3. int32 same shape [3]      :  x1=2,  x2=3, x3=5  -> y=11
//   4. fp32 broadcast            :  x1=[4,1]=2, x2=[1,4]=3, x3=[]/scalar=1 -> y[4,4]=7

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

#include "nn_other.h"
#include "../../op_graph/fused_mul_add_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

static string GetTime()
{
    time_t t;
    time(&t);
    char buf[64];
    strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", localtime(&t));
    return buf;
}

static uint32_t GetDataTypeSize(DataType dt)
{
    if (dt == ge::DT_FLOAT)   return 4;
    if (dt == ge::DT_FLOAT16) return 2;
    if (dt == ge::DT_INT32)   return 4;
    return 1;
}

// IEEE-754 fp32 -> half (round-to-nearest-even).
static uint16_t FloatToHalfBits(float v)
{
    uint32_t x;
    memcpy(&x, &v, sizeof(x));
    uint32_t sign  = (x >> 31) & 0x1;
    uint32_t exp32 = (x >> 23) & 0xff;
    uint32_t mant  = x & 0x7fffff;
    uint16_t out;
    if (exp32 == 0xff) {
        out = (uint16_t)((sign << 15) | 0x7c00 | (mant ? 0x200 : 0));
    } else if (exp32 == 0) {
        out = (uint16_t)(sign << 15);
    } else {
        int32_t newExp = (int32_t)exp32 - 127 + 15;
        if (newExp >= 31) {
            out = (uint16_t)((sign << 15) | 0x7c00);
        } else if (newExp <= 0) {
            out = (uint16_t)(sign << 15);
        } else {
            uint32_t roundBit = (mant >> 12) & 0x1;
            uint16_t m = (uint16_t)(mant >> 13);
            out = (uint16_t)((sign << 15) | (newExp << 10) | m);
            if (roundBit) out++;
        }
    }
    return out;
}

static float HalfBitsToFloat(uint16_t h)
{
    uint32_t sign  = (h >> 15) & 0x1;
    uint32_t exp16 = (h >> 10) & 0x1f;
    uint32_t mant  = h & 0x3ff;
    uint32_t out;
    if (exp16 == 0) {
        out = sign << 31;
    } else if (exp16 == 31) {
        out = (sign << 31) | 0x7f800000 | (mant << 13);
    } else {
        uint32_t newExp = exp16 - 15 + 127;
        out = (sign << 31) | (newExp << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &out, sizeof(f));
    return f;
}

// Allocate a tensor and fill with a single value of the given dtype.
static int32_t GenConstTensor(
    const vector<int64_t>& shape, DataType dtype, double value, Tensor& tensor, TensorDesc& desc)
{
    desc.SetRealDimCnt(shape.size());
    size_t numel = 1;
    for (auto d : shape) numel *= (size_t)d;
    size_t bytes = numel * GetDataTypeSize(dtype);

    if (dtype == ge::DT_FLOAT) {
        float* p = new (std::nothrow) float[numel];
        for (size_t i = 0; i < numel; ++i) p[i] = (float)value;
        tensor = Tensor(desc, (uint8_t*)p, bytes);
    } else if (dtype == ge::DT_FLOAT16) {
        uint16_t* p = new (std::nothrow) uint16_t[numel];
        uint16_t v = FloatToHalfBits((float)value);
        for (size_t i = 0; i < numel; ++i) p[i] = v;
        tensor = Tensor(desc, (uint8_t*)p, bytes);
    } else if (dtype == ge::DT_INT32) {
        int32_t* p = new (std::nothrow) int32_t[numel];
        for (size_t i = 0; i < numel; ++i) p[i] = (int32_t)value;
        tensor = Tensor(desc, (uint8_t*)p, bytes);
    } else {
        return FAILED;
    }
    return SUCCESS;
}

// Build a graph that computes y = x1 * x2 + x3 with three constant inputs.
static int BuildFusedMulAddGraph(
    Graph& graph, std::vector<ge::Tensor>& inputData, std::vector<Operator>& inputs,
    std::vector<Operator>& outputs, DataType dtype, double v1, double v2, double v3,
    const vector<int64_t>& s1, const vector<int64_t>& s2, const vector<int64_t>& s3,
    const vector<int64_t>& outShape, const string& caseTag)
{
    auto op = ge::op::FusedMulAdd("fma_" + caseTag);

    // Input x1
    {
        auto ph = ge::op::Data("ph1_" + caseTag).set_attr_index(0);
        TensorDesc d(ge::Shape(s1), FORMAT_ND, dtype);
        d.SetPlacement(ge::kPlacementHost);
        d.SetFormat(FORMAT_ND);
        Tensor t;
        if (GenConstTensor(s1, dtype, v1, t, d) != SUCCESS) return FAILED;
        ph.update_input_desc_x(d);
        inputData.push_back(t);
        graph.AddOp(ph);
        op.set_input_x1(ph);
        inputs.push_back(ph);
    }
    // Input x2
    {
        auto ph = ge::op::Data("ph2_" + caseTag).set_attr_index(0);
        TensorDesc d(ge::Shape(s2), FORMAT_ND, dtype);
        d.SetPlacement(ge::kPlacementHost);
        d.SetFormat(FORMAT_ND);
        Tensor t;
        if (GenConstTensor(s2, dtype, v2, t, d) != SUCCESS) return FAILED;
        ph.update_input_desc_x(d);
        inputData.push_back(t);
        graph.AddOp(ph);
        op.set_input_x2(ph);
        inputs.push_back(ph);
    }
    // Input x3
    {
        auto ph = ge::op::Data("ph3_" + caseTag).set_attr_index(0);
        TensorDesc d(ge::Shape(s3), FORMAT_ND, dtype);
        d.SetPlacement(ge::kPlacementHost);
        d.SetFormat(FORMAT_ND);
        Tensor t;
        if (GenConstTensor(s3, dtype, v3, t, d) != SUCCESS) return FAILED;
        ph.update_input_desc_x(d);
        inputData.push_back(t);
        graph.AddOp(ph);
        op.set_input_x3(ph);
        inputs.push_back(ph);
    }
    // Output y
    {
        TensorDesc d(ge::Shape(outShape), FORMAT_ND, dtype);
        op.update_output_desc_y(d);
    }
    outputs.push_back(op);
    return SUCCESS;
}

struct CaseSpec {
    string                 tag;
    DataType               dtype;
    double                 v1, v2, v3;
    vector<int64_t>        s1, s2, s3, outShape;
    double                 expected;
    double                 absTol;
};

static int RunOneCase(const CaseSpec& spec)
{
    printf("\n%s - INFO - [XIR]: ===== case %s start =====\n", GetTime().c_str(), spec.tag.c_str());
    Graph graph(("g_" + spec.tag).c_str());
    std::vector<ge::Tensor> inputData;
    std::vector<Operator> inputs;
    std::vector<Operator> outputs;
    if (BuildFusedMulAddGraph(graph, inputData, inputs, outputs, spec.dtype,
                              spec.v1, spec.v2, spec.v3,
                              spec.s1, spec.s2, spec.s3, spec.outShape, spec.tag) != SUCCESS) {
        printf("BuildFusedMulAddGraph failed\n");
        return FAILED;
    }
    graph.SetInputs(inputs).SetOutputs(outputs);

    std::map<AscendString, AscendString> buildOpts;
    ge::Session* session = new Session(buildOpts);
    if (!session) return FAILED;

    std::map<AscendString, AscendString> graphOpts;
    uint32_t graphId = 1;
    if (session->AddGraph(graphId, graph, graphOpts) != SUCCESS) {
        printf("AddGraph failed\n");
        delete session;
        return FAILED;
    }

    std::vector<ge::Tensor> output;
    Status ret = session->RunGraph(graphId, inputData, output);
    if (ret != SUCCESS) {
        printf("RunGraph failed\n");
        delete session;
        return FAILED;
    }

    int64_t expNumel = 1;
    for (auto d : spec.outShape) expNumel *= d;

    int failCount = 0;
    for (size_t i = 0; i < output.size(); ++i) {
        int64_t numel = output[i].GetTensorDesc().GetShape().GetShapeSize();
        if (numel != expNumel) {
            printf("  SHAPE MISMATCH: output numel=%ld expected=%ld\n", (long)numel, (long)expNumel);
            failCount++;
        }
        uint8_t* raw = output[i].GetData();
        for (int64_t j = 0; j < numel; ++j) {
            double got;
            if (spec.dtype == ge::DT_FLOAT) {
                got = (double)reinterpret_cast<const float*>(raw)[j];
            } else if (spec.dtype == ge::DT_FLOAT16) {
                got = (double)HalfBitsToFloat(reinterpret_cast<const uint16_t*>(raw)[j]);
            } else { // INT32
                got = (double)reinterpret_cast<const int32_t*>(raw)[j];
            }
            double diff = std::abs(got - spec.expected);
            if (diff > spec.absTol) {
                printf("  MISMATCH @%ld: got=%.6f expected=%.6f diff=%.6f\n",
                       (long)j, got, spec.expected, diff);
                failCount++;
            }
        }
    }
    delete session;

    if (failCount != 0) {
        printf("%s - FAIL - case %s: %d mismatches (numel=%ld)\n",
               GetTime().c_str(), spec.tag.c_str(), failCount,
               (long)output[0].GetTensorDesc().GetShape().GetShapeSize());
        return FAILED;
    }
    printf("%s - PASS - case %s (expected=%.4f)\n",
           GetTime().c_str(), spec.tag.c_str(), spec.expected);
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    (void)argc; (void)argv;

    std::map<AscendString, AscendString> globalOpts = {
        {"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    if (ge::GEInitialize(globalOpts) != SUCCESS) {
        printf("GEInitialize failed\n");
        return FAILED;
    }

    std::vector<CaseSpec> cases = {
        // tag                  dtype           v1   v2   v3   s1        s2        s3       out       expected absTol
        {"fp32_same_shape",   ge::DT_FLOAT,   2.0, 3.0, 4.0, {2,2},    {2,2},    {2,2},   {2,2},    10.0,    1e-5},
        {"fp16_same_shape",   ge::DT_FLOAT16, 1.5, 2.0, 0.5, {4},      {4},      {4},     {4},      3.5,     5e-3},
        {"int32_same_shape",  ge::DT_INT32,   2.0, 3.0, 5.0, {3},      {3},      {3},     {3},      11.0,    0.5 },
        {"fp32_broadcast",    ge::DT_FLOAT,   2.0, 3.0, 1.0, {4,1},    {1,4},    {1},     {4,4},    7.0,     1e-5},
        // --- extra broadcast scenarios ---
        // x1 scalar, x2 full 2-D, x3 scalar
        {"fp32_bc_scalar",    ge::DT_FLOAT,   2.0, 3.0, 1.0, {1},      {3,4},    {1},     {3,4},    7.0,     1e-5},
        // row x column mutual broadcast {1,5} x {5,1} -> {5,5}, x3 full
        {"fp32_bc_rowcol",    ge::DT_FLOAT,   2.0, 3.0, 1.0, {1,5},    {5,1},    {5,5},   {5,5},    7.0,     1e-5},
        // 3-D mutual broadcast, x3 trailing-dim vector
        {"fp32_bc_3d",        ge::DT_FLOAT,   2.0, 3.0, 1.0, {2,1,4},  {1,3,4},  {4},     {2,3,4},  7.0,     1e-5},
        // fp16 2-D mutual broadcast
        {"fp16_bc_mixed",     ge::DT_FLOAT16, 1.5, 2.0, 0.5, {2,1},    {1,3},    {1},     {2,3},    3.5,     5e-3},
        // every input broadcasts on a different axis -> [4,3,5]
        {"fp32_bc_allaxis",   ge::DT_FLOAT,   2.0, 3.0, 1.0, {4,1,1},  {1,3,1},  {1,1,5}, {4,3,5},  7.0,     1e-5},
        // cross-rank: x2/x3 lower rank than the [3,4,5] output
        {"fp32_bc_crossrank", ge::DT_FLOAT,   2.0, 3.0, 1.0, {3,4,5},  {4,5},    {5},     {3,4,5},  7.0,     1e-5},
        // int32 2-D mutual broadcast
        {"int32_bc_mixed",    ge::DT_INT32,   2.0, 3.0, 5.0, {3,1},    {1,4},    {1},     {3,4},    11.0,    0.5 },
        // fp16 trailing column-vector broadcast {4,1} x {4,6} -> {4,6}
        {"fp16_bc_col",       ge::DT_FLOAT16, 1.5, 2.0, 0.5, {4,1},    {4,6},    {1},     {4,6},    3.5,     5e-3},
        // --- empty tensor scenarios (numel == 0) ---
        // all inputs empty 1-D -> empty output
        {"fp32_empty",        ge::DT_FLOAT,   2.0, 3.0, 1.0, {0},      {0},      {0},     {0},      7.0,     1e-5},
        // empty leading dim carried by x1, x2/x3 broadcast up -> empty [0,4]
        {"fp32_empty_bc",     ge::DT_FLOAT,   2.0, 3.0, 1.0, {0,4},    {1,4},    {1},     {0,4},    7.0,     1e-5},
    };

    int allFail = 0;
    for (const auto& c : cases) {
        if (RunOneCase(c) != SUCCESS) {
            allFail++;
        }
    }

    ge::AscendString errMsg = ge::GEGetErrorMsgV2();
    string errStr(errMsg.GetString());
    if (!errStr.empty()) std::cout << "GE error: " << errStr << std::endl;

    ge::GEFinalize();

    if (allFail != 0) {
        printf("\n%s - FAIL - [XIR]: %d / %zu cases failed\n",
               GetTime().c_str(), allFail, cases.size());
        return FAILED;
    }
    printf("\n%s - PASS - [XIR]: all %zu fused_mul_add cases passed\n",
           GetTime().c_str(), cases.size());
    return SUCCESS;
}
