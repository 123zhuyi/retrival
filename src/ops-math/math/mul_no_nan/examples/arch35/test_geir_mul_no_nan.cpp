/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// End-to-end GE-IR example for MulNoNan on Ascend 950.
//
// Each test case builds a one-op graph, runs it on device, and checks the
// output against a pre-computed expected value. The example exits with
// status 0 only if every case passes.
//
// Semantics: y = (x2 == 0) ? 0 : x1 * x2 (element-wise, broadcast).
// Cases below specifically exercise the special-value contract that
// distinguishes MulNoNan from plain Mul:
//   * x1 = +/- inf, x2 = 0   ->  y = 0    (masks 0 * inf = NaN)
//   * x1 = NaN,     x2 = 0   ->  y = 0    (masks 0 * NaN = NaN)
//   * x1 = NaN,     x2 = 2   ->  y = NaN  (NaN propagates when x2 != 0)
//   * x1 = inf,     x2 = 2   ->  y = inf  (Inf propagates when x2 != 0)
//   * x1 = any,     x2 = -0  ->  y = 0    (-0 == 0 in IEEE-754)

#include <iostream>
#include <fstream>
#include <string.h>
#include <stdint.h>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <limits>
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
#include "../../op_graph/mul_no_nan_proto.h"

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
    if (dt == ge::DT_BF16)    return 2;
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

// IEEE-754 fp32 -> bfloat16 (round-to-nearest-even truncating low 16 bits).
static uint16_t FloatToBf16Bits(float v)
{
    uint32_t x;
    memcpy(&x, &v, sizeof(x));
    // Preserve NaN signaling bit pattern: ensure result stays NaN.
    if (((x >> 23) & 0xff) == 0xff && (x & 0x7fffff) != 0) {
        return (uint16_t)((x >> 16) | 0x40);
    }
    uint32_t rounded = x + 0x7fff + ((x >> 16) & 1);
    return (uint16_t)(rounded >> 16);
}

static float Bf16BitsToFloat(uint16_t b)
{
    uint32_t x = ((uint32_t)b) << 16;
    float f;
    memcpy(&f, &x, sizeof(f));
    return f;
}

// Allocate a tensor and fill with a single value of the given dtype.
// `value` is a double so that callers can pass inf / nan / -0.0 directly.
static int32_t GenConstTensor(
    const vector<int64_t>& shape, DataType dtype, double value, Tensor& tensor, TensorDesc& desc)
{
    desc.SetRealDimCnt(shape.size());
    size_t numel = 1;
    for (auto d : shape) numel *= (size_t)d;
    size_t bytes = numel * GetDataTypeSize(dtype);

    if (dtype == ge::DT_FLOAT) {
        float* p = new (std::nothrow) float[numel];
        float v = (float)value;
        for (size_t i = 0; i < numel; ++i) p[i] = v;
        tensor = Tensor(desc, (uint8_t*)p, bytes);
    } else if (dtype == ge::DT_FLOAT16) {
        uint16_t* p = new (std::nothrow) uint16_t[numel];
        uint16_t v = FloatToHalfBits((float)value);
        for (size_t i = 0; i < numel; ++i) p[i] = v;
        tensor = Tensor(desc, (uint8_t*)p, bytes);
    } else if (dtype == ge::DT_BF16) {
        uint16_t* p = new (std::nothrow) uint16_t[numel];
        uint16_t v = FloatToBf16Bits((float)value);
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

// Build a graph that computes y = MulNoNan(x1, x2) with two constant inputs.
static int BuildMulNoNanGraph(
    Graph& graph, std::vector<ge::Tensor>& inputData, std::vector<Operator>& inputs,
    std::vector<Operator>& outputs, DataType dtype, double v1, double v2,
    const vector<int64_t>& s1, const vector<int64_t>& s2,
    const vector<int64_t>& outShape, const string& caseTag)
{
    auto op = ge::op::MulNoNan("mnn_" + caseTag);

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
    // Output y
    {
        TensorDesc d(ge::Shape(outShape), FORMAT_ND, dtype);
        op.update_output_desc_y(d);
    }
    outputs.push_back(op);
    return SUCCESS;
}

// Expected comparison policy for a single element.
//   kExact: |got - expected| <= absTol (numeric)
//   kIsNaN: got must be NaN regardless of expected
struct ExpectPolicy {
    enum class Mode { kExact, kIsNaN };
    Mode   mode;
    double expected;
    double absTol;
};

struct CaseSpec {
    string                 tag;
    DataType               dtype;
    double                 v1, v2;
    vector<int64_t>        s1, s2, outShape;
    ExpectPolicy           expect;
};

static bool MatchOne(double got, const ExpectPolicy& expect)
{
    if (expect.mode == ExpectPolicy::Mode::kIsNaN) {
        return std::isnan(got);
    }
    if (got == expect.expected) {  // handles inf==inf / -inf==-inf where (got-expected) would be NaN
        return true;
    }
    return std::abs(got - expect.expected) <= expect.absTol;
}

static int RunOneCase(const CaseSpec& spec)
{
    printf("\n%s - INFO - [XIR]: ===== case %s start =====\n", GetTime().c_str(), spec.tag.c_str());
    Graph graph(("g_" + spec.tag).c_str());
    std::vector<ge::Tensor> inputData;
    std::vector<Operator> inputs;
    std::vector<Operator> outputs;
    if (BuildMulNoNanGraph(graph, inputData, inputs, outputs, spec.dtype,
                           spec.v1, spec.v2,
                           spec.s1, spec.s2, spec.outShape, spec.tag) != SUCCESS) {
        printf("BuildMulNoNanGraph failed\n");
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
            } else if (spec.dtype == ge::DT_BF16) {
                got = (double)Bf16BitsToFloat(reinterpret_cast<const uint16_t*>(raw)[j]);
            } else { // INT32
                got = (double)reinterpret_cast<const int32_t*>(raw)[j];
            }
            if (!MatchOne(got, spec.expect)) {
                if (spec.expect.mode == ExpectPolicy::Mode::kIsNaN) {
                    printf("  MISMATCH @%ld: got=%.6f expected=NaN\n", (long)j, got);
                } else {
                    printf("  MISMATCH @%ld: got=%.6f expected=%.6f absTol=%.6f\n",
                           (long)j, got, spec.expect.expected, spec.expect.absTol);
                }
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
    if (spec.expect.mode == ExpectPolicy::Mode::kIsNaN) {
        printf("%s - PASS - case %s (expected=NaN)\n", GetTime().c_str(), spec.tag.c_str());
    } else {
        printf("%s - PASS - case %s (expected=%.4f)\n",
               GetTime().c_str(), spec.tag.c_str(), spec.expect.expected);
    }
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

    const double kInf  = std::numeric_limits<double>::infinity();
    const double kNeg0 = -0.0;
    const double kNaN  = std::numeric_limits<double>::quiet_NaN();

    auto exact = [](double v, double tol) {
        return ExpectPolicy{ExpectPolicy::Mode::kExact, v, tol};
    };
    auto nan = []() {
        return ExpectPolicy{ExpectPolicy::Mode::kIsNaN, 0.0, 0.0};
    };

    std::vector<CaseSpec> cases = {
        // tag                       dtype           v1     v2    s1       s2       out      expect
        // --- fp32 basics + special values ---
        {"fp32_basic",             ge::DT_FLOAT,   2.0,   3.0,  {2,2},  {2,2},  {2,2},  exact(6.0,    1e-5)},
        {"fp32_zero_y",            ge::DT_FLOAT,   5.0,   0.0,  {2,2},  {2,2},  {2,2},  exact(0.0,    1e-5)},
        {"fp32_inf_times_zero",    ge::DT_FLOAT,   kInf,  0.0,  {2,2},  {2,2},  {2,2},  exact(0.0,    1e-5)},
        {"fp32_nan_times_zero",    ge::DT_FLOAT,   kNaN,  0.0,  {2,2},  {2,2},  {2,2},  exact(0.0,    1e-5)},
        {"fp32_normal_nan",        ge::DT_FLOAT,   kNaN,  2.0,  {2,2},  {2,2},  {2,2},  nan()              },
        {"fp32_inf_normal",        ge::DT_FLOAT,   kInf,  2.0,  {2,2},  {2,2},  {2,2},  exact(kInf,   0.0 )},
        // --- fp16 ---
        {"fp16_basic",             ge::DT_FLOAT16, 1.5,   2.0,  {4},    {4},    {4},    exact(3.0,    5e-3)},
        {"fp16_zero_y",            ge::DT_FLOAT16, 6.5e4, 0.0,  {4},    {4},    {4},    exact(0.0,    5e-3)},
        // --- bf16 ---
        {"bf16_basic",             ge::DT_BF16,    1.0,   2.0,  {4},    {4},    {4},    exact(2.0,    2e-2)},
        {"bf16_inf_times_zero",    ge::DT_BF16,    kInf,  0.0,  {4},    {4},    {4},    exact(0.0,    2e-2)},
        // --- int32 ---
        {"int32_basic",            ge::DT_INT32,   3.0,   4.0,  {3},    {3},    {3},    exact(12.0,   0.5 )},
        {"int32_zero_y",           ge::DT_INT32,   7.0,   0.0,  {3},    {3},    {3},    exact(0.0,    0.5 )},
        // --- broadcast ---
        {"fp32_broadcast_zero",    ge::DT_FLOAT,   2.0,   0.0,  {4,1},  {1,4},  {4,4},  exact(0.0,    1e-5)},
        {"fp32_broadcast_mixed",   ge::DT_FLOAT,   2.0,   3.0,  {4,1},  {1,4},  {4,4},  exact(6.0,    1e-5)},
        // x1 scalar broadcast against a 2-D x2
        {"fp32_bc_scalar_x1",      ge::DT_FLOAT,   3.0,   2.0,  {1},    {3,4},  {3,4},  exact(6.0,    1e-5)},
        // x2 scalar zero broadcast must still trigger the zero arm everywhere
        {"fp32_bc_scalar_zero",    ge::DT_FLOAT,   5.0,   0.0,  {3,4},  {1},    {3,4},  exact(0.0,    1e-5)},
        // row-vector x1 broadcast over rows
        {"fp32_bc_row",            ge::DT_FLOAT,   2.0,   4.0,  {1,5},  {3,5},  {3,5},  exact(8.0,    1e-5)},
        // column-vector x1 broadcast over columns
        {"fp32_bc_col",            ge::DT_FLOAT,   2.0,   4.0,  {3,1},  {3,5},  {3,5},  exact(8.0,    1e-5)},
        // 3-D mutual broadcast {2,1,4} x {1,3,4} -> {2,3,4}
        {"fp32_bc_3d",             ge::DT_FLOAT,   2.0,   3.0,  {2,1,4},{1,3,4},{2,3,4},exact(6.0,    1e-5)},
        // fp16 2-D mutual broadcast
        {"fp16_bc_mixed",          ge::DT_FLOAT16, 1.5,   2.0,  {2,1},  {1,3},  {2,3},  exact(3.0,    5e-3)},
        // cross-rank: lower-rank x1 broadcast against higher-rank x2
        {"fp32_bc_cross_rank",     ge::DT_FLOAT,   2.0,   4.0,  {5},    {3,4,5},{3,4,5},exact(8.0,    1e-5)},
        // trailing column-vector broadcast {4,1} x {4,6} -> {4,6}
        {"fp32_bc_col_trailing",   ge::DT_FLOAT,   2.0,   3.0,  {4,1},  {4,6},  {4,6},  exact(6.0,    1e-5)},
        // int32 2-D mutual broadcast
        {"int32_bc_mixed",         ge::DT_INT32,   3.0,   4.0,  {3,1},  {1,4},  {3,4},  exact(12.0,   0.5 )},
        // bf16 2-D mutual broadcast
        {"bf16_bc_mixed",          ge::DT_BF16,    1.0,   2.0,  {2,1},  {1,3},  {2,3},  exact(2.0,    2e-2)},
        // --- negative zero: -0 == 0 should still trigger the zero arm ---
        {"fp32_neg_zero",          ge::DT_FLOAT,   kNaN,  kNeg0,{2,2},  {2,2},  {2,2},  exact(0.0,    1e-5)},
        // --- empty tensor scenarios (numel == 0) ---
        // both inputs empty 1-D -> empty output
        {"fp32_empty",             ge::DT_FLOAT,   2.0,   3.0,  {0},    {0},    {0},    exact(6.0,    1e-5)},
        // empty leading dim carried by x1, x2 broadcast up -> empty [0,4]
        {"fp32_empty_bc",          ge::DT_FLOAT,   2.0,   3.0,  {0,4},  {1,4},  {0,4},  exact(6.0,    1e-5)},
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
    printf("\n%s - PASS - [XIR]: all %zu mul_no_nan cases passed\n",
           GetTime().c_str(), cases.size());
    return SUCCESS;
}
