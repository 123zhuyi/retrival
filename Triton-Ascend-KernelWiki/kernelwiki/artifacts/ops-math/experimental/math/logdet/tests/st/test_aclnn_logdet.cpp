/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <functional>
#include <fstream>
#include <sstream>
#include <string>

#ifndef USE_MOCK_ACLNN
#include "acl/acl.h"
#include "aclnn_logdet.h"
#endif

#define LOG_PRINT(fmt, ...) printf(fmt "\n", ##__VA_ARGS__)

// ============================================================================
// 辅助函数
// ============================================================================

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t size = 1;
    for (auto dim : shape) size *= dim;
    return size;
}

int64_t GetBatchSize(const std::vector<int64_t>& shape) {
    int64_t batch = 1;
    for (size_t i = 0; i + 2 < shape.size(); i++) {
        batch *= shape[i];
    }
    return batch;
}

int64_t GetMatrixSize(const std::vector<int64_t>& shape) {
    return shape[shape.size() - 1];
}

float RandomFloat(float min_val, float max_val) {
    double t = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
    double val = static_cast<double>(min_val) + t * (static_cast<double>(max_val) - static_cast<double>(min_val));
    return static_cast<float>(val);
}

// ============================================================================
// LU 分解（带部分主元选取）
// 返回: 1 (正), -1 (负), 0 (奇异)
// ============================================================================

int LuDecompose(float* A, int n, std::vector<int>& piv) {
    int sign = 1;
    for (int k = 0; k < n; k++) {
        float max_val = std::abs(A[k * n + k]);
        int max_row = k;
        for (int i = k + 1; i < n; i++) {
            float val = std::abs(A[i * n + k]);
            if (val > max_val) {
                max_val = val;
                max_row = i;
            }
        }
        piv[k] = max_row;

        if (max_row != k) {
            sign = -sign;
            for (int j = 0; j < n; j++) {
                std::swap(A[k * n + j], A[max_row * n + j]);
            }
        }

        if (std::abs(A[k * n + k]) < 1e-38f) {
            return 0;
        }

        for (int i = k + 1; i < n; i++) {
            A[i * n + k] /= A[k * n + k];
            for (int j = k + 1; j < n; j++) {
                A[i * n + j] -= A[i * n + k] * A[k * n + j];
            }
        }
    }
    return sign;
}

float ComputeSingleLogdet(const float* matrix, int n) {
    if (n == 0) return 0.0f;

    std::vector<float> A(matrix, matrix + n * n);
    std::vector<int> piv(n);

    int sign = LuDecompose(A.data(), n, piv);

    if (sign == 0) {
        return -std::numeric_limits<float>::infinity();
    }

    float log_abs_det = 0.0f;
    float det_sign = static_cast<float>(sign);

    for (int i = 0; i < n; i++) {
        float diag = A[i * n + i];
        log_abs_det += std::log(std::abs(diag));
        if (diag < 0.0f) det_sign = -det_sign;
    }

    if (det_sign > 0.0f) {
        return log_abs_det;
    } else if (det_sign < 0.0f) {
        return std::numeric_limits<float>::quiet_NaN();
    } else {
        return -std::numeric_limits<float>::infinity();
    }
}

// ============================================================================
// CPU Golden 计算
// ============================================================================

std::vector<float> ComputeGolden(const float* self_data,
                                  const std::vector<int64_t>& self_shape) {
    int64_t n = GetMatrixSize(self_shape);
    int64_t batch_size = GetBatchSize(self_shape);

    std::vector<float> output(batch_size);

    if (batch_size == 0) {
        return output;
    }

    for (int64_t b = 0; b < batch_size; b++) {
        const float* matrix = self_data + b * n * n;
        output[b] = ComputeSingleLogdet(matrix, static_cast<int>(n));
    }

    return output;
}

// ============================================================================
// 精度比对函数
// ============================================================================

bool CompareResults(const float* golden, const float* actual, size_t size,
                    double rtol = 5e-2, double atol = 1e-2) {
    if (golden == nullptr || actual == nullptr) {
        LOG_PRINT("  [ERROR] null pointer input");
        return false;
    }
    if (size == 0) return true;

    int nan_count = 0;
    int inf_count = 0;
    int mismatch = 0;

    for (size_t i = 0; i < size; ++i) {
        float g = golden[i];
        float a = actual[i];

        bool g_nan = std::isnan(g);
        bool a_nan = std::isnan(a);
        bool g_inf = std::isinf(g);
        bool a_inf = std::isinf(a);

        if (g_nan && a_nan) {
            nan_count++;
            continue;
        }
        if (g_inf && a_inf && std::signbit(g) == std::signbit(a)) {
            inf_count++;
            continue;
        }
        if (g_nan != a_nan || g_inf != a_inf) {
            mismatch++;
            if (mismatch <= 5) {
                LOG_PRINT("  不匹配 [%zu]: 期望=%e, 实际=%e", i, g, a);
            }
            continue;
        }

        double diff = std::abs(static_cast<double>(a) - static_cast<double>(g));
        double denom = std::max(std::abs(static_cast<double>(g)), 1e-7);
        if (diff > atol && diff / denom > rtol) {
            mismatch++;
            if (mismatch <= 5) {
                LOG_PRINT("  不匹配 [%zu]: 期望=%e, 实际=%e, diff=%.2e, rel=%.2e",
                          i, g, a, diff, diff / denom);
            }
        }
    }

    if (mismatch == 0) {
        LOG_PRINT("  [PASS] 所有 %zu 个元素一致 (nan=%d, inf=%d)",
                  size, nan_count, inf_count);
        return true;
    } else {
        LOG_PRINT("  [FAIL] 发现 %d 个不匹配 (nan=%d, inf=%d)",
                  mismatch, nan_count, inf_count);
        return false;
    }
}

// ============================================================================
// CPU Golden 自测
// ============================================================================

void TestGoldenCorrectness() {
    LOG_PRINT("\n========================================");
    LOG_PRINT("CPU Golden 正确性自测");
    LOG_PRINT("========================================");

    {
        LOG_PRINT("\n测试 1: 2x2 单位矩阵 → log(det(I)) = 0");
        float I[] = {1.0f, 0.0f, 0.0f, 1.0f};
        std::vector<int64_t> shape = {2, 2};
        auto golden = ComputeGolden(I, shape);
        float expected[] = {0.0f};
        bool pass = CompareResults(expected, golden.data(), 1);
        LOG_PRINT("  结果: %s", pass ? "PASS" : "FAIL");
    }

    {
        LOG_PRINT("\n测试 2: 2x2 全零矩阵 → -inf");
        float Z[] = {0.0f, 0.0f, 0.0f, 0.0f};
        std::vector<int64_t> shape = {2, 2};
        auto golden = ComputeGolden(Z, shape);
        bool pass = std::isinf(golden[0]) && golden[0] < 0;
        LOG_PRINT("  golden=%e, is_neg_inf=%d", golden[0], pass);
        LOG_PRINT("  结果: %s", pass ? "PASS" : "FAIL");
    }

    {
        LOG_PRINT("\n测试 3: 3x3 对角矩阵 diag(2,3,4) → log(24) ≈ 3.178054");
        float D[] = {2.0f, 0.0f, 0.0f, 0.0f, 3.0f, 0.0f, 0.0f, 0.0f, 4.0f};
        std::vector<int64_t> shape = {3, 3};
        auto golden = ComputeGolden(D, shape);
        float expected = std::log(24.0f);
        float diff = std::abs(golden[0] - expected);
        bool pass = diff < 1e-4f;
        LOG_PRINT("  golden=%e, expected=%e, diff=%.2e", golden[0], expected, diff);
        LOG_PRINT("  结果: %s", pass ? "PASS" : "FAIL");
    }

    {
        LOG_PRINT("\n测试 4: 1x1 矩阵 [5.0] → log(5) ≈ 1.609438");
        float S[] = {5.0f};
        std::vector<int64_t> shape = {1, 1};
        auto golden = ComputeGolden(S, shape);
        float expected = std::log(5.0f);
        float diff = std::abs(golden[0] - expected);
        bool pass = diff < 1e-4f;
        LOG_PRINT("  golden=%e, expected=%e, diff=%.2e", golden[0], expected, diff);
        LOG_PRINT("  结果: %s", pass ? "PASS" : "FAIL");
    }

    {
        LOG_PRINT("\n测试 5: batch 2x(2x2) 单位矩阵 → [0, 0]");
        float I2[] = {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f};
        std::vector<int64_t> shape = {2, 2, 2};
        auto golden = ComputeGolden(I2, shape);
        float expected[] = {0.0f, 0.0f};
        bool pass = CompareResults(expected, golden.data(), 2);
        LOG_PRINT("  结果: %s", pass ? "PASS" : "FAIL");
    }

    {
        LOG_PRINT("\n测试 6: 2x2 负行列式矩阵 [[0,1],[1,0]] → NaN (det=-1)");
        float N[] = {0.0f, 1.0f, 1.0f, 0.0f};
        std::vector<int64_t> shape = {2, 2};
        auto golden = ComputeGolden(N, shape);
        bool pass = std::isnan(golden[0]);
        LOG_PRINT("  golden=%e, is_nan=%d", golden[0], pass);
        LOG_PRINT("  结果: %s", pass ? "PASS" : "FAIL");
    }

    LOG_PRINT("\n========================================");
}

// ============================================================================
// 测试用例定义
// ============================================================================

enum ValueRangeType {
    RANGE_RANDOM,
    RANGE_IDENTITY,
    RANGE_POS_ZERO,
    RANGE_NEG_ZERO,
};

struct LogdetTestCase {
    std::string name;
    std::vector<int64_t> self_shape;
    std::vector<int64_t> out_shape;
    ValueRangeType range_type;
    float range_min;
    float range_max;
    bool expect_error;
    bool is_empty;
    bool skip_large;
};

enum class L2ErrorType {
    DTYPE_NOT_SUPPORTED,
    NULL_INPUT,
    NULL_OUTPUT,
    SHAPE_MISMATCH,
};

struct LogdetErrorTestCase {
    std::string name;
    L2ErrorType type;
};

std::vector<LogdetTestCase> GetTestCases() {
    return {
        {"L0_001: rank=2 4x4", {4, 4}, {}, RANGE_RANDOM, 0.001f, 0.01f, false, false, false},
        {"L0_002: rank=3 batch 7x359x359", {7, 359, 359}, {7}, RANGE_RANDOM, 0.0f, 0.001f, false, false, false},
        {"L0_003: rank=4 10x1x7x7", {10, 1, 7, 7}, {10, 1}, RANGE_RANDOM, -1.0f, -0.01f, false, false, false},
        {"L0_004: rank=5 7x3x2x5x5 (-0)", {7, 3, 2, 5, 5}, {7, 3, 2}, RANGE_NEG_ZERO, 0.0f, 0.0f, false, false, false},
        {"L0_005: rank=6 15x2788x2x1x7x7", {15, 2788, 2, 1, 7, 7}, {15, 2788, 2, 1}, RANGE_RANDOM, -10.0f, -2.0f, false, false, false},
        {"L0_006: rank=7 5x29x1x1x4x4x4", {5, 29, 1, 1, 4, 4, 4}, {5, 29, 1, 1, 4}, RANGE_RANDOM, -1.0f, 1.0f, false, false, false},
        {"L0_007: rank=8 2x2x1x1x1x1x3x3", {2, 2, 1, 1, 1, 1, 3, 3}, {2, 2, 1, 1, 1, 1}, RANGE_RANDOM, -3.4028235e+38f, -3.4028235e+38f, false, false, false},
        {"L0_008: rank=9 2x1x1x1x1x1x1x3x3", {2, 1, 1, 1, 1, 1, 1, 3, 3}, {2, 1, 1, 1, 1, 1, 1}, RANGE_RANDOM, -100.0f, 100.0f, false, false, false},
        {"L0_009: rank=10 2x1x1x1x1x1x1x1x3x3", {2, 1, 1, 1, 1, 1, 1, 1, 3, 3}, {2, 1, 1, 1, 1, 1, 1, 1}, RANGE_RANDOM, -3.4028235e+38f, 3.4028235e+38f, false, false, false},
        {"L0_010: rank=11 2x1x1x1x1x1x1x1x1x3x3", {2, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3}, {2, 1, 1, 1, 1, 1, 1, 1, 1}, RANGE_RANDOM, 2.0f, 10.0f, false, false, false},
        {"L0_011: rank=12 2x1x1x1x1x1x1x1x1x1x3x3", {2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3}, {2, 1, 1, 1, 1, 1, 1, 1, 1, 1}, RANGE_RANDOM, 10.0f, 1000.0f, false, false, false},
        {"L0_012: rank=13 2x1x1x1x1x1x1x1x1x1x1x3x3", {2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3}, {2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, RANGE_RANDOM, -0.001f, 0.0f, false, false, false},
        {"L0_013: rank=14 2x1x1x1x1x1x1x1x1x1x1x1x3x3", {2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3}, {2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, RANGE_RANDOM, -0.01f, -0.001f, false, false, false},
        {"L0_014: rank=15 2x1x1x1x1x1x1x1x1x1x1x1x1x3x3", {2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3}, {2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, RANGE_RANDOM, -2.0f, -1.0f, false, false, false},
        {"L0_015: +0 特殊值 255x255", {255, 255}, {}, RANGE_POS_ZERO, 0.0f, 0.0f, false, false, false},
        {"L0_016: rank=16 2x1x1x1x1x1x1x1x1x1x1x1x1x1x3x3", {2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3}, {2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, RANGE_RANDOM, -1000.0f, -10.0f, false, false, false},
        {"L0_017: boundary 2x2 单位矩阵", {2, 2}, {}, RANGE_IDENTITY, 0.0f, 0.0f, false, false, false},
        {"L0_018: boundary 2x2 奇异矩阵(全零)", {2, 2}, {}, RANGE_POS_ZERO, 0.0f, 0.0f, false, false, false},
        {"L0_019: boundary 1x1 方阵", {1, 1}, {}, RANGE_RANDOM, -0.01f, 0.01f, false, false, false},
        {"L0_020: boundary 空Tensor (0,3,3)", {0, 3, 3}, {0}, RANGE_RANDOM, 0.01f, 1.0f, false, true, false},
        {"L0_021: boundary 非方阵(2,3) → 报错", {2, 3}, {}, RANGE_RANDOM, 1.0f, 2.0f, true, false, false},
        {"L0_022: boundary rank=0 标量 → 报错", {}, {}, RANGE_RANDOM, -3.0517578125e-05f, 3.0517578125e-05f, true, false, false},
        {"L0_023: extreme fp16上溢边界 60000.0", {2, 2}, {}, RANGE_RANDOM, 60000.0f, 60000.0f, false, false, false}
    };
}

std::vector<LogdetErrorTestCase> GetL2TestCases() {
    return {
        {"L2_001: dtype_not_supported(float16)", L2ErrorType::DTYPE_NOT_SUPPORTED},
        {"L2_002: null_input", L2ErrorType::NULL_INPUT},
        {"L2_003: null_output", L2ErrorType::NULL_OUTPUT},
        {"L2_004: shape_mismatch(out shape mismatch)", L2ErrorType::SHAPE_MISMATCH},
    };
}

std::vector<LogdetTestCase> GetL1TestCases() {
    return {
        {"L1_001: aclnnLogdet_L1_001", {4, 4}, {}, RANGE_RANDOM, -100.0f, 100.0f, false, false, false},
        {"L1_002: aclnnLogdet_L1_002", {4, 4}, {}, RANGE_RANDOM, -1.0f, 1.0f, false, false, false},
        {"L1_003: aclnnLogdet_L1_003", {4, 4}, {}, RANGE_RANDOM, 2.0f, 10.0f, false, false, false},
        {"L1_004: aclnnLogdet_L1_004", {4, 4}, {}, RANGE_RANDOM, 10.0f, 1000.0f, false, false, false},
        {"L1_005: aclnnLogdet_L1_005", {3, 8, 8}, {3}, RANGE_RANDOM, -100.0f, 100.0f, false, false, false},
        {"L1_006: aclnnLogdet_L1_006", {3, 8, 8}, {3}, RANGE_RANDOM, -1.0f, 1.0f, false, false, false},
        {"L1_007: aclnnLogdet_L1_007", {3, 8, 8}, {3}, RANGE_RANDOM, 2.0f, 10.0f, false, false, false},
        {"L1_008: aclnnLogdet_L1_008", {3, 8, 8}, {3}, RANGE_RANDOM, 10.0f, 1000.0f, false, false, false},
        {"L1_009: aclnnLogdet_L1_009", {2, 3, 6, 6}, {2, 3}, RANGE_RANDOM, -100.0f, 100.0f, false, false, false},
        {"L1_010: aclnnLogdet_L1_010", {2, 3, 6, 6}, {2, 3}, RANGE_RANDOM, -1.0f, 1.0f, false, false, false},
        {"L1_011: aclnnLogdet_L1_011", {2, 3, 6, 6}, {2, 3}, RANGE_RANDOM, 2.0f, 10.0f, false, false, false},
        {"L1_012: aclnnLogdet_L1_012", {2, 3, 6, 6}, {2, 3}, RANGE_RANDOM, 10.0f, 1000.0f, false, false, false},
        {"L1_013: aclnnLogdet_L1_013", {2, 2, 2, 5, 5}, {2, 2, 2}, RANGE_RANDOM, -100.0f, 100.0f, false, false, false},
        {"L1_014: aclnnLogdet_L1_014", {2, 2, 2, 5, 5}, {2, 2, 2}, RANGE_RANDOM, -1.0f, 1.0f, false, false, false},
        {"L1_015: aclnnLogdet_L1_015", {2, 2, 2, 5, 5}, {2, 2, 2}, RANGE_RANDOM, 2.0f, 10.0f, false, false, false},
        {"L1_016: aclnnLogdet_L1_016", {2, 2, 2, 5, 5}, {2, 2, 2}, RANGE_RANDOM, 10.0f, 1000.0f, false, false, false},
        {"L1_017: aclnnLogdet_L1_017", {2, 2, 1, 1, 4, 4}, {2, 2, 1, 1}, RANGE_RANDOM, -100.0f, 100.0f, false, false, false},
        {"L1_018: aclnnLogdet_L1_018", {2, 2, 1, 1, 4, 4}, {2, 2, 1, 1}, RANGE_RANDOM, -1.0f, 1.0f, false, false, false},
        {"L1_019: aclnnLogdet_L1_019", {2, 2, 1, 1, 4, 4}, {2, 2, 1, 1}, RANGE_RANDOM, 2.0f, 10.0f, false, false, false},
        {"L1_020: aclnnLogdet_L1_020", {2, 2, 1, 1, 4, 4}, {2, 2, 1, 1}, RANGE_RANDOM, 10.0f, 1000.0f, false, false, false},
        {"L1_021: aclnnLogdet_L1_021", {2, 1, 1, 1, 1, 3, 3}, {2, 1, 1, 1, 1}, RANGE_RANDOM, -100.0f, 100.0f, false, false, false},
        {"L1_022: aclnnLogdet_L1_022", {2, 1, 1, 1, 1, 3, 3}, {2, 1, 1, 1, 1}, RANGE_RANDOM, -1.0f, 1.0f, false, false, false},
        {"L1_023: aclnnLogdet_L1_023", {2, 1, 1, 1, 1, 3, 3}, {2, 1, 1, 1, 1}, RANGE_RANDOM, 2.0f, 10.0f, false, false, false},
        {"L1_024: aclnnLogdet_L1_024", {2, 1, 1, 1, 1, 3, 3}, {2, 1, 1, 1, 1}, RANGE_RANDOM, 10.0f, 1000.0f, false, false, false},
        {"L1_025: aclnnLogdet_L1_025", {2, 1, 1, 1, 1, 1, 3, 3}, {2, 1, 1, 1, 1, 1}, RANGE_RANDOM, -100.0f, 100.0f, false, false, false},
        {"L1_026: aclnnLogdet_L1_026", {2, 1, 1, 1, 1, 1, 3, 3}, {2, 1, 1, 1, 1, 1}, RANGE_RANDOM, -1.0f, 1.0f, false, false, false},
        {"L1_027: aclnnLogdet_L1_027", {2, 1, 1, 1, 1, 1, 3, 3}, {2, 1, 1, 1, 1, 1}, RANGE_RANDOM, 2.0f, 10.0f, false, false, false},
        {"L1_028: aclnnLogdet_L1_028", {2, 1, 1, 1, 1, 1, 3, 3}, {2, 1, 1, 1, 1, 1}, RANGE_RANDOM, 10.0f, 1000.0f, false, false, false},
        {"L1_029: aclnnLogdet_L1_029", {2, 1, 1, 1, 1, 1, 1, 3, 3}, {2, 1, 1, 1, 1, 1, 1}, RANGE_RANDOM, -100.0f, 100.0f, false, false, false},
        {"L1_030: aclnnLogdet_L1_030", {2, 1, 1, 1, 1, 1, 1, 3, 3}, {2, 1, 1, 1, 1, 1, 1}, RANGE_RANDOM, -1.0f, 1.0f, false, false, false},
        {"L1_031: aclnnLogdet_L1_031", {2, 1, 1, 1, 1, 1, 1, 3, 3}, {2, 1, 1, 1, 1, 1, 1}, RANGE_RANDOM, 2.0f, 10.0f, false, false, false},
        {"L1_032: aclnnLogdet_L1_032", {2, 1, 1, 1, 1, 1, 1, 3, 3}, {2, 1, 1, 1, 1, 1, 1}, RANGE_RANDOM, 10.0f, 1000.0f, false, false, false},
        {"L1_033: aclnnLogdet_L1_033", {2, 1, 1, 1, 1, 1, 1, 1, 3, 3}, {2, 1, 1, 1, 1, 1, 1, 1}, RANGE_RANDOM, -100.0f, 100.0f, false, false, false},
        {"L1_034: aclnnLogdet_L1_034", {2, 1, 1, 1, 1, 1, 1, 1, 3, 3}, {2, 1, 1, 1, 1, 1, 1, 1}, RANGE_RANDOM, -1.0f, 1.0f, false, false, false},
        {"L1_035: aclnnLogdet_L1_035", {2, 1, 1, 1, 1, 1, 1, 1, 3, 3}, {2, 1, 1, 1, 1, 1, 1, 1}, RANGE_RANDOM, 2.0f, 10.0f, false, false, false},
        {"L1_036: aclnnLogdet_L1_036", {2, 1, 1, 1, 1, 1, 1, 1, 3, 3}, {2, 1, 1, 1, 1, 1, 1, 1}, RANGE_RANDOM, 10.0f, 1000.0f, false, false, false},
        {"L1_037: aclnnLogdet_L1_037", {2, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3}, {2, 1, 1, 1, 1, 1, 1, 1, 1}, RANGE_RANDOM, -100.0f, 100.0f, false, false, false},
        {"L1_038: aclnnLogdet_L1_038", {2, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3}, {2, 1, 1, 1, 1, 1, 1, 1, 1}, RANGE_RANDOM, -1.0f, 1.0f, false, false, false},
        {"L1_039: aclnnLogdet_L1_039", {2, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3}, {2, 1, 1, 1, 1, 1, 1, 1, 1}, RANGE_RANDOM, 2.0f, 10.0f, false, false, false},
        {"L1_040: aclnnLogdet_L1_040", {2, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3}, {2, 1, 1, 1, 1, 1, 1, 1, 1}, RANGE_RANDOM, 10.0f, 1000.0f, false, false, false},
        {"L1_041: aclnnLogdet_L1_041", {2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3}, {2, 1, 1, 1, 1, 1, 1, 1, 1, 1}, RANGE_RANDOM, -100.0f, 100.0f, false, false, false},
        {"L1_042: aclnnLogdet_L1_042", {2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3}, {2, 1, 1, 1, 1, 1, 1, 1, 1, 1}, RANGE_RANDOM, -1.0f, 1.0f, false, false, false},
        {"L1_043: aclnnLogdet_L1_043", {2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3}, {2, 1, 1, 1, 1, 1, 1, 1, 1, 1}, RANGE_RANDOM, 2.0f, 10.0f, false, false, false},
        {"L1_044: aclnnLogdet_L1_044", {2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3}, {2, 1, 1, 1, 1, 1, 1, 1, 1, 1}, RANGE_RANDOM, 10.0f, 1000.0f, false, false, false},
        {"L1_045: aclnnLogdet_L1_045", {2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3}, {2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, RANGE_RANDOM, -100.0f, 100.0f, false, false, false},
        {"L1_046: aclnnLogdet_L1_046", {2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3}, {2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, RANGE_RANDOM, -1.0f, 1.0f, false, false, false},
        {"L1_047: aclnnLogdet_L1_047", {2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3}, {2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, RANGE_RANDOM, 2.0f, 10.0f, false, false, false},
        {"L1_048: aclnnLogdet_L1_048", {2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3}, {2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, RANGE_RANDOM, 10.0f, 1000.0f, false, false, false},
        {"L1_049: aclnnLogdet_L1_049", {2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3}, {2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, RANGE_RANDOM, -100.0f, 100.0f, false, false, false},
        {"L1_050: aclnnLogdet_L1_050", {2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3}, {2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, RANGE_RANDOM, -1.0f, 1.0f, false, false, false},
        {"L1_051: aclnnLogdet_L1_051", {2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3}, {2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, RANGE_RANDOM, 2.0f, 10.0f, false, false, false},
        {"L1_052: aclnnLogdet_L1_052", {2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3}, {2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, RANGE_RANDOM, 10.0f, 1000.0f, false, false, false},
        {"L1_053: aclnnLogdet_L1_053", {2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3}, {2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, RANGE_RANDOM, -100.0f, 100.0f, false, false, false},
        {"L1_054: aclnnLogdet_L1_054", {2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3}, {2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, RANGE_RANDOM, -1.0f, 1.0f, false, false, false},
        {"L1_055: aclnnLogdet_L1_055", {2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3}, {2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, RANGE_RANDOM, 2.0f, 10.0f, false, false, false},
        {"L1_056: aclnnLogdet_L1_056", {2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3}, {2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, RANGE_RANDOM, 10.0f, 1000.0f, false, false, false},
        {"L1_057: aclnnLogdet_L1_057", {2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3}, {2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, RANGE_RANDOM, -100.0f, 100.0f, false, false, false},
        {"L1_058: aclnnLogdet_L1_058", {2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3}, {2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, RANGE_RANDOM, -1.0f, 1.0f, false, false, false},
        {"L1_059: aclnnLogdet_L1_059", {2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3}, {2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, RANGE_RANDOM, 2.0f, 10.0f, false, false, false},
        {"L1_060: aclnnLogdet_L1_060", {2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3}, {2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, RANGE_RANDOM, 10.0f, 1000.0f, false, false, false}
    };
}

// ============================================================================
// 输入数据生成
// ============================================================================

std::vector<float> GenerateInputData(const LogdetTestCase& tc) {
    int64_t total_size = GetShapeSize(tc.self_shape);
    if (total_size == 0) return {};

    std::vector<float> data(total_size);

    switch (tc.range_type) {
    case RANGE_IDENTITY: {
        int64_t n = GetMatrixSize(tc.self_shape);
        int64_t batch = GetBatchSize(tc.self_shape);
        std::fill(data.begin(), data.end(), 0.0f);
        for (int64_t b = 0; b < batch; b++) {
            int64_t offset = b * n * n;
            for (int64_t i = 0; i < n; i++) {
                data[offset + i * n + i] = 1.0f;
            }
        }
        break;
    }
    case RANGE_POS_ZERO:
        std::fill(data.begin(), data.end(), 0.0f);
        break;
    case RANGE_NEG_ZERO:
        std::fill(data.begin(), data.end(), -0.0f);
        break;
    case RANGE_RANDOM:
        if (tc.range_min == tc.range_max) {
            std::fill(data.begin(), data.end(), tc.range_min);
        } else {
            for (auto& v : data) {
                v = RandomFloat(tc.range_min, tc.range_max);
            }
        }
        break;
    }

    return data;
}

// ============================================================================
// Mock 模式测试执行
// ============================================================================

#ifdef USE_MOCK_ACLNN

bool RunMockTest(const LogdetTestCase& tc) {
    LOG_PRINT("\n[Mock] %s", tc.name);
    LOG_PRINT("  self_shape: (");
    for (size_t i = 0; i < tc.self_shape.size(); i++) {
        LOG_PRINT("    %ld%s", tc.self_shape[i], i + 1 < tc.self_shape.size() ? "," : "");
    }
    LOG_PRINT("  )");

    if (tc.expect_error) {
        LOG_PRINT("  [INFO] 期望报错用例，Mock 模式跳过 NPU 调用验证");
        LOG_PRINT("  [PASS] 错误用例定义正确 (Mock)");
        return true;
    }

    if (tc.is_empty) {
        LOG_PRINT("  [INFO] 空 Tensor 用例，Mock 模式验证空输出");
        auto input = GenerateInputData(tc);
        LOG_PRINT("  input_size=%zu, 期望 output_size=0", input.size());
        LOG_PRINT("  [PASS] 空 Tensor 用例定义正确 (Mock)");
        return true;
    }

    if (tc.skip_large) {
        LOG_PRINT("  [INFO] 大规模用例 (总元素 > 5M)，跳过 CPU golden 计算");
        LOG_PRINT("  [PASS] 大规模用例定义正确 (Mock)");
        return true;
    }

    auto input = GenerateInputData(tc);
    int64_t n = GetMatrixSize(tc.self_shape);

    if (n > 100 && tc.range_type == RANGE_POS_ZERO) {
        LOG_PRINT("  [INFO] 大矩阵全零 (n=%ld)，直接计算 golden=-inf", n);
        int64_t batch = GetBatchSize(tc.self_shape);
        std::vector<float> golden(batch, -std::numeric_limits<float>::infinity());
        LOG_PRINT("  golden[0]=%e (batch=%ld)", golden[0], batch);
        LOG_PRINT("  [PASS] 大矩阵全零 golden 计算正确 (Mock)");
        return true;
    }

    auto golden = ComputeGolden(input.data(), tc.self_shape);

    LOG_PRINT("  input_size=%zu, golden_size=%zu", input.size(), golden.size());

    if (golden.size() > 0) {
        LOG_PRINT("  golden[0]=%e", golden[0]);
    }

    int nan_count = 0, inf_count = 0, finite_count = 0;
    for (auto v : golden) {
        if (std::isnan(v)) nan_count++;
        else if (std::isinf(v)) inf_count++;
        else finite_count++;
    }
    LOG_PRINT("  golden 统计: finite=%d, inf=%d, nan=%d", finite_count, inf_count, nan_count);

    LOG_PRINT("  [PASS] golden 计算完成 (Mock)");
    return true;
}

#else

// ============================================================================
// Real 模式辅助函数
// ============================================================================

std::vector<int64_t> ComputeStrides(const std::vector<int64_t>& shape) {
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    return strides;
}

template<typename T>
int CreateAclTensor(const std::vector<T>& hostData,
                   const std::vector<int64_t>& shape,
                   void** deviceAddr,
                   aclDataType dataType,
                   aclTensor** tensor) {
    size_t size = GetShapeSize(shape) * sizeof(T);

    if (size == 0) {
        *deviceAddr = nullptr;
        auto strides = ComputeStrides(shape);
        *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(),
                                  0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
        return (*tensor != nullptr) ? ACL_SUCCESS : ACL_ERROR_FAILURE;
    }

    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) return ret;

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) { aclrtFree(*deviceAddr); return ret; }

    auto strides = ComputeStrides(shape);
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(),
                              0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    return ACL_SUCCESS;
}

bool RunRealTest(const LogdetTestCase& tc, aclrtStream stream) {
    LOG_PRINT("\n[Real] %s", tc.name.c_str());

    if (tc.expect_error) {
        LOG_PRINT("  [INFO] 期望报错用例");

        if (tc.self_shape.empty()) {
            LOG_PRINT("  [INFO] rank=0 标量输入，验证报错");
            aclTensor* self_tensor = nullptr;
            void* self_dev = nullptr;
            aclTensor* out_tensor = nullptr;
            void* out_dev = nullptr;

            std::vector<float> dummy(1, 1.0f);
            std::vector<int64_t> scalar_shape = {};
            if (CreateAclTensor(dummy, scalar_shape, &self_dev, ACL_FLOAT, &self_tensor) != ACL_SUCCESS) {
                LOG_PRINT("  [PASS] 标量 tensor 创建失败 (符合预期)");
                return true;
            }
            if (CreateAclTensor(dummy, scalar_shape, &out_dev, ACL_FLOAT, &out_tensor) != ACL_SUCCESS) {
                aclDestroyTensor(self_tensor);
                aclrtFree(self_dev);
                LOG_PRINT("  [PASS] 标量 tensor 创建失败 (符合预期)");
                return true;
            }

            uint64_t workspaceSize = 0;
            aclOpExecutor* executor = nullptr;
            auto ret = aclnnLogdetGetWorkspaceSize(self_tensor, out_tensor, &workspaceSize, &executor);

            aclDestroyTensor(self_tensor);
            aclDestroyTensor(out_tensor);
            aclrtFree(self_dev);
            aclrtFree(out_dev);

            if (ret != ACL_SUCCESS) {
                LOG_PRINT("  [PASS] 返回错误码 %d (符合预期)", ret);
                return true;
            } else {
                LOG_PRINT("  [FAIL] 期望报错但未报错");
                return false;
            }
        }

        if (tc.self_shape.size() >= 2 && tc.self_shape[tc.self_shape.size() - 1] != tc.self_shape[tc.self_shape.size() - 2]) {
            LOG_PRINT("  [INFO] 非方阵输入，验证报错");
            auto input = GenerateInputData(tc);
            void* self_dev = nullptr;
            void* out_dev = nullptr;
            aclTensor* self_tensor = nullptr;
            aclTensor* out_tensor = nullptr;

            std::vector<float> out_data(GetShapeSize(tc.out_shape), 0.0f);

            if (CreateAclTensor(input, tc.self_shape, &self_dev, ACL_FLOAT, &self_tensor) != ACL_SUCCESS) {
                LOG_PRINT("  [FAIL] 创建 self tensor 失败");
                return false;
            }
            if (CreateAclTensor(out_data, tc.out_shape, &out_dev, ACL_FLOAT, &out_tensor) != ACL_SUCCESS) {
                aclDestroyTensor(self_tensor);
                aclrtFree(self_dev);
                LOG_PRINT("  [FAIL] 创建 out tensor 失败");
                return false;
            }

            uint64_t workspaceSize = 0;
            aclOpExecutor* executor = nullptr;
            auto ret = aclnnLogdetGetWorkspaceSize(self_tensor, out_tensor, &workspaceSize, &executor);

            aclDestroyTensor(self_tensor);
            aclDestroyTensor(out_tensor);
            aclrtFree(self_dev);
            aclrtFree(out_dev);

            if (ret != ACL_SUCCESS) {
                LOG_PRINT("  [PASS] 返回错误码 %d (符合预期)", ret);
                return true;
            } else {
                LOG_PRINT("  [FAIL] 期望报错但未报错");
                return false;
            }
        }

        LOG_PRINT("  [PASS] 错误用例 (Mock 验证)");
        return true;
    }

    if (tc.is_empty) {
        LOG_PRINT("  [INFO] 空 Tensor 用例");
        void* self_dev = nullptr;
        void* out_dev = nullptr;
        aclTensor* self_tensor = nullptr;
        aclTensor* out_tensor = nullptr;

        std::vector<float> empty_input;
        std::vector<float> empty_output;

        if (CreateAclTensor(empty_input, tc.self_shape, &self_dev, ACL_FLOAT, &self_tensor) != ACL_SUCCESS) {
            LOG_PRINT("  [FAIL] 创建 self tensor 失败");
            return false;
        }
        if (CreateAclTensor(empty_output, tc.out_shape, &out_dev, ACL_FLOAT, &out_tensor) != ACL_SUCCESS) {
            aclDestroyTensor(self_tensor);
            aclrtFree(self_dev);
            LOG_PRINT("  [FAIL] 创建 out tensor 失败");
            return false;
        }

        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        auto ret = aclnnLogdetGetWorkspaceSize(self_tensor, out_tensor, &workspaceSize, &executor);

        aclDestroyTensor(self_tensor);
        aclDestroyTensor(out_tensor);
        aclrtFree(self_dev);
        aclrtFree(out_dev);

        if (ret == ACL_SUCCESS && workspaceSize == 0) {
            LOG_PRINT("  [PASS] 空 Tensor 返回成功, workspaceSize=0 (符合预期)");
            return true;
        } else {
            LOG_PRINT("  [FAIL] 空 Tensor 处理异常: ret=%d, workspaceSize=%lu", ret, workspaceSize);
            return false;
        }
    }

    if (tc.skip_large) {
        LOG_PRINT("  [INFO] 大规模用例 (总元素 > 5M)，跳过 NPU 调用");
        LOG_PRINT("  [PASS] 大规模用例定义正确 (Real 模式跳过)");
        return true;
    }

    auto input = GenerateInputData(tc);
    int64_t n = GetMatrixSize(tc.self_shape);

    auto golden = ComputeGolden(input.data(), tc.self_shape);
    std::vector<float> npu_output(golden.size(), 0.0f);

    void* self_dev = nullptr;
    void* out_dev = nullptr;
    aclTensor* self_tensor = nullptr;
    aclTensor* out_tensor = nullptr;

    if (CreateAclTensor(input, tc.self_shape, &self_dev, ACL_FLOAT, &self_tensor) != ACL_SUCCESS) {
        LOG_PRINT("  [FAIL] 创建 self tensor 失败");
        return false;
    }
    if (CreateAclTensor(npu_output, tc.out_shape, &out_dev, ACL_FLOAT, &out_tensor) != ACL_SUCCESS) {
        aclDestroyTensor(self_tensor);
        aclrtFree(self_dev);
        LOG_PRINT("  [FAIL] 创建 out tensor 失败");
        return false;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnLogdetGetWorkspaceSize(self_tensor, out_tensor, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  GetWorkspaceSize 失败: %d", ret);
        aclDestroyTensor(self_tensor);
        aclDestroyTensor(out_tensor);
        aclrtFree(self_dev);
        aclrtFree(out_dev);
        return false;
    }

    void* workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("  workspace 分配失败: %d (size=%lu)", ret, workspaceSize);
            aclDestroyTensor(self_tensor);
            aclDestroyTensor(out_tensor);
            aclrtFree(self_dev);
            aclrtFree(out_dev);
            return false;
        }
    }

    ret = aclnnLogdet(workspace, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  aclnnLogdet 失败: %d", ret);
        if (workspace) aclrtFree(workspace);
        aclDestroyTensor(self_tensor);
        aclDestroyTensor(out_tensor);
        aclrtFree(self_dev);
        aclrtFree(out_dev);
        return false;
    }

    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  流同步失败: %d", ret);
        if (workspace) aclrtFree(workspace);
        aclDestroyTensor(self_tensor);
        aclDestroyTensor(out_tensor);
        aclrtFree(self_dev);
        aclrtFree(out_dev);
        return false;
    }

    ret = aclrtMemcpy(npu_output.data(), golden.size() * sizeof(float),
                      out_dev, golden.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  D2H 数据拷贝失败: %d", ret);
        if (workspace) aclrtFree(workspace);
        aclDestroyTensor(self_tensor);
        aclDestroyTensor(out_tensor);
        aclrtFree(self_dev);
        aclrtFree(out_dev);
        return false;
    }

    bool passed = CompareResults(golden.data(), npu_output.data(), golden.size());

    if (workspace) aclrtFree(workspace);
    aclDestroyTensor(self_tensor);
    aclDestroyTensor(out_tensor);
    aclrtFree(self_dev);
    aclrtFree(out_dev);

    return passed;
}

bool RunRealErrorTest(const LogdetErrorTestCase& tc) {
    LOG_PRINT("\n[Real-L2] %s", tc.name.c_str());

    void* self_dev = nullptr;
    void* out_dev = nullptr;
    aclTensor* self_tensor = nullptr;
    aclTensor* out_tensor = nullptr;
    aclnnStatus ret = 0;

    std::vector<int64_t> self_shape = {2, 2};
    std::vector<int64_t> out_shape = {};
    std::vector<int64_t> wrong_out_shape = {3};

    std::vector<float> self_data = {1.0f, 0.0f, 0.0f, 1.0f};
    std::vector<float> scalar_out = {0.0f};

    auto cleanup = [&]() {
        if (self_tensor != nullptr) {
            aclDestroyTensor(self_tensor);
        }
        if (out_tensor != nullptr) {
            aclDestroyTensor(out_tensor);
        }
        if (self_dev != nullptr) {
            aclrtFree(self_dev);
        }
        if (out_dev != nullptr) {
            aclrtFree(out_dev);
        }
    };

    switch (tc.type) {
        case L2ErrorType::DTYPE_NOT_SUPPORTED: {
            std::vector<aclFloat16> self_fp16 = {
                aclFloat16(1.0f), aclFloat16(0.0f), aclFloat16(0.0f), aclFloat16(1.0f)};
            std::vector<aclFloat16> out_fp16 = {aclFloat16(0.0f)};
            if (CreateAclTensor(self_fp16, self_shape, &self_dev, ACL_FLOAT16, &self_tensor) != ACL_SUCCESS) {
                LOG_PRINT("  [FAIL] 创建 float16 self tensor 失败");
                return false;
            }
            if (CreateAclTensor(out_fp16, out_shape, &out_dev, ACL_FLOAT16, &out_tensor) != ACL_SUCCESS) {
                cleanup();
                LOG_PRINT("  [FAIL] 创建 float16 out tensor 失败");
                return false;
            }
            break;
        }
        case L2ErrorType::NULL_INPUT: {
            if (CreateAclTensor(scalar_out, out_shape, &out_dev, ACL_FLOAT, &out_tensor) != ACL_SUCCESS) {
                LOG_PRINT("  [FAIL] 创建 out tensor 失败");
                return false;
            }
            break;
        }
        case L2ErrorType::NULL_OUTPUT: {
            if (CreateAclTensor(self_data, self_shape, &self_dev, ACL_FLOAT, &self_tensor) != ACL_SUCCESS) {
                LOG_PRINT("  [FAIL] 创建 self tensor 失败");
                return false;
            }
            break;
        }
        case L2ErrorType::SHAPE_MISMATCH: {
            if (CreateAclTensor(self_data, self_shape, &self_dev, ACL_FLOAT, &self_tensor) != ACL_SUCCESS) {
                LOG_PRINT("  [FAIL] 创建 self tensor 失败");
                return false;
            }
            std::vector<float> wrong_out(3, 0.0f);
            if (CreateAclTensor(wrong_out, wrong_out_shape, &out_dev, ACL_FLOAT, &out_tensor) != ACL_SUCCESS) {
                cleanup();
                LOG_PRINT("  [FAIL] 创建错误 shape 的 out tensor 失败");
                return false;
            }
            break;
        }
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnLogdetGetWorkspaceSize(
        tc.type == L2ErrorType::NULL_INPUT ? nullptr : self_tensor,
        tc.type == L2ErrorType::NULL_OUTPUT ? nullptr : out_tensor,
        &workspaceSize,
        &executor);

    cleanup();

    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  [PASS] 返回错误码 %d (符合预期)", ret);
        return true;
    }

    LOG_PRINT("  [FAIL] 期望报错但未报错, ret=%d, workspaceSize=%lu", ret, workspaceSize);
    return false;
}

#endif

// ============================================================================
// 主函数
// ============================================================================

int main(int argc, char* argv[]) {
    srand(42);

    LOG_PRINT("\n========================================");
    LOG_PRINT("logdet 算子 ST 测试");
    LOG_PRINT("========================================");

#ifdef USE_MOCK_ACLNN
    LOG_PRINT("模式: Mock (CPU golden)");
#else
    LOG_PRINT("模式: Real (NPU)");
#endif

    TestGoldenCorrectness();

    int passed = 0, failed = 0;

#ifndef USE_MOCK_ACLNN
    int32_t deviceId = 0;
    aclrtStream stream;

    auto initRet = aclInit(nullptr);
    if (initRet != ACL_SUCCESS) {
        LOG_PRINT("[FATAL] aclInit 失败: %d", initRet);
        return 1;
    }
    initRet = aclrtSetDevice(deviceId);
    if (initRet != ACL_SUCCESS) {
        LOG_PRINT("[FATAL] aclrtSetDevice(%d) 失败: %d", deviceId, initRet);
        aclFinalize();
        return 1;
    }
    initRet = aclrtCreateStream(&stream);
    if (initRet != ACL_SUCCESS) {
        LOG_PRINT("[FATAL] aclrtCreateStream 失败: %d", initRet);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return 1;
    }
#endif

    LOG_PRINT("\n执行 L0 测试用例 (%zu 条)...", GetTestCases().size());

    auto test_cases = GetTestCases();

    for (const auto& tc : test_cases) {
#ifdef USE_MOCK_ACLNN
        if (RunMockTest(tc)) passed++; else failed++;
#else
        if (RunRealTest(tc, stream)) passed++; else failed++;
#endif
    }

    LOG_PRINT("\n执行 L1 测试用例 (%zu 条)...", GetL1TestCases().size());

    auto l1_test_cases = GetL1TestCases();

    for (const auto& tc : l1_test_cases) {
#ifdef USE_MOCK_ACLNN
        if (RunMockTest(tc)) passed++; else failed++;
#else
        if (RunRealTest(tc, stream)) passed++; else failed++;
#endif
    }

#ifndef USE_MOCK_ACLNN
    LOG_PRINT("\n执行 L2 测试用例 (%zu 条)...", GetL2TestCases().size());
    auto l2_test_cases = GetL2TestCases();
    for (const auto& tc : l2_test_cases) {
        if (RunRealErrorTest(tc)) passed++; else failed++;
    }
#endif

#ifndef USE_MOCK_ACLNN
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
#endif

    LOG_PRINT("\n========================================");
    LOG_PRINT("测试报告");
    LOG_PRINT("========================================");
    LOG_PRINT("总计: %d", passed + failed);
    LOG_PRINT("通过: %d", passed);
    LOG_PRINT("失败: %d", failed);
    LOG_PRINT("========================================\n");

    return failed == 0 ? 0 : 1;
}
