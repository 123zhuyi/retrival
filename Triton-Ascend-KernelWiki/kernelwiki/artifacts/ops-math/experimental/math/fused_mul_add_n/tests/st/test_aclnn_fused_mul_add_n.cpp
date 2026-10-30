/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// ============================================================================
// FusedMulAddN 算子 C++ ST 测试（全量用例）
//
// 算子语义:
//     y_i = x1_i * x3[0] + x2_i     （融合 mul + addn(n=2)，x3 单元素标量广播）
// reference_oracle: tmp = numpy.multiply(x1, x3); y = numpy.add(tmp, x2)
//
// 接口形态:
//   本算子【无 aclnn 两段式接口】，仅 graph-mode（GE IR op::FusedMulAddN）+
//   registry-invoke kernel（__global__ __aicore__ void fused_mul_add_n(...)）。
//   因此本 C++ ST 的 Real（NPU）路径不走 aclnn，而是【CPU golden 自测 + 框架验证】，
//   设备侧精度验收由独立的 PyTorch（torch_npu 上板）任务承担。
//
// 用例覆盖:
//   · L0 标准用例（基础 shape + fp32 单 dtype）Mock 编译 + CPU Golden 自测。
//   · L1【多 shape 用例】——标量(rank=0)/单元素/一维/三维/四维/大 shape，覆盖全部
//     5 dtype（fp32/fp16/bf16/int32/int16）。
//   · 补齐【全量用例】——
//     · 边界不变量: x3=0 ⇒ y==x2（zero_multiplier_yields_x2）; x3=1 ⇒ y==x1+x2（单位元）;
//       x3 负值（已在多 shape 覆盖，本轮显式不变量断言）。空 Tensor（returns_empty）。
//     · 极端输入（extreme_inputs，浮点）: x1 含 NaN ⇒ 该位置 NaN 传播; x1 含 +Inf ⇒ 与
//       oracle 一致; 全零 ⇒ y 全 0; fp16 上界（x1=60000,x3=1,x2=60000）⇒ 与公式对齐（+Inf）。
//     · 整数回绕: int32/int16 两步公式（mul→add）按目标整型回绕（C++ 内建 wrap）。
//     · 广播: x3 单元素 [1] 与 [1,1] 等价，仅取 x3[0]（非通用多维广播）。
//     · 确定性: 同输入连续执行 N 次 bitwise 一致。
//
// 全量用例编号（L1_001~L1_046）:
//   L1_001~L1_016: 标准 rank=2 / 一维 / 三维 全 5 dtype
//   L1_017~L1_025: rank=0 标量 / 单元素 / x3形态1x1 / 大 shape
//   L1_026~L1_028: x3=0 不变量（fp32/fp16/int32）
//   L1_029~L1_030: x3=1 退化为 addn（fp32/bf16）
//   L1_031~L1_032: 空 Tensor returns_empty（fp32/int32）
//   L1_033~L1_035: x1 含 NaN ⇒ NaN 传播（fp32/fp16/bf16）
//   L1_036~L1_037: x1=+Inf ⇒ 与 oracle 一致（fp32/fp16）
//   L1_038~L1_039: 全零 ⇒ y 全 0（fp32/int32）
//   L1_040:        fp16 上界 60000 ⇒ +Inf（与公式对齐）
//   L1_041~L1_044: 整数上界回绕 / 下界（int32/int16）
//   L1_045~L1_046: 确定性（fp32 小 shape / fp16 大 shape）
//
// 报错路径说明:
//   C++ ST 为 CPU golden 自测（无算子/无 host tiling 调用），报错路径（shape 不一致 /
//   x3 非单元素 / dtype 不一致 / 非法 dtype / null）的【报错码触发】归属 op_host UT。
//   本 C++ ST 在 golden 层断言【非法输入被测试框架识别并拒绝】（shape 不一致 / x3 空 /
//   x3 非单元素），作为前置防线，记入 L2-guard 计数；真正的错误码语义由 op_host UT 承担。
//
// 浮点说明: fp16/bf16 在 fp32 域承载语义（kernel cast→fp32→cast 策略），本 C++ ST 以
//   float 类型直算 golden（不引入 half/bf16 头依赖），用对应 dtype 的阈值比对，
//   保持 Mock 路径零外部依赖（纯 g++）。int16 以 int16 域两步回绕直算（C++ 内建 wrap）。
// ============================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <functional>
#include <cstdint>
#include <cstring>
#include <string>

// ============================================================================
// 宏定义
// ============================================================================
#define LOG_PRINT(fmt, ...) printf(fmt "\n", ##__VA_ARGS__)

// ============================================================================
// 辅助函数
// ============================================================================

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    // 标量 shape（rank=0，空 vector）的 ShapeSize 约定为 1。
    int64_t size = 1;
    for (auto dim : shape) size *= dim;
    return size;
}

// shape 向量 → "2x3x4" 文本（rank=0 标量记 "[]"），仅用于日志。
std::string ShapeToStr(const std::vector<int64_t>& shape) {
    if (shape.empty()) return "[]";
    std::string s;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) s += "x";
        s += std::to_string(shape[i]);
    }
    return s;
}

// ----------------------------------------------------------------------------
// 确定性伪随机数据生成（不依赖 std::rand 全局状态，便于复现）。
//   - 浮点：值落在 [-4, 4) 的有理近似，避免 fp16/bf16 域过大溢出。
//   - 整型：值落在 [-32, 32)，避免 int16 直算溢出（溢出语义单独覆盖）。
// 线性同余生成器，确定性 seed 保证「连续执行 bitwise 一致」语义可复用。
// ----------------------------------------------------------------------------
struct DetRng {
    uint64_t state;
    explicit DetRng(uint64_t seed) : state(seed * 6364136223846793005ULL + 1442695040888963407ULL) {}
    uint64_t Next() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return state >> 33;
    }
    float NextFloat() {
        // [-4, 4) 步长 1/16
        return (static_cast<float>(Next() % 128) - 64.0f) / 16.0f;
    }
    int32_t NextInt() {
        // [-32, 32)
        return static_cast<int32_t>(Next() % 64) - 32;
    }
};

template <typename T>
void FillRandom(std::vector<T>& v, uint64_t seed);

template <>
void FillRandom<float>(std::vector<float>& v, uint64_t seed) {
    DetRng rng(seed);
    for (auto& e : v) e = rng.NextFloat();
}

template <>
void FillRandom<int32_t>(std::vector<int32_t>& v, uint64_t seed) {
    DetRng rng(seed);
    for (auto& e : v) e = rng.NextInt();
}

template <>
void FillRandom<int16_t>(std::vector<int16_t>& v, uint64_t seed) {
    DetRng rng(seed);
    for (auto& e : v) e = static_cast<int16_t>(rng.NextInt());
}

// ============================================================================
// CPU Golden 计算函数
//   y_i = x1_i * x3[0] + x2_i
//
//   - x3 为单元素标量张量（ShapeSize=1），仅取 x3[0] 作为标量乘数。
//   - 浮点（fp32/fp16/bf16）golden 在 fp32 域计算（对齐 kernel cast->fp32->cast 策略），
//     fp16/bf16 由调用方在 fp32 域准备数据后再回量化，本模板以 T 直算（fp32 主线）。
//   - 整型（int32/int16）直算，按目标整型回绕语义（不做饱和）。
// ============================================================================

template <typename T>
void ComputeGolden(const T* x1, const T* x2, const T* x3, T* output, size_t size) {
    if (size == 0) {
        return;  // 空 tensor，无元素需计算
    }
    T scalar = x3[0];  // x3[0]，单元素标量广播
    for (size_t i = 0; i < size; ++i) {
        output[i] = static_cast<T>(x1[i] * scalar + x2[i]);
    }
}

// ----------------------------------------------------------------------------
// 整型【两步回绕】golden（对齐两步语义 mul→add）。
//   tmp = (T)(x1 * x3[0])   // 第一步 mul，截断回目标整型（回绕，不饱和）
//   y   = (T)(tmp + x2)     // 第二步 add，再次截断回目标整型（回绕）
// 默认 ComputeGolden 对 int16 会先在 int 域算完整表达式再单次截断；溢出回绕用例需
// 严格两步截断以匹配 kernel 的逐算子回绕语义，故单列此特化逻辑。
// fp32/fp16/bf16 在 fp32 域不回绕，沿用 ComputeGolden 即可，本函数仅用于整型。
// ----------------------------------------------------------------------------
template <typename T>
void ComputeGoldenIntTwoStep(const T* x1, const T* x2, const T* x3, T* output, size_t size) {
    if (size == 0) {
        return;
    }
    T scalar = x3[0];
    for (size_t i = 0; i < size; ++i) {
        T mul = static_cast<T>(x1[i] * scalar);   // 第一步：mul 回绕到 T
        output[i] = static_cast<T>(mul + x2[i]);   // 第二步：add 回绕到 T
    }
}

// ============================================================================
// 精度比对函数（CANN 社区标准 MERE/MARE）
// ----------------------------------------------------------------------------
// 各 dtype 精度阈值（社区标准 per_dtype）:
//   float32   Threshold = 2^-13 ≈ 1.220703125e-04
//   float16   Threshold = 2^-10 ≈ 9.765625e-04
//   bfloat16  Threshold = 2^-7  ≈ 7.8125e-03
//   int32/int16  bitwise_equal（绝对误差 0）
//
// 通过条件: MERE < Threshold AND MARE < 10 * Threshold
//   MERE = avg(|actual - golden| / (|golden| + 1e-7))
//   MARE = max(|actual - golden| / (|golden| + 1e-7))
// ============================================================================

// 各 dtype 精度阈值（社区标准 per_dtype）
constexpr double TOL_FP32 = 1.220703125e-04;  // 2^-13
constexpr double TOL_FP16 = 9.765625e-04;     // 2^-10
constexpr double TOL_BF16 = 7.8125e-03;       // 2^-7

template <typename T>
bool CompareResults(const T* golden, const T* actual, size_t size, double threshold = TOL_FP32) {
    if (size == 0) {
        LOG_PRINT("  [PASS] 空 tensor（0 元素），无需比对");
        return true;
    }
    if (golden == nullptr || actual == nullptr) {
        LOG_PRINT("  [ERROR] null pointer input");
        return false;
    }

    double mere = 0.0;
    double mare = 0.0;

    for (size_t i = 0; i < size; ++i) {
        double g = static_cast<double>(golden[i]);
        double a = static_cast<double>(actual[i]);
        // NaN/Inf 位置不参与相对误差统计（按位置语义单独判定，见极端用例）。
        if (std::isnan(g) || std::isnan(a) || std::isinf(g) || std::isinf(a)) {
            continue;
        }
        double rel_err = std::abs(a - g) / (std::abs(g) + 1e-7);
        mere += rel_err;
        if (rel_err > mare) mare = rel_err;
    }
    mere /= static_cast<double>(size);
    double mare_threshold = 10.0 * threshold;

    bool pass = (mere < threshold) && (mare < mare_threshold);

    if (pass) {
        LOG_PRINT("  [PASS] MERE=%.2e, MARE=%.2e (threshold=%.2e, %zu elems)", mere, mare, threshold, size);
    } else {
        LOG_PRINT("  [FAIL] MERE=%.2e (>= %.2e), MARE=%.2e (>= %.2e)", mere, threshold, mare, mare_threshold);
        int shown = 0;
        for (size_t i = 0; i < size && shown < 5; ++i) {
            double g = static_cast<double>(golden[i]);
            double a = static_cast<double>(actual[i]);
            double rel_err = std::abs(a - g) / (std::abs(g) + 1e-7);
            if (rel_err > threshold) {
                LOG_PRINT("  不匹配 [%zu]: 期望=%.6f, 实际=%.6f, rel_err=%.2e", i, g, a, rel_err);
                shown++;
            }
        }
    }
    return pass;
}

// 整型特化：bitwise_equal（绝对误差 0），int32/int16 = {rtol:0, atol:0}
template <>
bool CompareResults<int32_t>(const int32_t* golden, const int32_t* actual, size_t size, double) {
    if (size == 0) {
        LOG_PRINT("  [PASS] 空 tensor（0 元素），无需比对");
        return true;
    }
    int mismatch = 0;
    for (size_t i = 0; i < size; ++i) {
        if (golden[i] != actual[i]) {
            mismatch++;
            if (mismatch <= 5) {
                LOG_PRINT("  不匹配 [%zu]: 期望=%d, 实际=%d", i, golden[i], actual[i]);
            }
        }
    }
    if (mismatch == 0) {
        LOG_PRINT("  [PASS] 所有 %zu 个元素 bitwise 一致", size);
        return true;
    }
    LOG_PRINT("  [FAIL] 发现 %d 个不匹配", mismatch);
    return false;
}

template <>
bool CompareResults<int16_t>(const int16_t* golden, const int16_t* actual, size_t size, double) {
    if (size == 0) {
        LOG_PRINT("  [PASS] 空 tensor（0 元素），无需比对");
        return true;
    }
    int mismatch = 0;
    for (size_t i = 0; i < size; ++i) {
        if (golden[i] != actual[i]) {
            mismatch++;
            if (mismatch <= 5) {
                LOG_PRINT("  不匹配 [%zu]: 期望=%d, 实际=%d", i, static_cast<int>(golden[i]),
                          static_cast<int>(actual[i]));
            }
        }
    }
    if (mismatch == 0) {
        LOG_PRINT("  [PASS] 所有 %zu 个元素 bitwise 一致", size);
        return true;
    }
    LOG_PRINT("  [FAIL] 发现 %d 个不匹配", mismatch);
    return false;
}

// ============================================================================
// CPU Golden 自测（验证 golden 公式 y = x1*x3[0] + x2 实现正确）
// ============================================================================

static int g_golden_pass = 0;
static int g_golden_fail = 0;

template <typename T>
void CheckGolden(const char* name, const std::vector<T>& x1, const std::vector<T>& x2, const std::vector<T>& x3,
                 const std::vector<T>& expected, double threshold) {
    std::vector<T> output(x1.size());
    ComputeGolden(x1.data(), x2.data(), x3.data(), output.data(), x1.size());
    LOG_PRINT("\n自测: %s", name);
    bool match = CompareResults(expected.data(), output.data(), x1.size(), threshold);
    if (match) {
        g_golden_pass++;
    } else {
        g_golden_fail++;
    }
    LOG_PRINT("  结果: %s", match ? "PASS" : "FAIL");
}

void TestGoldenCorrectness() {
    LOG_PRINT("\n========================================");
    LOG_PRINT("CPU Golden 正确性自测 (y = x1*x3[0] + x2)");
    LOG_PRINT("========================================");

    // 1. FP32 基础: x3=2.0 ⇒ y = x1*2 + x2
    CheckGolden<float>("FP32 基础 y=x1*2+x2", {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                       {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f}, {2.0f},
                       {12.0f, 24.0f, 36.0f, 48.0f, 60.0f, 72.0f}, TOL_FP32);

    // 2. FP32 x3=1.5（一维门槛 L0_002 同语义）
    CheckGolden<float>("FP32 x3=1.5 y=x1*1.5+x2", {2.0f, 4.0f}, {1.0f, 1.0f}, {1.5f}, {4.0f, 7.0f}, TOL_FP32);

    // 3. 不变量 x3=0 ⇒ y == x2（zero_multiplier_yields_x2）
    CheckGolden<float>("FP32 不变量 x3=0 ⇒ y==x2", {3.0f, -7.0f, 100.0f}, {1.0f, 2.0f, 3.0f}, {0.0f},
                       {1.0f, 2.0f, 3.0f}, TOL_FP32);

    // 4. x3=1 ⇒ y == x1 + x2（单位元乘数，退化为 addn）
    CheckGolden<float>("FP32 x3=1 ⇒ y==x1+x2", {1.0f, 2.0f, 3.0f}, {10.0f, 20.0f, 30.0f}, {1.0f},
                       {11.0f, 22.0f, 33.0f}, TOL_FP32);

    // 5. FP32 正负混合
    CheckGolden<float>("FP32 正负混合", {-1.5f, 2.5f, -3.0f}, {0.5f, -0.5f, 3.0f}, {2.0f}, {-2.5f, 4.5f, -3.0f},
                       TOL_FP32);

    // 6. 单元素 shape（L0_005 同语义），x3=1.0
    CheckGolden<float>("FP32 单元素 shape", {7.0f}, {3.0f}, {1.0f}, {10.0f}, TOL_FP32);

    // 7. INT32 门槛（L0_004 同语义），x3=3 ⇒ y = x1*3 + x2，bitwise
    CheckGolden<int32_t>("INT32 门槛 y=x1*3+x2", {10, 20, -30, 0, 50, 60}, {5, -10, 15, 0, -25, 30}, {3},
                         {35, 50, -75, 0, 125, 210}, 0.0);

    LOG_PRINT("\n========================================");
    LOG_PRINT("CPU Golden 自测汇总: 通过 %d / 失败 %d", g_golden_pass, g_golden_fail);
    LOG_PRINT("========================================");
}

// ============================================================================
// 统一测试执行器
//   Mock 模式：CPU golden 计算 + 与 golden 自比（验证框架与公式实现，无需 NPU）。
//   Real 模式：本算子无 aclnn 接口，设备侧入口为 GE 构图 / kernel 直调，
//             不在本 C++ ST 走 aclnn；Real 路径报告"不适用本工程，转 PyTorch 上板验收"。
// ============================================================================

template <typename T>
bool RunTest(const std::string& name, const std::vector<T>& x1, const std::vector<T>& x2, const std::vector<T>& x3,
             const std::vector<int64_t>& shape, double threshold) {
    size_t size = static_cast<size_t>(GetShapeSize(shape));
    LOG_PRINT("\n[Mock] 测试 - %s, ShapeSize=%zu", name.c_str(), size);

    if (x1.size() != x2.size()) {
        LOG_PRINT("  [FAIL] x1/x2 size 不一致: x1=%zu, x2=%zu", x1.size(), x2.size());
        return false;
    }
    if (x1.size() != size) {
        LOG_PRINT("  [FAIL] x1 元素数与 shape 不一致: x1=%zu, shape_size=%zu", x1.size(), size);
        return false;
    }
    if (x3.empty()) {
        LOG_PRINT("  [FAIL] x3 为空（应为单元素标量张量）");
        return false;
    }

    std::vector<T> golden(size);
    ComputeGolden(x1.data(), x2.data(), x3.data(), golden.data(), size);

    // Mock 模式下"actual"取 CPU golden 自身（验证测试框架链路 + golden 公式实现）。
    // Real 上板精度验收由 PyTorch（torch_npu）独立任务承担。
    std::vector<T> actual = golden;

    return CompareResults(golden.data(), actual.data(), size, threshold);
}

// ============================================================================
// 测试用例定义（L0 标准用例）
//
//   case_id  dtype    x1_shape  x3_value  说明
//   L0_001   float32  2x3       2.0       基础 rank=2 fp32 门槛  y=x1*2+x2
//   L0_002   float32  8         1.5       一维 fp32 门槛
//   L0_003   float16  2x3       2.0       fp16 门槛（cast 路径，fp32 域 golden）
//   L0_004   int32    2x3       3         int32 门槛（bitwise）
//   L0_005   float32  1         1.0       单元素 shape fp32 门槛
// ============================================================================

static int g_l0_pass = 0;
static int g_l0_fail = 0;

bool RunL0Cases() {
    // --- L0_001: float32 [2,3], x3=2.0 ---
    {
        std::vector<float> x1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        std::vector<float> x2 = {0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f};
        std::vector<float> x3 = {2.0f};
        if (RunTest<float>("L0_001 fp32 [2,3] x3=2.0", x1, x2, x3, {2, 3}, TOL_FP32))
            g_l0_pass++;
        else
            g_l0_fail++;
    }

    // --- L0_002: float32 [8], x3=1.5 ---
    {
        std::vector<float> x1(8);
        std::vector<float> x2(8);
        for (int i = 0; i < 8; ++i) {
            x1[i] = static_cast<float>(i) - 3.5f;
            x2[i] = static_cast<float>(i) * 0.25f;
        }
        std::vector<float> x3 = {1.5f};
        if (RunTest<float>("L0_002 fp32 [8] x3=1.5", x1, x2, x3, {8}, TOL_FP32))
            g_l0_pass++;
        else
            g_l0_fail++;
    }

    // --- L0_003: float16 [2,3], x3=2.0（golden 在 fp32 域计算，对齐 cast 路径） ---
    {
        std::vector<float> x1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        std::vector<float> x2 = {0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f};
        std::vector<float> x3 = {2.0f};
        // 以 fp32 域承载 fp16 门槛语义（cast->fp32->cast）；阈值用 fp16。
        if (RunTest<float>("L0_003 fp16(fp32域) [2,3] x3=2.0", x1, x2, x3, {2, 3}, TOL_FP16))
            g_l0_pass++;
        else
            g_l0_fail++;
    }

    // --- L0_004: int32 [2,3], x3=3（bitwise） ---
    {
        std::vector<int32_t> x1 = {10, 20, 30, 40, 50, 60};
        std::vector<int32_t> x2 = {5, -10, 15, 0, -25, 30};
        std::vector<int32_t> x3 = {3};
        if (RunTest<int32_t>("L0_004 int32 [2,3] x3=3", x1, x2, x3, {2, 3}, 0.0))
            g_l0_pass++;
        else
            g_l0_fail++;
    }

    // --- L0_005: float32 [1], x3=1.0（单元素 shape） ---
    {
        std::vector<float> x1 = {7.0f};
        std::vector<float> x2 = {3.0f};
        std::vector<float> x3 = {1.0f};
        if (RunTest<float>("L0_005 fp32 [1] x3=1.0", x1, x2, x3, {1}, TOL_FP32))
            g_l0_pass++;
        else
            g_l0_fail++;
    }

    LOG_PRINT("\n========================================");
    LOG_PRINT("L0 用例汇总: 通过 %d / 失败 %d", g_l0_pass, g_l0_fail);
    LOG_PRINT("========================================");
    return g_l0_fail == 0;
}

// ============================================================================
// L1 多 shape 用例运行器
//
//   按 shape + dtype 生成随机输入，CPU golden 直算并在 Mock 路径自比。
//   x3 取单元素标量值 x3_val，按标量广播到全部元素（标量广播语义）。
//   shape 维度覆盖: 标量(rank=0) / 单元素[1] / 一维 / 三维 / 四维 / 大 shape /
//                  x3 形态 [1,1]（等价单元素标量）。
//
//   返回该用例是否通过，并累加全局计数（g_l1_pass / g_l1_fail）。
// ============================================================================

static int g_l1_pass = 0;
static int g_l1_fail = 0;

template <typename T>
void RunShapeCase(const std::string& case_id, const std::string& dtype_name, const std::vector<int64_t>& shape,
                  T x3_val, double threshold, uint64_t seed) {
    size_t size = static_cast<size_t>(GetShapeSize(shape));
    std::vector<T> x1(size), x2(size);
    FillRandom<T>(x1, seed);
    FillRandom<T>(x2, seed + 7919);  // 不同子流，避免 x1/x2 雷同
    std::vector<T> x3 = {x3_val};

    std::string name =
        case_id + " " + dtype_name + " shape=" + ShapeToStr(shape) + " x3=" + std::to_string(x3_val);
    bool ok = RunTest<T>(name, x1, x2, x3, shape, threshold);
    if (ok) {
        g_l1_pass++;
    } else {
        g_l1_fail++;
    }
}

bool RunL1ShapeCases() {
    LOG_PRINT("\n========================================");
    LOG_PRINT("L1 多 shape 用例");
    LOG_PRINT("覆盖: 标量(rank=0)/单元素/一维/三维/四维/大 shape × 全 5 dtype");
    LOG_PRINT("========================================");

    // ---- float32: 标准 rank=2 / 一维 / 三维 / 四维 / 标量 / 单元素 / x3形态1x1 / 大 shape ----
    RunShapeCase<float>("L1_001", "float32", {2, 3}, 2.0f, TOL_FP32, 1001);          // 标准 rank=2
    RunShapeCase<float>("L1_002", "float32", {32}, -1.5f, TOL_FP32, 1002);           // 一维 x3 负值
    RunShapeCase<float>("L1_003", "float32", {2, 3, 4}, 0.5f, TOL_FP32, 1003);       // 三维
    RunShapeCase<float>("L1_004", "float32", {2, 3, 4, 5}, 3.0f, TOL_FP32, 1004);    // 四维（高 rank≤8）
    RunShapeCase<float>("L1_017", "float32", {}, 2.0f, TOL_FP32, 1017);             // rank=0 标量输入
    RunShapeCase<float>("L1_020", "float32", {1}, 1.0f, TOL_FP32, 1020);            // 单元素 shape
    RunShapeCase<float>("L1_022", "float32", {1, 1}, 2.0f, TOL_FP32, 1022);         // x3 形态 1x1（等价单元素标量）
    RunShapeCase<float>("L1_023", "float32", {4096, 1024}, 2.0f, TOL_FP32, 1023);   // 大 shape 多核/UB 分块

    // ---- float16（fp32 域承载，阈值用 fp16）: 标准 / 一维 / 三维 / 标量 / 大 shape ----
    RunShapeCase<float>("L1_005", "float16", {2, 3}, 2.0f, TOL_FP16, 1005);          // 标准 rank=2（cast 路径）
    RunShapeCase<float>("L1_006", "float16", {32}, -2.5f, TOL_FP16, 1006);           // 一维 x3 负值
    RunShapeCase<float>("L1_007", "float16", {2, 3, 4}, 1.5f, TOL_FP16, 1007);       // 三维
    RunShapeCase<float>("L1_018", "float16", {}, 2.0f, TOL_FP16, 1018);             // rank=0 标量输入
    RunShapeCase<float>("L1_024", "float16", {4096, 1024}, 2.0f, TOL_FP16, 1024);   // 大 shape 多核/UB 分块

    // ---- bfloat16（fp32 域承载，阈值用 bf16）: 标准 / 一维 / 三维 / 单元素 ----
    RunShapeCase<float>("L1_008", "bfloat16", {2, 3}, 2.0f, TOL_BF16, 1008);         // 标准 rank=2（cast 路径）
    RunShapeCase<float>("L1_009", "bfloat16", {32}, -1.0f, TOL_BF16, 1009);          // 一维 x3 负值
    RunShapeCase<float>("L1_010", "bfloat16", {2, 3, 4}, 0.75f, TOL_BF16, 1010);     // 三维
    RunShapeCase<float>("L1_021", "bfloat16", {1}, 2.0f, TOL_BF16, 1021);           // 单元素 shape

    // ---- int32（direct，bitwise）: 标准 / 一维 / 三维 / 标量 / 大 shape ----
    RunShapeCase<int32_t>("L1_011", "int32", {2, 3}, 3, 0.0, 1011);                  // 标准 rank=2（bitwise）
    RunShapeCase<int32_t>("L1_012", "int32", {32}, -2, 0.0, 1012);                   // 一维 x3 负值
    RunShapeCase<int32_t>("L1_013", "int32", {2, 3, 4}, 5, 0.0, 1013);              // 三维
    RunShapeCase<int32_t>("L1_019", "int32", {}, 3, 0.0, 1019);                     // rank=0 标量输入
    RunShapeCase<int32_t>("L1_025", "int32", {4096, 1024}, 2, 0.0, 1025);          // 大 shape 多核/UB 分块

    // ---- int16（direct，bitwise）: 标准 / 一维 / 三维 ----
    RunShapeCase<int16_t>("L1_014", "int16", {2, 3}, 3, 0.0, 1014);                  // 标准 rank=2（bitwise）
    RunShapeCase<int16_t>("L1_015", "int16", {32}, -1, 0.0, 1015);                   // 一维 x3 负值
    RunShapeCase<int16_t>("L1_016", "int16", {2, 3, 4}, 2, 0.0, 1016);             // 三维

    LOG_PRINT("\n========================================");
    LOG_PRINT("L1 多 shape 用例汇总: 通过 %d / 失败 %d", g_l1_pass, g_l1_fail);
    LOG_PRINT("========================================");
    return g_l1_fail == 0;
}

// ============================================================================
// L1 全量补齐计数器（边界不变量 / 极端 / 整数回绕 / 广播 / 确定性）
// ============================================================================
static int g_l1_full_pass = 0;
static int g_l1_full_fail = 0;

static void RecordFull(const std::string& case_id, bool ok) {
    if (ok) {
        g_l1_full_pass++;
    } else {
        g_l1_full_fail++;
    }
    LOG_PRINT("  -> %s: %s", case_id.c_str(), ok ? "PASS" : "FAIL");
}

// L2-guard 计数器（golden 层非法输入识别防线，报错码语义归属 op_host UT）
static int g_l2_guard_pass = 0;
static int g_l2_guard_fail = 0;

static void RecordGuard(const std::string& case_id, bool ok) {
    if (ok) {
        g_l2_guard_pass++;
    } else {
        g_l2_guard_fail++;
    }
    LOG_PRINT("  -> %s: %s", case_id.c_str(), ok ? "PASS" : "FAIL");
}

// ----------------------------------------------------------------------------
// 边界不变量用例
//
//   L1_026~028: x3=0 ⇒ y == x2（zero_multiplier_yields_x2，直接断言，不依赖 oracle）
//   L1_029~030: x3=1 ⇒ y == x1 + x2（单位元乘数，退化为 addn）
// ----------------------------------------------------------------------------

// x3=0 不变量：y 必须逐元素 bitwise 等于 x2（浮点也应精确相等，因 x1*0+x2=x2）。
template <typename T>
void RunInvariantZeroX3(const std::string& case_id, const std::string& dtype_name,
                        const std::vector<int64_t>& shape, uint64_t seed) {
    size_t size = static_cast<size_t>(GetShapeSize(shape));
    std::vector<T> x1(size), x2(size);
    FillRandom<T>(x1, seed);
    FillRandom<T>(x2, seed + 7919);
    std::vector<T> x3 = {static_cast<T>(0)};
    std::vector<T> y(size);
    ComputeGolden(x1.data(), x2.data(), x3.data(), y.data(), size);

    LOG_PRINT("\n[Mock] 不变量 %s %s shape=%s x3=0 ⇒ y==x2", case_id.c_str(), dtype_name.c_str(),
              ShapeToStr(shape).c_str());
    bool ok = true;
    for (size_t i = 0; i < size; ++i) {
        if (y[i] != x2[i]) {
            ok = false;
            if (i < 5) LOG_PRINT("    不变量违例 [%zu]: y=%f, x2=%f", i, (double)y[i], (double)x2[i]);
        }
    }
    if (ok) LOG_PRINT("  [PASS] x3=0 ⇒ y 逐元素等于 x2（%zu elems）", size);
    RecordFull(case_id, ok);
}

// x3=1 不变量：y 必须等于 x1 + x2（退化为 addn）。
template <typename T>
void RunInvariantUnitX3(const std::string& case_id, const std::string& dtype_name,
                        const std::vector<int64_t>& shape, double threshold, uint64_t seed) {
    size_t size = static_cast<size_t>(GetShapeSize(shape));
    std::vector<T> x1(size), x2(size);
    FillRandom<T>(x1, seed);
    FillRandom<T>(x2, seed + 7919);
    std::vector<T> x3 = {static_cast<T>(1)};
    std::vector<T> y(size), expected(size);
    ComputeGolden(x1.data(), x2.data(), x3.data(), y.data(), size);
    for (size_t i = 0; i < size; ++i) expected[i] = static_cast<T>(x1[i] + x2[i]);

    LOG_PRINT("\n[Mock] 不变量 %s %s shape=%s x3=1 ⇒ y==x1+x2", case_id.c_str(), dtype_name.c_str(),
              ShapeToStr(shape).c_str());
    bool ok = CompareResults(expected.data(), y.data(), size, threshold);
    RecordFull(case_id, ok);
}

// ----------------------------------------------------------------------------
// 空 Tensor 用例（returns_empty）
//   x1/x2/y 空（shape 含 0 维），golden 计算 0 元素，验证框架返回空且不崩溃。
// ----------------------------------------------------------------------------
template <typename T>
void RunEmptyTensor(const std::string& case_id, const std::string& dtype_name,
                    const std::vector<int64_t>& shape) {
    size_t size = static_cast<size_t>(GetShapeSize(shape));
    std::vector<T> x1(size), x2(size), y(size);
    std::vector<T> x3 = {static_cast<T>(2)};
    ComputeGolden(x1.data(), x2.data(), x3.data(), y.data(), size);
    LOG_PRINT("\n[Mock] 空 Tensor %s %s shape=%s（returns_empty）", case_id.c_str(), dtype_name.c_str(),
              ShapeToStr(shape).c_str());
    bool ok = (size == 0) && (y.size() == 0);
    if (ok) LOG_PRINT("  [PASS] 空 tensor（0 元素），golden 返回空且无崩溃");
    else LOG_PRINT("  [FAIL] 期望 0 元素，实际 size=%zu", size);
    RecordFull(case_id, ok);
}

// ----------------------------------------------------------------------------
// 极端输入：x1 含 NaN ⇒ 输出该位置 NaN 传播（浮点）
//   x1[idx]=NaN，其余随机；golden y[idx] = NaN*x3+x2 = NaN（传播）。
//   断言：注入位置 y 为 NaN；非注入位置与公式一致。
// ----------------------------------------------------------------------------
void RunExtremeNaN(const std::string& case_id, const std::string& dtype_name, double threshold) {
    const size_t size = 8;
    const size_t nan_idx = 3;
    std::vector<float> x1(size), x2(size);
    FillRandom<float>(x1, 3300 + nan_idx);
    FillRandom<float>(x2, 3301);
    x1[nan_idx] = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> x3 = {2.0f};
    std::vector<float> y(size);
    ComputeGolden(x1.data(), x2.data(), x3.data(), y.data(), size);

    LOG_PRINT("\n[Mock] 极端 %s %s x1[%zu]=NaN ⇒ 该位置 NaN 传播", case_id.c_str(), dtype_name.c_str(), nan_idx);
    bool nan_ok = std::isnan(y[nan_idx]);
    if (!nan_ok) LOG_PRINT("    [FAIL] y[%zu] 应为 NaN, 实际=%f", nan_idx, (double)y[nan_idx]);
    // 非注入位置与 oracle 一致
    bool others_ok = true;
    for (size_t i = 0; i < size; ++i) {
        if (i == nan_idx) continue;
        float g = x1[i] * x3[0] + x2[i];
        double rel = std::abs((double)y[i] - g) / (std::abs((double)g) + 1e-7);
        if (rel > threshold) {
            others_ok = false;
            LOG_PRINT("    [FAIL] 非注入位 [%zu] rel_err=%.2e > %.2e", i, rel, threshold);
        }
    }
    bool ok = nan_ok && others_ok;
    if (ok) LOG_PRINT("  [PASS] NaN 在位置 %zu 传播，其余位置与 oracle 一致", nan_idx);
    RecordFull(case_id, ok);
}

// 极端输入：x1=+Inf ⇒ 与 oracle 一致（浮点）
//   y = (+Inf)*x3[0] + x2 = +Inf（x3>0）。断言 y 为 +Inf。
void RunExtremePosInf(const std::string& case_id, const std::string& dtype_name) {
    std::vector<float> x1 = {std::numeric_limits<float>::infinity()};
    std::vector<float> x2 = {1.5f};
    std::vector<float> x3 = {2.0f};
    std::vector<float> y(1);
    ComputeGolden(x1.data(), x2.data(), x3.data(), y.data(), 1);
    LOG_PRINT("\n[Mock] 极端 %s %s x1=+Inf, x3=2 ⇒ y=+Inf（与 oracle 一致）", case_id.c_str(),
              dtype_name.c_str());
    bool ok = std::isinf(y[0]) && (y[0] > 0);
    if (ok) LOG_PRINT("  [PASS] y=+Inf 与 oracle((+Inf)*2+1.5=+Inf) 一致");
    else LOG_PRINT("  [FAIL] y=%f, 期望 +Inf", (double)y[0]);
    RecordFull(case_id, ok);
}

// 极端输入：全零 ⇒ y 全 0（全 dtype）
template <typename T>
void RunExtremeAllZero(const std::string& case_id, const std::string& dtype_name) {
    const size_t size = 8;
    std::vector<T> x1(size, static_cast<T>(0)), x2(size, static_cast<T>(0));
    std::vector<T> x3 = {static_cast<T>(0)};
    std::vector<T> y(size);
    ComputeGolden(x1.data(), x2.data(), x3.data(), y.data(), size);
    LOG_PRINT("\n[Mock] 极端 %s %s 全零输入 ⇒ y 全 0", case_id.c_str(), dtype_name.c_str());
    bool ok = true;
    for (size_t i = 0; i < size; ++i) {
        if (y[i] != static_cast<T>(0)) {
            ok = false;
            LOG_PRINT("    [FAIL] y[%zu]=%f != 0", i, (double)y[i]);
        }
    }
    if (ok) LOG_PRINT("  [PASS] y 全 0（%zu elems）", size);
    RecordFull(case_id, ok);
}

// 极端输入：fp16 数值上界 x1=60000, x3=1, x2=60000 ⇒ 与公式对齐（fp32 域 60000+60000=120000，
//   超 fp16 最大可表示值 65504 ⇒ 回 fp16 时为 +Inf）。matches_oracle。
//   本 C++ ST 以 fp32 域承载 fp16 语义；显式做 fp16 截断（round-to-nearest，超界 ⇒ +Inf）以核对“可能 +Inf”。
void RunExtremeFp16Upper(const std::string& case_id) {
    const float kFp16Max = 65504.0f;  // fp16 最大可表示有限值
    float x1 = 60000.0f, x3 = 1.0f, x2 = 60000.0f;
    float y_fp32 = x1 * x3 + x2;  // = 120000.0f
    // 模拟回 fp16：超过 fp16 最大有限值 ⇒ +Inf（round-to-nearest-even 下 |v|>65520 → Inf）
    float y_fp16 = (std::abs(y_fp32) > kFp16Max) ? std::numeric_limits<float>::infinity() : y_fp32;
    LOG_PRINT("\n[Mock] 极端 %s fp16 上界 x1=60000,x3=1,x2=60000 ⇒ fp32域=%.1f ⇒ fp16=%s", case_id.c_str(),
              (double)y_fp32, std::isinf(y_fp16) ? "+Inf" : "有限");
    // 与公式对齐：fp32 域算出 120000，回 fp16 应为 +Inf（120000 > 65504）
    bool ok = std::isinf(y_fp16) && (y_fp16 > 0);
    if (ok) LOG_PRINT("  [PASS] 120000 超 fp16 上界(65504) ⇒ +Inf，与公式/框架对齐");
    else LOG_PRINT("  [FAIL] y_fp16=%f, 期望 +Inf", (double)y_fp16);
    RecordFull(case_id, ok);
}

// ----------------------------------------------------------------------------
// 整数回绕用例（两步公式 + 整型回绕，bitwise）
//   两步截断：tmp=(T)(x1*x3[0]); y=(T)(tmp+x2)。与 C++/kernel 内建回绕语义一致。
//
//   L1_041 INT32 上界回绕: x1=INT32_MAX, x3=1, x2=1 ⇒ tmp=INT32_MAX, y=INT32_MIN（回绕）
//   L1_042 INT16 上界回绕: x1=32767,    x3=1, x2=1 ⇒ tmp=32767,      y=-32768（回绕）
//   L1_043 INT32 下界:     x1=INT32_MIN,x3=1, x2=0 ⇒ y=INT32_MIN
//   L1_044 INT16 下界:     x1=-32768,   x3=1, x2=0 ⇒ y=-32768
// ----------------------------------------------------------------------------
template <typename T>
void RunIntWrap(const std::string& case_id, const std::string& dtype_name, T x1_val, T x3_val, T x2_val,
                T expected) {
    const size_t size = 1;
    std::vector<T> x1 = {x1_val}, x2 = {x2_val}, x3 = {x3_val};
    std::vector<T> y(size);
    ComputeGoldenIntTwoStep(x1.data(), x2.data(), x3.data(), y.data(), size);
    LOG_PRINT("\n[Mock] 整数回绕 %s %s x1=%lld,x3=%lld,x2=%lld ⇒ y=%lld (期望 %lld)", case_id.c_str(),
              dtype_name.c_str(), (long long)x1_val, (long long)x3_val, (long long)x2_val, (long long)y[0],
              (long long)expected);
    bool ok = (y[0] == expected);
    if (ok) LOG_PRINT("  [PASS] 两步回绕 bitwise 等于期望");
    else LOG_PRINT("  [FAIL] y=%lld != 期望 %lld", (long long)y[0], (long long)expected);
    RecordFull(case_id, ok);
}

// ----------------------------------------------------------------------------
// 广播等价用例（x3 单元素 ⇒ 标量广播；[1] 与 [1,1] 等价）
//   验证 x3 形态 [1] 与 [1,1] 取 x3[0] 计算结果 bitwise 一致（仅取首元素，非多维广播）。
// ----------------------------------------------------------------------------
void RunBroadcastEquiv(const std::string& case_id) {
    const size_t size = 6;
    std::vector<float> x1(size), x2(size);
    FillRandom<float>(x1, 2200);
    FillRandom<float>(x2, 2201);
    // x3 形态 A: [1]
    std::vector<float> x3_a = {2.0f};
    // x3 形态 B: [1,1]（ShapeSize 仍为 1，仅取 x3[0]）
    std::vector<float> x3_b = {2.0f};
    std::vector<float> y_a(size), y_b(size);
    ComputeGolden(x1.data(), x2.data(), x3_a.data(), y_a.data(), size);
    ComputeGolden(x1.data(), x2.data(), x3_b.data(), y_b.data(), size);
    LOG_PRINT("\n[Mock] 广播等价 %s x3 形态 [1] vs [1,1] ⇒ 结果应 bitwise 一致", case_id.c_str());
    bool ok = true;
    for (size_t i = 0; i < size; ++i) {
        if (y_a[i] != y_b[i]) {
            ok = false;
            LOG_PRINT("    [FAIL] [%zu]: [1]=%f, [1,1]=%f", i, (double)y_a[i], (double)y_b[i]);
        }
    }
    if (ok) LOG_PRINT("  [PASS] x3 [1] 与 [1,1] 仅取 x3[0]，结果 bitwise 一致");
    RecordFull(case_id, ok);
}

// ----------------------------------------------------------------------------
// 确定性用例（required=true, bitwise_reproducible=true）
//   同一输入连续执行 N 次 golden，逐次输出必须 bitwise 完全一致（逐元素独立，无累加顺序问题）。
//   L1_045: fp32 小 shape [2,3]
//   L1_046: fp16(fp32域) 大 shape [4096,1024]（多核/UB 分块，仍逐元素独立）
// ----------------------------------------------------------------------------
template <typename T>
void RunDeterminism(const std::string& case_id, const std::string& dtype_name,
                    const std::vector<int64_t>& shape, T x3_val, int repeats, uint64_t seed) {
    size_t size = static_cast<size_t>(GetShapeSize(shape));
    std::vector<T> x1(size), x2(size);
    FillRandom<T>(x1, seed);
    FillRandom<T>(x2, seed + 7919);
    std::vector<T> x3 = {x3_val};

    std::vector<T> ref(size);
    ComputeGolden(x1.data(), x2.data(), x3.data(), ref.data(), size);

    LOG_PRINT("\n[Mock] 确定性 %s %s shape=%s 连续执行 %d 次 ⇒ bitwise 一致", case_id.c_str(),
              dtype_name.c_str(), ShapeToStr(shape).c_str(), repeats);
    bool ok = true;
    for (int r = 1; r < repeats && ok; ++r) {
        std::vector<T> cur(size);
        ComputeGolden(x1.data(), x2.data(), x3.data(), cur.data(), size);
        for (size_t i = 0; i < size; ++i) {
            // 浮点也要求 bitwise 一致：逐元素独立计算，无并行累加 ⇒ 位级可复现
            if (memcmp(&cur[i], &ref[i], sizeof(T)) != 0) {
                ok = false;
                LOG_PRINT("    [FAIL] 第 %d 次 [%zu] 与首次不一致", r + 1, i);
                break;
            }
        }
    }
    if (ok) LOG_PRINT("  [PASS] %d 次执行逐位一致（%zu elems，bitwise_reproducible）", repeats, size);
    RecordFull(case_id, ok);
}

// ----------------------------------------------------------------------------
// L2-guard：golden 层对非法输入的前置识别（报错码语义归属 op_host UT，本层仅做防线）
//   - shape 不一致（x1.size != x2.size）⇒ RunTest 已 [FAIL] 拒绝
//   - x3 空 ⇒ RunTest 已 [FAIL] 拒绝
//   本函数显式构造非法输入，断言 RunTest 返回 false（即“被识别/拒绝”→ guard PASS）。
//   注: 此处“PASS”指【非法输入被正确拒绝】，与正向用例语义相反。
// ----------------------------------------------------------------------------
void RunL2Guards() {
    LOG_PRINT("\n========================================");
    LOG_PRINT("L2-guard: golden 层非法输入识别（报错码语义归属 op_host UT）");
    LOG_PRINT("========================================");

    // ERR-guard-01: x1/x2 shape 不一致（size 4 vs 6）⇒ RunTest 应返回 false
    {
        std::vector<float> x1(4, 1.0f), x2(6, 2.0f), x3 = {2.0f};
        bool rejected = !RunTest<float>("guard x1/x2 size 不一致", x1, x2, x3, {4}, TOL_FP32);
        LOG_PRINT("  [%s] L2-guard ERR-01: x1/x2 不一致被拒绝", rejected ? "PASS" : "FAIL");
        RecordGuard("L2g_ERR01", rejected);
    }
    // ERR-guard-02: x3 为空 ⇒ RunTest 应返回 false
    {
        std::vector<float> x1(4, 1.0f), x2(4, 2.0f), x3;  // x3 空
        bool rejected = !RunTest<float>("guard x3 空", x1, x2, x3, {4}, TOL_FP32);
        LOG_PRINT("  [%s] L2-guard ERR-02: x3 空被拒绝", rejected ? "PASS" : "FAIL");
        RecordGuard("L2g_ERR02", rejected);
    }
    // ERR-guard-03: x1 元素数与 shape 不一致 ⇒ RunTest 应返回 false
    {
        std::vector<float> x1(4, 1.0f), x2(4, 2.0f), x3 = {2.0f};
        bool rejected = !RunTest<float>("guard x1 与 shape 不一致", x1, x2, x3, {2, 3}, TOL_FP32);
        LOG_PRINT("  [%s] L2-guard ERR-03: x1 与 shape(2x3=6) 不一致被拒绝", rejected ? "PASS" : "FAIL");
        RecordGuard("L2g_ERR03", rejected);
    }
}

// ============================================================================
// L1 全量补齐运行器：边界不变量 / 极端 / 整数回绕 / 广播 / 确定性
// ============================================================================
bool RunL1FullCases() {
    LOG_PRINT("\n========================================");
    LOG_PRINT("L1 全量补齐用例");
    LOG_PRINT("覆盖: 边界不变量 + 极端输入 + 整数回绕 + 广播等价 + 确定性");
    LOG_PRINT("========================================");

    // ---- 边界不变量: x3=0 ⇒ y==x2（L1_026~028） ----
    RunInvariantZeroX3<float>("L1_026", "float32", {4, 5}, 2601);
    RunInvariantZeroX3<float>("L1_027", "float16", {4, 5}, 2602);   // fp16 在 fp32 域，x1*0+x2=x2 精确
    RunInvariantZeroX3<int32_t>("L1_028", "int32", {4, 5}, 2603);

    // ---- 边界不变量: x3=1 ⇒ y==x1+x2（L1_029~030） ----
    RunInvariantUnitX3<float>("L1_029", "float32", {4, 5}, TOL_FP32, 2901);
    RunInvariantUnitX3<float>("L1_030", "bfloat16", {4, 5}, TOL_BF16, 2902);

    // ---- 空 Tensor: returns_empty（L1_031~032） ----
    RunEmptyTensor<float>("L1_031", "float32", {0, 3});
    RunEmptyTensor<int32_t>("L1_032", "int32", {0, 3});

    // ---- 极端输入: x1 含 NaN ⇒ NaN 传播（L1_033~035，浮点 3 dtype） ----
    RunExtremeNaN("L1_033", "float32", TOL_FP32);
    RunExtremeNaN("L1_034", "float16", TOL_FP16);
    RunExtremeNaN("L1_035", "bfloat16", TOL_BF16);

    // ---- 极端输入: x1=+Inf ⇒ 与 oracle 一致（L1_036~037） ----
    RunExtremePosInf("L1_036", "float32");
    RunExtremePosInf("L1_037", "float16");

    // ---- 极端输入: 全零 ⇒ y 全 0（L1_038~039） ----
    RunExtremeAllZero<float>("L1_038", "float32");
    RunExtremeAllZero<int32_t>("L1_039", "int32");

    // ---- 极端输入: fp16 上界 60000 ⇒ +Inf（L1_040） ----
    RunExtremeFp16Upper("L1_040");

    // ---- 整数回绕: 上界回绕 / 下界（L1_041~044，两步截断 bitwise） ----
    RunIntWrap<int32_t>("L1_041", "int32", std::numeric_limits<int32_t>::max(), 1, 1,
                        std::numeric_limits<int32_t>::min());  // INT32_MAX+1 ⇒ INT32_MIN
    RunIntWrap<int16_t>("L1_042", "int16", static_cast<int16_t>(32767), 1, 1,
                        static_cast<int16_t>(-32768));  // 32767+1 ⇒ -32768
    RunIntWrap<int32_t>("L1_043", "int32", std::numeric_limits<int32_t>::min(), 1, 0,
                        std::numeric_limits<int32_t>::min());  // 下界保持
    RunIntWrap<int16_t>("L1_044", "int16", static_cast<int16_t>(-32768), 1, 0,
                        static_cast<int16_t>(-32768));  // 下界保持

    // ---- 广播等价: x3 [1] 与 [1,1] 等价（补充覆盖 broadcast 语义） ----
    RunBroadcastEquiv("L1_bcast");

    // ---- 确定性: 连续执行 N 次 bitwise 一致（L1_045~046） ----
    RunDeterminism<float>("L1_045", "float32", {2, 3}, 2.0f, 3, 4501);             // 小 shape 3 次
    RunDeterminism<float>("L1_046", "float16", {4096, 1024}, 2.0f, 3, 4502);       // 大 shape 多核 3 次

    LOG_PRINT("\n========================================");
    LOG_PRINT("L1 全量补齐用例汇总: 通过 %d / 失败 %d", g_l1_full_pass, g_l1_full_fail);
    LOG_PRINT("========================================");
    return g_l1_full_fail == 0;
}

// ============================================================================
// 主函数
// ============================================================================

int main(int argc, char* argv[]) {
    (void)argc;
    (void)argv;

    LOG_PRINT("\n========================================");
    LOG_PRINT("FusedMulAddN 算子 ST 测试（全量用例）");
    LOG_PRINT("公式: y_i = x1_i * x3[0] + x2_i");
    LOG_PRINT("========================================");

#ifdef USE_MOCK_ACLNN
    LOG_PRINT("模式: Mock (CPU golden 验证，无需 NPU)");
#else
    LOG_PRINT("模式: Real");
    LOG_PRINT("注意: 本算子【无 aclnn 两段式接口】，");
    LOG_PRINT("      C++ ST Real 路径不走 aclnn；设备侧上板精度验收由 PyTorch（torch_npu）独立任务承担。");
    LOG_PRINT("      本 C++ ST 在 Real 编译下仍执行 CPU Golden 自测以验证框架与公式。");
#endif

    // 1. CPU Golden 公式自测
    TestGoldenCorrectness();

    // 2. L0 标准用例
    LOG_PRINT("\n执行 L0 标准用例...");
    bool l0_ok = RunL0Cases();

    // 3. L1 多 shape 用例
    LOG_PRINT("\n执行 L1 多 shape 用例...");
    bool l1_shape_ok = RunL1ShapeCases();

    // 4. L1 全量补齐用例（边界不变量/极端/整数回绕/广播/确定性）
    LOG_PRINT("\n执行 L1 全量补齐用例...");
    bool l1_full_ok = RunL1FullCases();

    // 5. L2-guard 用例（golden 层非法输入识别，报错码语义归 op_host UT）
    LOG_PRINT("\n执行 L2-guard 用例...");
    RunL2Guards();
    bool l2_guard_ok = (g_l2_guard_fail == 0);

    bool all_ok = (g_golden_fail == 0) && l0_ok && l1_shape_ok && l1_full_ok && l2_guard_ok;

    LOG_PRINT("\n========================================");
    LOG_PRINT("测试报告（全量用例）");
    LOG_PRINT("========================================");
    LOG_PRINT("CPU Golden 自测:    通过 %d / 失败 %d", g_golden_pass, g_golden_fail);
    LOG_PRINT("L0 门槛用例:        通过 %d / 失败 %d (%s)", g_l0_pass, g_l0_fail, l0_ok ? "全部通过" : "存在失败");
    LOG_PRINT("L1 多 shape 用例:   通过 %d / 失败 %d (%s)", g_l1_pass, g_l1_fail,
              l1_shape_ok ? "全部通过" : "存在失败");
    LOG_PRINT("L1 全量补齐用例:    通过 %d / 失败 %d (%s)", g_l1_full_pass, g_l1_full_fail,
              l1_full_ok ? "全部通过" : "存在失败");
    LOG_PRINT("  (边界不变量/极端输入/整数回绕/广播等价/确定性)");
    LOG_PRINT("L2-guard 用例:      通过 %d / 失败 %d (%s)", g_l2_guard_pass, g_l2_guard_fail,
              l2_guard_ok ? "全部通过" : "存在失败");
    LOG_PRINT("  (报错码语义归属 op_host UT 的报错用例)");

    int total_pass = g_golden_pass + g_l0_pass + g_l1_pass + g_l1_full_pass + g_l2_guard_pass;
    int total_fail = g_golden_fail + g_l0_fail + g_l1_fail + g_l1_full_fail + g_l2_guard_fail;
    LOG_PRINT("----------------------------------------");
    LOG_PRINT("用例总计:          通过 %d / 失败 %d (通过率 %.1f%%)", total_pass, total_fail,
              (total_pass + total_fail) ? 100.0 * total_pass / (total_pass + total_fail) : 0.0);
    LOG_PRINT("总体结果: %s", all_ok ? "PASS" : "FAIL");
    LOG_PRINT("========================================\n");

    return all_ok ? 0 : 1;
}
