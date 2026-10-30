/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_fused_mul_add_n.cpp
 * \brief FusedMulAddN A2(ascend910b) op_kernel 数值 UT。
 *
 * 被测对象：A2 flat kernel `op_kernel/fused_mul_add_n.cpp`，语义 y = x1 * x3[0] + x2。
 * 验证手段：CPU 模式直跑 kernel（tikicpulib + ICPU_RUN_KF）+ inline CPU golden 比对。
 *   本算子无 aclnn 接口，数值验证以 op_kernel UT 为准。
 * 覆盖范围：全 5 dtype（fp32/fp16/int32/int16/bf16）× 多种 shape/对齐/尾块/单元素/标量 x3 取值：
 *   - 对齐基础 / 非对齐尾块 / 单元素 / 多 D / 中规模单核多 tile / 大规模多核多 tile /
 *     大规模多核非对齐尾块 / x3 特殊值（0/1/负）。
 *   - x3 形态等价：x3 以 [1,1] 形态传入 vs [1]，验证数值逐元素严格相等（kernel 仅取
 *     x3[0]，张量形态不改数值）。
 *   - 边界强化：极端尾块 tail=1（多核场景末核仅 1 元素）、各 dtype 32B 对齐粒度边界
 *     （fp32/int32=8、fp16/bf16/int16=16，及 ±1 邻域）、超大多核多 tile 的代表性规模、
 *     空 tensor（totalNum=0）。
 *
 * 广播说明：本算子 x1/x2/y 同 shape 逐元素，【无 x1/x2 broadcast】；x3 为单元素标量
 *   广播（kernel 取 x3[0] 后作用于全部元素）。x3 标量广播覆盖 = x3 单值作用于多种
 *   x1/x2 shape（[1024]/[37]/[1]/多 D/大规模 等用例已覆盖）。
 *
 * 精度标准（浮点社区精度）：
 *   - fp32：MERE < 2^-13 且 MARE < 10*2^-13。
 *   - fp16：MERE < 2^-10 且 MARE < 10*2^-10。
 *   - bf16：MERE < 2^-7  且 MARE < 10*2^-7。
 *   - int32 / int16 整数计算类：二进制一致 / 绝对误差 0。
 *
 * dtype 分发：A2 kernel `fused_mul_add_n.cpp` 按 **TilingKey** 在运行期选择 dtype 模板，
 *   入口签名与 DTYPE_X 编译宏无关（已核对 op_kernel 源码不引用 DTYPE_X）。因此本 UT
 *   用同一份编译产物，按 dtype 设置对应 ICPU_SET_TILING_KEY（0=fp32/1=fp16/2=int32/
 *   3=int16/4=bf16）即可覆盖全部 5 dtype。
 *
 * golden 语义：
 *   - fp32：原生 float 直算（中间即 fp32）。
 *   - int32/int16：整型同 dtype 计算（C++ 整型运算后截断回该 dtype，与 kernel Muls/Add 同位宽语义一致）。
 *   - fp16/bf16：升 fp32 计算后舍回半精度（round-to-nearest-even，对齐 kernel Cast→fp32→Muls→Add→Cast(CAST_RINT)）。
 *
 * TilingData：使用本目录 fused_mul_add_n_tiling.h 的扁平 POD。
 *   多核 / 多 tile 切分逻辑严格按 op_host/fused_mul_add_n_tiling.cpp 的算法
 *   移植（参数化 coreNum/ubSize），使 UT 切分与 host 一致。
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <random>
#include <type_traits>

#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"

#include "fused_mul_add_n_tiling.h"

// A2 flat kernel 入口：实验态 op_kernel UT 直接 #include kernel .cpp 编入测试翻译单元
//（与 experimental/math/accumulate_nv2、acos 等已验证用例一致）。kernel 内的
// GET_TILING_DATA_WITH_STRUCT 已被本目录 fused_mul_add_n_tiling.h（cmake -include 注入）
// 重定义为从扁平 tiling 缓冲反序列化，故 UT 完全自包含。
// 必须在 `using namespace std;` 之前 include，避免 AscendC 的 dec 与 std::dec 二义冲突。
#include "../../../op_kernel/fused_mul_add_n.cpp"

using namespace std;
// bfloat16_t（= bfloat16::Bf16T）与 half 由 tikicpulib（kernel_bf16.h / kernel_fp16.h）
// 提供为全局类型，直接使用（与 math/is_inf op_kernel UT 一致）。

class FusedMulAddN : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "FusedMulAddN SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "FusedMulAddN TearDown" << endl;
    }
};

template <typename T1, typename T2>
static inline T1 CeilAlign(T1 a, T2 b)
{
    return (a + b - 1) / b * b;
}

template <typename T1, typename T2>
static inline T1 CeilDiv(T1 a, T2 b)
{
    return (a + b - 1) / b;
}

// ===========================================================================
// dtype → TilingKey / sizeof / 每元素 UB 字节数（与 host op_host/fused_mul_add_n_tiling.cpp 完全一致）
// ===========================================================================
enum class DType {
    FP32 = 0,
    FP16 = 1,
    INT32 = 2,
    INT16 = 3,
    BF16 = 4
};

static int64_t DTypeTilingKey(DType d)
{
    return static_cast<int64_t>(d);
}

static int64_t DTypeElemSize(DType d)
{
    switch (d) {
        case DType::FP32:
        case DType::INT32:
            return 4;
        case DType::FP16:
        case DType::BF16:
        case DType::INT16:
            return 2;
    }
    return 4;
}

// 每元素 UB 占用字节（host GetBytesPerElem 同值）：直算 6×sizeof(T)；
//   bf16 cast 域 5-op 固定 20（2 块 fp32 scratch）；fp16 Axpy 融合 16（1 块 fp32 scratch）。
static int64_t DTypeBytesPerElem(DType d)
{
    switch (d) {
        case DType::FP32:
        case DType::INT32:
            return 6 * 4; // 24
        case DType::INT16:
            return 6 * 2; // 12
        case DType::FP16:
            return 16; // Axpy 融合：3×2×2 + 1×4
        case DType::BF16:
            return 20; // bf16 5-op cast 域：3×2×2 + 2×4
    }
    return 24;
}

// ---------------------------------------------------------------------------
// Tiling 数据构造（UT 内移植 host op_host/fused_mul_add_n_tiling.cpp 多核/多 tile 切分算法，参数化 coreNum/ubSize）
// ---------------------------------------------------------------------------
// 字段语义见 fused_mul_add_n_tiling.h。
//   - blockNum：实际使用核数（ICPU_RUN_KF 的 numBlocks 须与之一致）
//   - 单核内按 ubFormer 做 ub 循环切分（former 核 / 尾核区分）
struct UtTilingPlan {
    FusedMulAddNTilingDataUT data;
    int64_t blockNum;
};

// 给定 totalNum / dtype / 模拟平台(coreNum,ubSize) 产出切分。
// 与 op_host/fused_mul_add_n_tiling.cpp 一致：含 min-bytes-per-core 核数 cap。
static UtTilingPlan BuildTiling(int64_t totalNum, DType d, int64_t coreNum, int64_t ubSize)
{
    const int64_t BLOCK_BYTES = 32;
    const int64_t REMAINED_UB = 16 * 1024;
    // 核数 cap（与 host 一致）：每核至少 48KB（3 路 IO）。
    const int64_t MIN_BYTES_PER_CORE = 48 * 1024;
    const int64_t IO_PATHS_PER_ELEM = 3;

    int64_t elemSize = DTypeElemSize(d);
    int64_t elemPerBlockAlign = BLOCK_BYTES / elemSize; // fp32/int32=8；fp16/bf16/int16=16

    int64_t blockNum = 1;
    int64_t blockFormer = 0;
    int64_t blockTail = 0;
    int64_t ubFormer = 0;
    int64_t ubLoopOfFormerBlock = 0;
    int64_t ubLoopOfTailBlock = 0;
    int64_t ubTailOfFormerBlock = 0;
    int64_t ubTailOfTailBlock = 0;

    if (totalNum > 0) {
        // 按 min-bytes-per-core cap 封顶有效核数（小 shape 降核），再参与 perCore 计算。
        int64_t minElemsPerCore = MIN_BYTES_PER_CORE / (IO_PATHS_PER_ELEM * elemSize);
        if (minElemsPerCore < 1) {
            minElemsPerCore = 1;
        }
        int64_t effCoreNum = totalNum / minElemsPerCore; // 向下取整
        if (effCoreNum < 1) {
            effCoreNum = 1;
        }
        if (effCoreNum > coreNum) {
            effCoreNum = coreNum;
        }
        int64_t perCore = CeilDiv(totalNum, effCoreNum);
        blockFormer = CeilAlign(perCore, elemPerBlockAlign);
        if (blockFormer < elemPerBlockAlign) {
            blockFormer = elemPerBlockAlign;
        }
        blockNum = CeilDiv(totalNum, blockFormer);
        if (blockNum < 1) {
            blockNum = 1;
        }
        blockTail = totalNum - blockFormer * (blockNum - 1);

        int64_t usableUb = ubSize - REMAINED_UB;
        if (usableUb < 0) {
            usableUb = 0;
        }
        int64_t bytesPerElem = DTypeBytesPerElem(d);
        int64_t ubFormerMax = usableUb / bytesPerElem;
        ubFormerMax = ubFormerMax / elemPerBlockAlign * elemPerBlockAlign; // 32B 对齐
        if (ubFormerMax < elemPerBlockAlign) {
            ubFormerMax = elemPerBlockAlign;
        }
        ubFormer = (blockFormer < ubFormerMax) ? blockFormer : ubFormerMax;
        if (ubFormer < elemPerBlockAlign) {
            ubFormer = elemPerBlockAlign;
        }

        ubLoopOfFormerBlock = CeilDiv(blockFormer, ubFormer);
        ubTailOfFormerBlock = blockFormer - ubFormer * (ubLoopOfFormerBlock - 1);
        ubLoopOfTailBlock = CeilDiv(blockTail, ubFormer);
        ubTailOfTailBlock = blockTail - ubFormer * (ubLoopOfTailBlock - 1);
    } else {
        blockNum = 1;
        ubFormer = elemPerBlockAlign;
    }

    UtTilingPlan plan;
    plan.data.totalNum = totalNum;
    plan.data.blockNum = blockNum;
    plan.data.blockFormer = blockFormer;
    plan.data.blockTail = blockTail;
    plan.data.ubFormer = ubFormer;
    plan.data.ubLoopOfFormerBlock = ubLoopOfFormerBlock;
    plan.data.ubLoopOfTailBlock = ubLoopOfTailBlock;
    plan.data.ubTailOfFormerBlock = ubTailOfFormerBlock;
    plan.data.ubTailOfTailBlock = ubTailOfTailBlock;
    plan.blockNum = blockNum;
    return plan;
}

// ===========================================================================
// 精度比对
// ===========================================================================
struct PrecResult {
    double mere = 0.0;
    double mare = 0.0;
    double threshold = 0.0;
    int64_t mismatch = 0; // 整型用：不一致元素数
    bool isPass = false;
};

// 浮点社区标准 MERE/MARE：在 fp32 域用最终 dtype 数值（半精度已舍回）比对。
//   MERE = mean(|a-g| / (|g|+1e-7)) ; MARE = max(同) ; 判定 MERE<thr && MARE<10*thr
static PrecResult CheckFloatCommunity(
    const std::vector<float>& actual, const std::vector<float>& golden, double threshold)
{
    const double eps = 1e-7;
    int64_t n = static_cast<int64_t>(golden.size());
    double sumRe = 0.0;
    double maxRe = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        double diff = std::fabs(static_cast<double>(actual[i]) - static_cast<double>(golden[i]));
        double re = diff / (std::fabs(static_cast<double>(golden[i])) + eps);
        sumRe += re;
        if (re > maxRe) {
            maxRe = re;
        }
    }
    PrecResult r;
    r.mere = (n > 0) ? sumRe / static_cast<double>(n) : 0.0;
    r.mare = maxRe;
    r.threshold = threshold;
    r.isPass = (r.mere < threshold) && (r.mare < 10.0 * threshold);
    return r;
}

// 整数计算类标准：要求二进制一致 / 绝对误差 0。
template <typename IntT>
static PrecResult CheckIntegerExact(const std::vector<IntT>& actual, const std::vector<IntT>& golden)
{
    int64_t n = static_cast<int64_t>(golden.size());
    int64_t mismatch = 0;
    for (int64_t i = 0; i < n; ++i) {
        if (actual[i] != golden[i]) {
            ++mismatch;
        }
    }
    PrecResult r;
    r.mismatch = mismatch;
    r.threshold = 0.0;
    r.isPass = (mismatch == 0);
    return r;
}

// 各 dtype 社区标准阈值
static constexpr double FP32_THRESHOLD = 1.0 / 8192.0; // 2^-13 ≈ 1.2207e-4
static constexpr double FP16_THRESHOLD = 1.0 / 1024.0; // 2^-10 ≈ 9.7656e-4
static constexpr double BF16_THRESHOLD = 1.0 / 128.0;  // 2^-7  ≈ 7.8125e-3

static double FloatThresholdOf(DType d)
{
    switch (d) {
        case DType::FP32:
            return FP32_THRESHOLD;
        case DType::FP16:
            return FP16_THRESHOLD;
        case DType::BF16:
            return BF16_THRESHOLD;
        default:
            return FP32_THRESHOLD;
    }
}

// ===========================================================================
// 通用执行器（模板）：构造输入/tiling → 跑 kernel → 取 dtype 对应 golden → 比对
//   T       : kernel 端真实 dtype（float/half/bfloat16_t/int32_t/int16_t）
//   d       : DType 枚举（决定 TilingKey / golden 域 / 阈值）
//   totalNum: x1/x2/y 总元素数
//   x3Value : 标量 x3（以 float 给出，整型场景取整后存）
//   coreNum/ubSize: 模拟平台参数（控制多核/多 tile 切分）
//   seed    : 数据生成种子
// ===========================================================================

// 浮点（fp32/fp16/bf16）执行器
template <typename T>
static PrecResult RunFloatCase(
    DType d, int64_t totalNum, float x3Value, int64_t coreNum, int64_t ubSize, unsigned seed, float lo, float hi)
{
    UtTilingPlan plan = BuildTiling(totalNum, d, coreNum, ubSize);

    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(lo, hi);

    size_t inByteSize = static_cast<size_t>(totalNum) * sizeof(T);
    size_t x3ByteSize = sizeof(T);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(CeilAlign(inByteSize, 32));
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(CeilAlign(inByteSize, 32));
    uint8_t* x3 = (uint8_t*)AscendC::GmAlloc(CeilAlign(x3ByteSize, 32));
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(CeilAlign(inByteSize, 32));
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(static_cast<size_t>(16 * 1024 * 1024));
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(FusedMulAddNTilingDataUT));

    T* x1t = reinterpret_cast<T*>(x1);
    T* x2t = reinterpret_cast<T*>(x2);
    T* x3t = reinterpret_cast<T*>(x3);

    // 用 fp32 域生成数据，再写入 dtype（fp16/bf16 经构造函数舍入到半精度）。
    // 同步保存舍入回 fp32 的输入副本，golden 必须用「实际写入 kernel 的半精度值」计算，
    // 与 kernel 读到的输入完全一致（避免标杆比 kernel 多/少一次舍入误差）。
    std::vector<float> x1f(static_cast<size_t>(totalNum));
    std::vector<float> x2f(static_cast<size_t>(totalNum));
    for (int64_t i = 0; i < totalNum; ++i) {
        T a = static_cast<T>(dist(gen));
        T b = static_cast<T>(dist(gen));
        x1t[i] = a;
        x2t[i] = b;
        x1f[i] = static_cast<float>(a); // 半精度舍回 fp32
        x2f[i] = static_cast<float>(b);
    }
    T x3rounded = static_cast<T>(x3Value);
    x3t[0] = x3rounded;
    float x3f = static_cast<float>(x3rounded);

    std::memcpy(tiling, &plan.data, sizeof(FusedMulAddNTilingDataUT));

    ICPU_SET_TILING_KEY(DTypeTilingKey(d));
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(fused_mul_add_n, plan.blockNum, x1, x2, x3, y, workspace, tiling);

    // golden：升 fp32 计算 y=x1*x3+x2，再舍回半精度（fp32 时舍回为恒等），最后回 fp32 比对。
    //   ⚠ 必须忠实镜像 kernel 的【分步】Muls→Add（见 fused_mul_add_n_align*.h：
    //     Muls(y, x1, x3) 后 Add(y, y, x2)，两次独立 fp32 舍入），而非单表达式
    //     `x1*x3 + x2`。后者可能被编译器 contraction 成 fma(x1,x3,x2)（单次舍入），
    //     在零交叉/抵消处产生与硬件不同的 last-bit 结果导致假阴。用 volatile 阻止
    //     乘加合并，强制乘法先单独求值并落 fp32。
    std::vector<float> yGolden(static_cast<size_t>(totalNum));
    for (int64_t i = 0; i < totalNum; ++i) {
        volatile float mul = x1f[i] * x3f;            // 步骤1：fp32 乘（单独求值，阻止 FMA 合并）
        float val = static_cast<float>(mul) + x2f[i]; // 步骤2：fp32 加
        T rounded = static_cast<T>(val);              // round-to-nearest-even 回半精度（fp32 恒等）
        yGolden[i] = static_cast<float>(rounded);
    }

    // 实际输出转回 fp32 比对
    T* yt = reinterpret_cast<T*>(y);
    std::vector<float> yActual(static_cast<size_t>(totalNum));
    for (int64_t i = 0; i < totalNum; ++i) {
        yActual[i] = static_cast<float>(yt[i]);
    }

    PrecResult res = CheckFloatCommunity(yActual, yGolden, FloatThresholdOf(d));

    AscendC::GmFree((void*)x1);
    AscendC::GmFree((void*)x2);
    AscendC::GmFree((void*)x3);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    return res;
}

// 整型（int32/int16）执行器
template <typename T>
static PrecResult RunIntCase(
    DType d, int64_t totalNum, int32_t x3Value, int64_t coreNum, int64_t ubSize, unsigned seed, int32_t lo, int32_t hi)
{
    UtTilingPlan plan = BuildTiling(totalNum, d, coreNum, ubSize);

    std::mt19937 gen(seed);
    std::uniform_int_distribution<int32_t> dist(lo, hi);

    size_t inByteSize = static_cast<size_t>(totalNum) * sizeof(T);
    size_t x3ByteSize = sizeof(T);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(CeilAlign(inByteSize, 32));
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(CeilAlign(inByteSize, 32));
    uint8_t* x3 = (uint8_t*)AscendC::GmAlloc(CeilAlign(x3ByteSize, 32));
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(CeilAlign(inByteSize, 32));
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(static_cast<size_t>(16 * 1024 * 1024));
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(FusedMulAddNTilingDataUT));

    T* x1t = reinterpret_cast<T*>(x1);
    T* x2t = reinterpret_cast<T*>(x2);
    T* x3t = reinterpret_cast<T*>(x3);
    std::vector<T> x1v(static_cast<size_t>(totalNum));
    std::vector<T> x2v(static_cast<size_t>(totalNum));
    for (int64_t i = 0; i < totalNum; ++i) {
        T a = static_cast<T>(dist(gen));
        T b = static_cast<T>(dist(gen));
        x1t[i] = a;
        x2t[i] = b;
        x1v[i] = a;
        x2v[i] = b;
    }
    T x3t0 = static_cast<T>(x3Value);
    x3t[0] = x3t0;

    std::memcpy(tiling, &plan.data, sizeof(FusedMulAddNTilingDataUT));

    ICPU_SET_TILING_KEY(DTypeTilingKey(d));
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(fused_mul_add_n, plan.blockNum, x1, x2, x3, y, workspace, tiling);

    // golden：整型同 dtype 计算 y = (T)((T)(x1*x3) + x2)
    //   逐步截断回 T，与 kernel Muls(T)+Add(T) 同位宽语义一致（取值范围内不溢出）。
    std::vector<T> yGolden(static_cast<size_t>(totalNum));
    for (int64_t i = 0; i < totalNum; ++i) {
        T mul = static_cast<T>(x1v[i] * x3t0);
        yGolden[i] = static_cast<T>(mul + x2v[i]);
    }

    T* yt = reinterpret_cast<T*>(y);
    std::vector<T> yActual(yt, yt + totalNum);

    PrecResult res = CheckIntegerExact<T>(yActual, yGolden);

    AscendC::GmFree((void*)x1);
    AscendC::GmFree((void*)x2);
    AscendC::GmFree((void*)x3);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    return res;
}

// ---------------------------------------------------------------------------
// fp32 输出捕获执行器（用于 x3 形态等价对比）：返回 kernel 实际输出（原样 float）。
//   与 RunFloatCase<float> 同路径，但额外参数 x3ElemCapacity 控制 x3 GM 缓冲的
//   逻辑元素容量（[1] → 1；[1,1] → 仍为 1 个逻辑元素，但以"形态 [1,1]"语义构造，
//   即分配 ≥1 元素并在 x3[0] 写入值）。kernel 仅 GetValue(0)，故两种形态应产出
//   完全相同的字节级输出 —— 证明 x3 张量形态不改数值。
//   ⚠ x3ElemCapacity 仅影响 GM 分配大小，不改变 kernel 读取语义（始终取 x3[0]）。
static std::vector<float> RunFp32Capture(
    int64_t totalNum, float x3Value, int64_t coreNum, int64_t ubSize, unsigned seed, float lo, float hi,
    int64_t x3ElemCapacity)
{
    UtTilingPlan plan = BuildTiling(totalNum, DType::FP32, coreNum, ubSize);

    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(lo, hi);

    size_t inByteSize = static_cast<size_t>(totalNum) * sizeof(float);
    size_t x3ByteSize = static_cast<size_t>(x3ElemCapacity) * sizeof(float);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(CeilAlign(inByteSize, 32));
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(CeilAlign(inByteSize, 32));
    uint8_t* x3 = (uint8_t*)AscendC::GmAlloc(CeilAlign(x3ByteSize, 32));
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(CeilAlign(inByteSize, 32));
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(static_cast<size_t>(16 * 1024 * 1024));
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(FusedMulAddNTilingDataUT));

    float* x1t = reinterpret_cast<float*>(x1);
    float* x2t = reinterpret_cast<float*>(x2);
    float* x3t = reinterpret_cast<float*>(x3);
    for (int64_t i = 0; i < totalNum; ++i) {
        x1t[i] = dist(gen);
        x2t[i] = dist(gen);
    }
    // x3[0] 写入标量值；若形态为 [1,1]（capacity≥2）其余位置置 0（kernel 不读，仅验形态无关）。
    for (int64_t i = 0; i < x3ElemCapacity; ++i) {
        x3t[i] = (i == 0) ? x3Value : 0.0f;
    }

    std::memcpy(tiling, &plan.data, sizeof(FusedMulAddNTilingDataUT));

    ICPU_SET_TILING_KEY(DTypeTilingKey(DType::FP32));
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(fused_mul_add_n, plan.blockNum, x1, x2, x3, y, workspace, tiling);

    float* yt = reinterpret_cast<float*>(y);
    std::vector<float> yActual(yt, yt + totalNum);

    AscendC::GmFree((void*)x1);
    AscendC::GmFree((void*)x2);
    AscendC::GmFree((void*)x3);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    return yActual;
}

// ---------------------------------------------------------------------------
// 断言辅助（按浮点 / 整型分别打印通过信息）
// ---------------------------------------------------------------------------
static void ExpectFloatPass(const PrecResult& r, const std::string& caseId)
{
    std::cout << "[" << caseId << "] MERE=" << r.mere << " (thr " << r.threshold << "), MARE=" << r.mare << " (thr "
              << 10.0 * r.threshold << ") => " << (r.isPass ? "PASS" : "FAIL") << std::endl;
    EXPECT_TRUE(r.isPass) << caseId << " precision not met: MERE=" << r.mere << " MARE=" << r.mare;
}

static void ExpectIntPass(const PrecResult& r, const std::string& caseId)
{
    std::cout << "[" << caseId << "] mismatch=" << r.mismatch << " (require 0, bitwise exact) => "
              << (r.isPass ? "PASS" : "FAIL") << std::endl;
    EXPECT_TRUE(r.isPass) << caseId << " not bitwise-exact: mismatch=" << r.mismatch;
}

// 多核/多 tile 模拟平台参数（ascend910b AIV 约 48 核；ubSize 取较小值以确保大规模 shape
// 触发 ubLoop>1。CPU UT 仅验数值正确性，平台值无需与硬件完全一致，只需驱动相应切分分支）。
static constexpr int64_t SIM_CORE_NUM_MULTI = 48;
static constexpr int64_t SIM_UB_SIZE_SMALL = 192 * 1024; // ~192KB（典型 A2 UB 量级）
// 单核场景：把 coreNum 设为 1 强制单核
static constexpr int64_t SIM_CORE_NUM_SINGLE = 1;
static constexpr int64_t SIM_UB_SIZE_LARGE = 192 * 1024;
// 单核多 tile 场景：单核 + 较小 UB，使中规模 shape 在单核内多 tile
static constexpr int64_t SIM_UB_SIZE_TINY = 32 * 1024; // 强制 ubFormer 偏小 → 多 tile

// ===========================================================================
// 基础用例：fp32 核心功能直通（对齐 / 2D / 非对齐尾块）
// ===========================================================================

// float32 [256] 1D 对齐基础形状 fp32 主路径冒烟，x3=2.0
TEST_F(FusedMulAddN, L0_001_fp32_1d_aligned_256)
{
    PrecResult r = RunFloatCase<float>(DType::FP32, 256, 2.0f, SIM_CORE_NUM_SINGLE, SIM_UB_SIZE_LARGE, 1, -1.0f, 1.0f);
    ExpectFloatPass(r, "L0-001");
}

// float32 [32,64]=2048 2D 对齐形状 fp32 主路径冒烟，x3=1.5
TEST_F(FusedMulAddN, L0_002_fp32_2d_aligned_32x64)
{
    PrecResult r =
        RunFloatCase<float>(DType::FP32, 32 * 64, 1.5f, SIM_CORE_NUM_SINGLE, SIM_UB_SIZE_LARGE, 2, -1.0f, 1.0f);
    ExpectFloatPass(r, "L0-002");
}

// float32 [100] 1D 非对齐尾块(100 非 8 整数倍) fp32 验 DataCopyPad，x3=3.0
TEST_F(FusedMulAddN, L0_003_fp32_1d_unaligned_tail_100)
{
    PrecResult r = RunFloatCase<float>(DType::FP32, 100, 3.0f, SIM_CORE_NUM_SINGLE, SIM_UB_SIZE_LARGE, 3, -1.0f, 1.0f);
    ExpectFloatPass(r, "L0-003");
}

// ===========================================================================
// 多 dtype × 代表性 shape 覆盖：全 5 dtype × 多种规模。
// 边界/超大多核多 tile 已纳入（1M / 1000003 / 8195），充分覆盖各字段路径。
// ===========================================================================

// ---- 对齐基础形状 [1024]（全 5 dtype，单核）----
// fp32 对齐基础，x3=2.5，uniform[-10,10]
TEST_F(FusedMulAddN, L1_001_fp32_aligned_1024)
{
    PrecResult r =
        RunFloatCase<float>(DType::FP32, 1024, 2.5f, SIM_CORE_NUM_SINGLE, SIM_UB_SIZE_LARGE, 11, -10.0f, 10.0f);
    ExpectFloatPass(r, "L1-001");
}
// fp16 对齐基础（升 fp32 计算）
TEST_F(FusedMulAddN, L1_002_fp16_aligned_1024)
{
    PrecResult r =
        RunFloatCase<half>(DType::FP16, 1024, 2.5f, SIM_CORE_NUM_SINGLE, SIM_UB_SIZE_LARGE, 12, -10.0f, 10.0f);
    ExpectFloatPass(r, "L1-002");
}
// bf16 对齐基础（升 fp32 计算）
TEST_F(FusedMulAddN, L1_003_bf16_aligned_1024)
{
    PrecResult r =
        RunFloatCase<bfloat16_t>(DType::BF16, 1024, 2.5f, SIM_CORE_NUM_SINGLE, SIM_UB_SIZE_LARGE, 13, -10.0f, 10.0f);
    ExpectFloatPass(r, "L1-003");
}
// int32 对齐基础，x3=3，randint[-100,100]
TEST_F(FusedMulAddN, L1_004_int32_aligned_1024)
{
    PrecResult r = RunIntCase<int32_t>(DType::INT32, 1024, 3, SIM_CORE_NUM_SINGLE, SIM_UB_SIZE_LARGE, 14, -100, 100);
    ExpectIntPass(r, "L1-004");
}
// int16 对齐基础
TEST_F(FusedMulAddN, L1_005_int16_aligned_1024)
{
    PrecResult r = RunIntCase<int16_t>(DType::INT16, 1024, 3, SIM_CORE_NUM_SINGLE, SIM_UB_SIZE_LARGE, 15, -100, 100);
    ExpectIntPass(r, "L1-005");
}

// ---- 非对齐尾块 [37]（单核，DataCopyPad）----
// fp32 非对齐尾块（37 非 8 倍）
TEST_F(FusedMulAddN, L1_006_fp32_unaligned_tail_37)
{
    PrecResult r =
        RunFloatCase<float>(DType::FP32, 37, 2.0f, SIM_CORE_NUM_SINGLE, SIM_UB_SIZE_LARGE, 16, -10.0f, 10.0f);
    ExpectFloatPass(r, "L1-006");
}
// fp16 非对齐尾块（37 非 16 倍）
TEST_F(FusedMulAddN, L1_007_fp16_unaligned_tail_37)
{
    PrecResult r = RunFloatCase<half>(DType::FP16, 37, 2.0f, SIM_CORE_NUM_SINGLE, SIM_UB_SIZE_LARGE, 17, -10.0f, 10.0f);
    ExpectFloatPass(r, "L1-007");
}
// bf16 非对齐尾块（37 非 16 倍）
TEST_F(FusedMulAddN, L1_008_bf16_unaligned_tail_37)
{
    PrecResult r =
        RunFloatCase<bfloat16_t>(DType::BF16, 37, 2.0f, SIM_CORE_NUM_SINGLE, SIM_UB_SIZE_LARGE, 18, -10.0f, 10.0f);
    ExpectFloatPass(r, "L1-008");
}
// int32 非对齐尾块（37 非 8 倍）
TEST_F(FusedMulAddN, L1_009_int32_unaligned_tail_37)
{
    PrecResult r = RunIntCase<int32_t>(DType::INT32, 37, 2, SIM_CORE_NUM_SINGLE, SIM_UB_SIZE_LARGE, 19, -100, 100);
    ExpectIntPass(r, "L1-009");
}
// int16 非对齐尾块（37 非 16 倍）
TEST_F(FusedMulAddN, L1_010_int16_unaligned_tail_37)
{
    PrecResult r = RunIntCase<int16_t>(DType::INT16, 37, 2, SIM_CORE_NUM_SINGLE, SIM_UB_SIZE_LARGE, 20, -100, 100);
    ExpectIntPass(r, "L1-010");
}

// ---- 单元素 [1]（最小规模，不足一个 block）----
// fp32 单元素
TEST_F(FusedMulAddN, L1_011_fp32_single_element)
{
    PrecResult r = RunFloatCase<float>(DType::FP32, 1, 2.0f, SIM_CORE_NUM_SINGLE, SIM_UB_SIZE_LARGE, 21, -10.0f, 10.0f);
    ExpectFloatPass(r, "L1-011");
}
// fp16 单元素
TEST_F(FusedMulAddN, L1_012_fp16_single_element)
{
    PrecResult r = RunFloatCase<half>(DType::FP16, 1, 2.0f, SIM_CORE_NUM_SINGLE, SIM_UB_SIZE_LARGE, 22, -10.0f, 10.0f);
    ExpectFloatPass(r, "L1-012");
}
// int32 单元素
TEST_F(FusedMulAddN, L1_013_int32_single_element)
{
    PrecResult r = RunIntCase<int32_t>(DType::INT32, 1, 2, SIM_CORE_NUM_SINGLE, SIM_UB_SIZE_LARGE, 23, -100, 100);
    ExpectIntPass(r, "L1-013");
}

// ---- 大规模 1D [1048576]（多核 + 单核内多 tile，ubLoop>1）----
// fp32 大规模，x3=1.25，uniform[-1,1]
TEST_F(FusedMulAddN, L1_014_fp32_multicore_multitile_1M)
{
    PrecResult r =
        RunFloatCase<float>(DType::FP32, 1048576, 1.25f, SIM_CORE_NUM_MULTI, SIM_UB_SIZE_SMALL, 24, -1.0f, 1.0f);
    ExpectFloatPass(r, "L1-014");
}
// fp16 大规模
TEST_F(FusedMulAddN, L1_015_fp16_multicore_multitile_1M)
{
    PrecResult r =
        RunFloatCase<half>(DType::FP16, 1048576, 1.25f, SIM_CORE_NUM_MULTI, SIM_UB_SIZE_SMALL, 25, -1.0f, 1.0f);
    ExpectFloatPass(r, "L1-015");
}
// bf16 大规模
TEST_F(FusedMulAddN, L1_016_bf16_multicore_multitile_1M)
{
    PrecResult r =
        RunFloatCase<bfloat16_t>(DType::BF16, 1048576, 1.25f, SIM_CORE_NUM_MULTI, SIM_UB_SIZE_SMALL, 26, -1.0f, 1.0f);
    ExpectFloatPass(r, "L1-016");
}
// int32 大规模
TEST_F(FusedMulAddN, L1_017_int32_multicore_multitile_1M)
{
    PrecResult r =
        RunIntCase<int32_t>(DType::INT32, 1048576, 2, SIM_CORE_NUM_MULTI, SIM_UB_SIZE_SMALL, 27, -1000, 1000);
    ExpectIntPass(r, "L1-017");
}
// int16 大规模
TEST_F(FusedMulAddN, L1_018_int16_multicore_multitile_1M)
{
    PrecResult r = RunIntCase<int16_t>(DType::INT16, 1048576, 2, SIM_CORE_NUM_MULTI, SIM_UB_SIZE_SMALL, 28, -100, 100);
    ExpectIntPass(r, "L1-018");
}

// ---- 大规模非对齐 [1000003]（多核 + 末核尾块非对齐）----
// fp32 质数级元素
TEST_F(FusedMulAddN, L1_019_fp32_multicore_unaligned_tail_1000003)
{
    PrecResult r =
        RunFloatCase<float>(DType::FP32, 1000003, 1.5f, SIM_CORE_NUM_MULTI, SIM_UB_SIZE_SMALL, 29, -1.0f, 1.0f);
    ExpectFloatPass(r, "L1-019");
}
// fp16 质数级元素
TEST_F(FusedMulAddN, L1_020_fp16_multicore_unaligned_tail_1000003)
{
    PrecResult r =
        RunFloatCase<half>(DType::FP16, 1000003, 1.5f, SIM_CORE_NUM_MULTI, SIM_UB_SIZE_SMALL, 30, -1.0f, 1.0f);
    ExpectFloatPass(r, "L1-020");
}
// bf16 质数级元素
TEST_F(FusedMulAddN, L1_036_bf16_multicore_unaligned_tail_1000003)
{
    PrecResult r =
        RunFloatCase<bfloat16_t>(DType::BF16, 1000003, 1.5f, SIM_CORE_NUM_MULTI, SIM_UB_SIZE_SMALL, 31, -1.0f, 1.0f);
    ExpectFloatPass(r, "L1-036");
}
// int32 质数级元素
TEST_F(FusedMulAddN, L1_037_int32_multicore_unaligned_tail_1000003)
{
    PrecResult r =
        RunIntCase<int32_t>(DType::INT32, 1000003, 2, SIM_CORE_NUM_MULTI, SIM_UB_SIZE_SMALL, 32, -1000, 1000);
    ExpectIntPass(r, "L1-037");
}

// ---- 多 D 形状（逐元素与维数无关）----
// fp32 3D [8,16,32]=4096
TEST_F(FusedMulAddN, L1_021_fp32_3d_8x16x32)
{
    PrecResult r =
        RunFloatCase<float>(DType::FP32, 8 * 16 * 32, 2.0f, SIM_CORE_NUM_SINGLE, SIM_UB_SIZE_LARGE, 33, -10.0f, 10.0f);
    ExpectFloatPass(r, "L1-021");
}
// fp16 4D [4,8,8,16]=4096
TEST_F(FusedMulAddN, L1_022_fp16_4d_4x8x8x16)
{
    PrecResult r = RunFloatCase<half>(
        DType::FP16, 4 * 8 * 8 * 16, 2.0f, SIM_CORE_NUM_SINGLE, SIM_UB_SIZE_LARGE, 34, -10.0f, 10.0f);
    ExpectFloatPass(r, "L1-022");
}
// bf16 3D 非对齐 [3,5,7]=105（非16倍）
TEST_F(FusedMulAddN, L1_023_bf16_3d_unaligned_3x5x7)
{
    PrecResult r = RunFloatCase<bfloat16_t>(
        DType::BF16, 3 * 5 * 7, 2.0f, SIM_CORE_NUM_SINGLE, SIM_UB_SIZE_LARGE, 35, -10.0f, 10.0f);
    ExpectFloatPass(r, "L1-023");
}

// ---- x3 特殊值（0/1/负）----
// fp32 x3=0 退化为 y=x2
TEST_F(FusedMulAddN, L1_024_fp32_x3_zero)
{
    PrecResult r =
        RunFloatCase<float>(DType::FP32, 1024, 0.0f, SIM_CORE_NUM_SINGLE, SIM_UB_SIZE_LARGE, 36, -10.0f, 10.0f);
    ExpectFloatPass(r, "L1-024");
}
// fp32 x3=1 退化为 y=x1+x2
TEST_F(FusedMulAddN, L1_025_fp32_x3_one)
{
    PrecResult r =
        RunFloatCase<float>(DType::FP32, 1024, 1.0f, SIM_CORE_NUM_SINGLE, SIM_UB_SIZE_LARGE, 37, -10.0f, 10.0f);
    ExpectFloatPass(r, "L1-025");
}
// fp32 x3 负值
TEST_F(FusedMulAddN, L1_026_fp32_x3_negative)
{
    PrecResult r =
        RunFloatCase<float>(DType::FP32, 1024, -3.5f, SIM_CORE_NUM_SINGLE, SIM_UB_SIZE_LARGE, 38, -10.0f, 10.0f);
    ExpectFloatPass(r, "L1-026");
}
// fp16 x3=0
TEST_F(FusedMulAddN, L1_027_fp16_x3_zero)
{
    PrecResult r =
        RunFloatCase<half>(DType::FP16, 1024, 0.0f, SIM_CORE_NUM_SINGLE, SIM_UB_SIZE_LARGE, 39, -10.0f, 10.0f);
    ExpectFloatPass(r, "L1-027");
}
// fp16 x3 负值
TEST_F(FusedMulAddN, L1_028_fp16_x3_negative)
{
    PrecResult r =
        RunFloatCase<half>(DType::FP16, 1024, -2.0f, SIM_CORE_NUM_SINGLE, SIM_UB_SIZE_LARGE, 40, -10.0f, 10.0f);
    ExpectFloatPass(r, "L1-028");
}
// bf16 x3=0
TEST_F(FusedMulAddN, L1_029_bf16_x3_zero)
{
    PrecResult r =
        RunFloatCase<bfloat16_t>(DType::BF16, 1024, 0.0f, SIM_CORE_NUM_SINGLE, SIM_UB_SIZE_LARGE, 41, -10.0f, 10.0f);
    ExpectFloatPass(r, "L1-029");
}
// bf16 x3=1
TEST_F(FusedMulAddN, L1_030_bf16_x3_one)
{
    PrecResult r =
        RunFloatCase<bfloat16_t>(DType::BF16, 1024, 1.0f, SIM_CORE_NUM_SINGLE, SIM_UB_SIZE_LARGE, 42, -10.0f, 10.0f);
    ExpectFloatPass(r, "L1-030");
}
// int32 x3=0
TEST_F(FusedMulAddN, L1_031_int32_x3_zero)
{
    PrecResult r = RunIntCase<int32_t>(DType::INT32, 1024, 0, SIM_CORE_NUM_SINGLE, SIM_UB_SIZE_LARGE, 43, -100, 100);
    ExpectIntPass(r, "L1-031");
}
// int32 x3 负值
TEST_F(FusedMulAddN, L1_032_int32_x3_negative)
{
    PrecResult r = RunIntCase<int32_t>(DType::INT32, 1024, -2, SIM_CORE_NUM_SINGLE, SIM_UB_SIZE_LARGE, 44, -100, 100);
    ExpectIntPass(r, "L1-032");
}
// int16 x3=1
TEST_F(FusedMulAddN, L1_033_int16_x3_one)
{
    PrecResult r = RunIntCase<int16_t>(DType::INT16, 1024, 1, SIM_CORE_NUM_SINGLE, SIM_UB_SIZE_LARGE, 45, -100, 100);
    ExpectIntPass(r, "L1-033");
}
// int16 x3 负值
TEST_F(FusedMulAddN, L1_034_int16_x3_negative)
{
    PrecResult r = RunIntCase<int16_t>(DType::INT16, 1024, -3, SIM_CORE_NUM_SINGLE, SIM_UB_SIZE_LARGE, 46, -50, 50);
    ExpectIntPass(r, "L1-034");
}

// ---- 中规模单核多 tile + 尾块 [8195]（非对齐，单核内 ubLoop>1）----
// int16 中规模非对齐（8195 非 16 倍）
TEST_F(FusedMulAddN, L1_038_int16_single_core_multitile_tail_8195)
{
    PrecResult r = RunIntCase<int16_t>(DType::INT16, 8195, 2, SIM_CORE_NUM_SINGLE, SIM_UB_SIZE_TINY, 47, -50, 50);
    ExpectIntPass(r, "L1-038");
}
// fp16 中规模非对齐
TEST_F(FusedMulAddN, L1_039_fp16_single_core_multitile_tail_8195)
{
    PrecResult r = RunFloatCase<half>(DType::FP16, 8195, 2.0f, SIM_CORE_NUM_SINGLE, SIM_UB_SIZE_TINY, 48, -1.0f, 1.0f);
    ExpectFloatPass(r, "L1-039");
}
// fp32 中规模非对齐
TEST_F(FusedMulAddN, L1_040_fp32_single_core_multitile_tail_8195)
{
    PrecResult r = RunFloatCase<float>(DType::FP32, 8195, 2.0f, SIM_CORE_NUM_SINGLE, SIM_UB_SIZE_TINY, 49, -1.0f, 1.0f);
    ExpectFloatPass(r, "L1-040");
}

// ===========================================================================
// x3 形态等价 + 边界强化（复用同一份编译产物 + CPU golden 比对）。
// ===========================================================================

// ---- x3 形态 [1,1] vs [1] 数值等价 ----
//   相同输入/种子/标量值，分别以 x3 形态 [1]（capacity=1）与 [1,1]（capacity=2，
//   仍 1 个逻辑元素）跑 kernel，断言两次输出【逐元素严格相等】（bit-exact），
//   证明 x3 张量形态对计算结果无影响。
TEST_F(FusedMulAddN, L1_035_fp32_x3_shape_1x1_equiv_to_1)
{
    const int64_t totalNum = 1024;
    const float x3Value = 2.0f;
    const unsigned seed = 135; // 两次必须同种子，保证 x1/x2 数据完全一致
    // 末参为 x3Cap：1 表示 x3 形态 [1]，2 表示 x3 形态 [1,1]（仍 1 个逻辑元素）。
    std::vector<float> yShape1 =
        RunFp32Capture(totalNum, x3Value, SIM_CORE_NUM_SINGLE, SIM_UB_SIZE_LARGE, seed, -10.0f, 10.0f, 1);
    std::vector<float> yShape1x1 =
        RunFp32Capture(totalNum, x3Value, SIM_CORE_NUM_SINGLE, SIM_UB_SIZE_LARGE, seed, -10.0f, 10.0f, 2);

    ASSERT_EQ(yShape1.size(), yShape1x1.size());
    int64_t mismatch = 0;
    for (size_t i = 0; i < yShape1.size(); ++i) {
        if (yShape1[i] != yShape1x1[i]) {
            ++mismatch;
        }
    }
    std::cout << "[L1-035] x3 shape [1,1] vs [1] mismatch=" << mismatch << " (require 0, bit-exact) => "
              << (mismatch == 0 ? "PASS" : "FAIL") << std::endl;
    EXPECT_EQ(mismatch, 0) << "L1-035 x3 shape [1,1] 与 [1] 输出不一致, mismatch=" << mismatch;

    // 同时对 [1,1] 形态做一次正确性 golden 比对（避免两形态同样错误而误判等价）。
    PrecResult r = RunFloatCase<float>(
        DType::FP32, totalNum, x3Value, SIM_CORE_NUM_SINGLE, SIM_UB_SIZE_LARGE, seed, -10.0f, 10.0f);
    ExpectFloatPass(r, "L1-035(golden)");
}

// ---- 边界强化 (a) 极端尾块 tail=1（多核，末核仅 1 元素）----
//   构造 totalNum=9, coreNum=2：perCore=5 → blockFormer=ceilAlign(5,8)=8, blockNum=2,
//   blockTail=9-8=1 → 末核处理 1 元素（sub-block DataCopyPad，ubTailOfTailBlock=1）。
TEST_F(FusedMulAddN, B1_boundary_tail1_fp32_multicore_9)
{
    // 第 4 参为 coreNum=2（多核），末核仅 1 元素。
    PrecResult r = RunFloatCase<float>(DType::FP32, 9, 2.5f, 2, SIM_UB_SIZE_LARGE, 101, -10.0f, 10.0f);
    ExpectFloatPass(r, "B1-tail1-fp32");
}
TEST_F(FusedMulAddN, B1_boundary_tail1_fp16_multicore_17)
{
    // fp16 block=16：totalNum=17, coreNum=2 → blockFormer=16, blockTail=1（末核 1 元素）。第 4 参为 coreNum=2。
    PrecResult r = RunFloatCase<half>(DType::FP16, 17, 2.5f, 2, SIM_UB_SIZE_LARGE, 102, -10.0f, 10.0f);
    ExpectFloatPass(r, "B1-tail1-fp16");
}
TEST_F(FusedMulAddN, B1_boundary_tail1_int16_multicore_17)
{
    // 第 4 参为 coreNum=2（多核），末核仅 1 元素。
    PrecResult r = RunIntCase<int16_t>(DType::INT16, 17, 2, 2, SIM_UB_SIZE_LARGE, 103, -50, 50);
    ExpectIntPass(r, "B1-tail1-int16");
}

// ---- 边界强化 (b) 32B 对齐粒度边界（恰一个 block / block±1）----
//   fp32/int32：block=8 元素；fp16/bf16/int16：block=16 元素。验证恰对齐、刚过一个
//   block（+1，触发第二个 tile 的 1 元素尾）、差一元素（-1，sub-block 尾块）。
// fp32 block=8 边界
TEST_F(FusedMulAddN, B1_align_fp32_exactly_8)
{
    PrecResult r =
        RunFloatCase<float>(DType::FP32, 8, 2.0f, SIM_CORE_NUM_SINGLE, SIM_UB_SIZE_LARGE, 110, -10.0f, 10.0f);
    ExpectFloatPass(r, "B1-align-fp32-8");
}
TEST_F(FusedMulAddN, B1_align_fp32_7)
{
    PrecResult r =
        RunFloatCase<float>(DType::FP32, 7, 2.0f, SIM_CORE_NUM_SINGLE, SIM_UB_SIZE_LARGE, 111, -10.0f, 10.0f);
    ExpectFloatPass(r, "B1-align-fp32-7");
}
TEST_F(FusedMulAddN, B1_align_fp32_9)
{
    PrecResult r =
        RunFloatCase<float>(DType::FP32, 9, 2.0f, SIM_CORE_NUM_SINGLE, SIM_UB_SIZE_LARGE, 112, -10.0f, 10.0f);
    ExpectFloatPass(r, "B1-align-fp32-9");
}
TEST_F(FusedMulAddN, B1_align_int32_exactly_8)
{
    PrecResult r = RunIntCase<int32_t>(DType::INT32, 8, 3, SIM_CORE_NUM_SINGLE, SIM_UB_SIZE_LARGE, 113, -100, 100);
    ExpectIntPass(r, "B1-align-int32-8");
}
// fp16/bf16/int16 block=16 边界
TEST_F(FusedMulAddN, B1_align_fp16_exactly_16)
{
    PrecResult r =
        RunFloatCase<half>(DType::FP16, 16, 2.0f, SIM_CORE_NUM_SINGLE, SIM_UB_SIZE_LARGE, 114, -10.0f, 10.0f);
    ExpectFloatPass(r, "B1-align-fp16-16");
}
TEST_F(FusedMulAddN, B1_align_fp16_15)
{
    PrecResult r =
        RunFloatCase<half>(DType::FP16, 15, 2.0f, SIM_CORE_NUM_SINGLE, SIM_UB_SIZE_LARGE, 115, -10.0f, 10.0f);
    ExpectFloatPass(r, "B1-align-fp16-15");
}
TEST_F(FusedMulAddN, B1_align_fp16_17)
{
    PrecResult r =
        RunFloatCase<half>(DType::FP16, 17, 2.0f, SIM_CORE_NUM_SINGLE, SIM_UB_SIZE_LARGE, 116, -10.0f, 10.0f);
    ExpectFloatPass(r, "B1-align-fp16-17");
}
TEST_F(FusedMulAddN, B1_align_bf16_exactly_16)
{
    PrecResult r =
        RunFloatCase<bfloat16_t>(DType::BF16, 16, 2.0f, SIM_CORE_NUM_SINGLE, SIM_UB_SIZE_LARGE, 117, -10.0f, 10.0f);
    ExpectFloatPass(r, "B1-align-bf16-16");
}
TEST_F(FusedMulAddN, B1_align_int16_exactly_16)
{
    PrecResult r = RunIntCase<int16_t>(DType::INT16, 16, 3, SIM_CORE_NUM_SINGLE, SIM_UB_SIZE_LARGE, 118, -100, 100);
    ExpectIntPass(r, "B1-align-int16-16");
}
TEST_F(FusedMulAddN, B1_align_int16_15)
{
    PrecResult r = RunIntCase<int16_t>(DType::INT16, 15, 3, SIM_CORE_NUM_SINGLE, SIM_UB_SIZE_LARGE, 119, -100, 100);
    ExpectIntPass(r, "B1-align-int16-15");
}

// ---- 边界强化 (c) 单元素（int16/bf16）----
TEST_F(FusedMulAddN, B1_single_element_bf16)
{
    PrecResult r =
        RunFloatCase<bfloat16_t>(DType::BF16, 1, 2.0f, SIM_CORE_NUM_SINGLE, SIM_UB_SIZE_LARGE, 120, -10.0f, 10.0f);
    ExpectFloatPass(r, "B1-single-bf16");
}
TEST_F(FusedMulAddN, B1_single_element_int16)
{
    PrecResult r = RunIntCase<int16_t>(DType::INT16, 1, 2, SIM_CORE_NUM_SINGLE, SIM_UB_SIZE_LARGE, 121, -50, 50);
    ExpectIntPass(r, "B1-single-int16");
}

// ---- 边界强化 (d) 空 tensor totalNum=0（如框架支持）----
//   BuildTiling(totalNum=0) → blockNum=1, ubFormer=block, 所有 ub 循环参数=0；
//   kernel Process 循环 0 次（不读不写），仅验证"无 compute 路径不崩溃"。
//   0 元素无数值可比对，断言 size==0 即视为通过（验证 kernel 优雅处理空规模）。
TEST_F(FusedMulAddN, B1_empty_tensor_fp32)
{
    UtTilingPlan plan = BuildTiling(0, DType::FP32, SIM_CORE_NUM_SINGLE, SIM_UB_SIZE_LARGE);
    // 分配至少 1 个 block 的 GM，避免 0 字节分配；kernel 不会触达任何元素。
    size_t oneBlock = 32;
    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(oneBlock);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(oneBlock);
    uint8_t* x3 = (uint8_t*)AscendC::GmAlloc(oneBlock);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(oneBlock);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(static_cast<size_t>(16 * 1024 * 1024));
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(FusedMulAddNTilingDataUT));
    reinterpret_cast<float*>(x3)[0] = 2.0f;
    std::memcpy(tiling, &plan.data, sizeof(FusedMulAddNTilingDataUT));

    ICPU_SET_TILING_KEY(DTypeTilingKey(DType::FP32));
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(fused_mul_add_n, plan.blockNum, x1, x2, x3, y, workspace, tiling);

    std::cout << "[B1-empty-fp32] totalNum=0 kernel returned without crash => PASS" << std::endl;
    SUCCEED() << "empty tensor (totalNum=0) handled gracefully";

    AscendC::GmFree((void*)x1);
    AscendC::GmFree((void*)x2);
    AscendC::GmFree((void*)x3);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}
