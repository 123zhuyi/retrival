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
 * \file test_aclnn_sort_with_index.cpp
 * \brief SortWithIndex (ascend910b) C++ ST test — full-coverage suite (4 dtype groups).
 *
 * ============================================================================
 * 覆盖范围：
 *
 *   dtype 覆盖（910B 首版 4 组，index 仅 int32；不覆盖 int64-index）：
 *     - float16  + int32  （half 直接 Sort）
 *     - float32  + int32  （float 直接 Sort）
 *     - bfloat16 + int32  （Cast f32<->bf16 通路）
 *     - int32    + int32  （Cast f32<->int32 通路，限 |x|<=2^24）
 *
 *   shape 覆盖：rank 0–8（标量 / 1D / 多维），单 tile 内多轴长（轴长 <= 单 tile 上限，
 *              fp16~3008 / fp32~2816 / bf16~2816 / int32~2560），多行分核。
 *
 *   属性覆盖：descending ∈ {false,true}；stable ∈ {false,true}；axis ∈ {-1, rank-1}。
 *
 *   boundary：rank0 标量、轴长=1、空 tensor（轴长0）、x≠index shape 报错、axis 非最后一维报错。
 *
 *   extreme（910B 语义）：
 *     - NaN 升序落「开头」（rank0），降序也落开头。
 *       ★ 关键：与 torch「NaN 末尾」约定不同！golden 按「910B NaN 开头」生成。
 *     - NaN 位型按 isnan 比较：升序 Muls(-1) 翻 NaN 符号位（0x7fff vs 0x7e00），
 *       bit 不一致但仍是 NaN —— 置换/bitwise 不变量对 NaN↔NaN 视为相等（isnan），不按 memcmp。
 *     - +Inf 升序末尾（+Inf 是有限/Inf 区最大，排末尾）；-Inf 升序开头。
 *     - 全零、全相等（stable 时 sorted_index 按原始位置升序）、±0（视为相等 ties）。
 *
 * ----------------------------------------------------------------------------
 * Golden（910B 语义；非 torch）：
 *   per-slice 沿最后一维：
 *     1) NaN 元素稳定（保持原始位置升序）放序列「开头」；
 *     2) 其余有限值/±Inf 按 ascending(descending=false) 或 descending(=true) 稳定排序；
 *     3) y[k] = x[perm[k]]，sorted_index[k] = index[perm[k]]（同一 permutation 跟随）。
 *   stable=true/false 在 golden 一律取 std::stable_sort（tie-break 原始位置升序）；NPU stable=false
 *   若 tie-break 不同，仍由「permutation 不变量 + value bitwise(isnan-aware)」兜底，indices 精确比对
 *   在无 ties 输入下成立（本套正向用例 random_uniform 选无重复值域，ties 仅在 all_same/all_zero
 *   等 extreme 用例，且这些用例只断言 permutation 不变量，不强制 index 顺序）。
 *
 * 精度：全 dtype rtol=atol=0，bitwise_equal。
 *   values bitwise（有限值精确，NaN 行 isnan 比较）；indices 精确。重排算子无 ULP 误差。
 *
 * Mock 模式（-DUSE_MOCK_ACLNN）：CPU golden 自验证（golden 充当 NPU 输出，验证比对/不变量闭环），无 NPU 依赖。
 *                              全 4 组 dtype + 边界 + extreme 均在 Mock 下跑。
 * Real 模式（默认）：调用 aclnnSortWithIndex 两段式接口，NPU 输出 vs CPU golden bitwise(isnan-aware)/精确比对。
 * ============================================================================
 */

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <limits>
#include <functional>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <numeric>

#ifndef USE_MOCK_ACLNN
#include "acl/acl.h"
#include "aclnn_sort_with_index.h"
#endif

// ============================================================================
// 宏定义
// ============================================================================
#define LOG_PRINT(fmt, ...) printf(fmt "\n", ##__VA_ARGS__)

// ============================================================================
// 形状辅助
// ============================================================================

static int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t size = 1;
    for (auto dim : shape) {
        size *= dim;
    }
    return size;
}

// 排序轴长度（最后一维）。rank=0（标量）视为单元素切片，长度=1。
static int64_t GetSortAxisLen(const std::vector<int64_t>& shape)
{
    if (shape.empty()) {
        return 1;
    }
    return shape.back();
}

// 行数 = 非排序维元素个数 = totalSize / axisLen。
static int64_t GetRowCount(const std::vector<int64_t>& shape)
{
    int64_t total = GetShapeSize(shape);
    int64_t axisLen = GetSortAxisLen(shape);
    if (axisLen == 0) {
        return 0;
    }
    return total / axisLen;
}

static std::string ShapeToStr(const std::vector<int64_t>& shape)
{
    std::string s = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        s += std::to_string(shape[i]);
        if (i + 1 < shape.size()) {
            s += ",";
        }
    }
    s += "]";
    return s;
}

// ============================================================================
// float16 <-> float 转换（golden 计算用；NPU 侧用 ACL_FLOAT16 直接存 uint16 位）
// ============================================================================

static float Fp16ToFloat(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000U) << 16;
    uint32_t exp = (h >> 10) & 0x1FU;
    uint32_t mant = h & 0x3FFU;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;  // ±0
        } else {
            // subnormal：规格化
            exp = 1;
            while ((mant & 0x400U) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FFU;
            f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
    } else if (exp == 0x1FU) {
        // Inf / NaN
        f = sign | 0x7F800000U | (mant << 13);
    } else {
        f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
}

static uint16_t FloatToFp16(float value)
{
    uint32_t f;
    std::memcpy(&f, &value, sizeof(f));
    uint32_t sign = (f >> 16) & 0x8000U;
    int32_t exp = (int32_t)((f >> 23) & 0xFFU) - 127 + 15;
    uint32_t mant = f & 0x7FFFFFU;

    if (((f >> 23) & 0xFFU) == 0xFFU) {
        // Inf / NaN
        uint32_t h = sign | 0x7C00U | (mant ? 0x200U : 0U);
        return (uint16_t)h;
    }
    // NOTE: compare against a SIGNED literal (31), NOT 0x1FU. exp is a signed int that is negative for
    // zero/subnormal inputs (e.g. 0.0f -> exp = -112). With an unsigned 0x1FU the signed exp would be
    // promoted to a huge unsigned value and wrongly take the overflow->Inf branch.
    if (exp >= 31) {
        return (uint16_t)(sign | 0x7C00U);  // 溢出 -> Inf
    }
    if (exp <= 0) {
        if (exp < -10) {
            return (uint16_t)sign;  // 下溢 -> ±0
        }
        mant |= 0x800000U;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t h = sign | (mant >> shift);
        return (uint16_t)h;
    }
    uint32_t h = sign | ((uint32_t)exp << 10) | (mant >> 13);
    return (uint16_t)h;
}

// ---- bf16 <-> float（golden / 输入构造用；NPU 侧 ACL_BF16 直接存 uint16 位）----
// bf16 = float 高 16 位（截断尾数）。构造正向用例时只用「float 高16位即原值」的可精确表示数。
static float Bf16ToFloat(uint16_t h)
{
    uint32_t f = static_cast<uint32_t>(h) << 16;
    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
}

static uint16_t FloatToBf16Truncate(float value)
{
    uint32_t f;
    std::memcpy(&f, &value, sizeof(f));
    return static_cast<uint16_t>(f >> 16);  // 截断（输入已保证低16位为0，无舍入误差）
}

// ============================================================================
// 排序比较语义（910B：NaN 升序/降序均落「开头」）
//
//   910B 升序 = Muls(-1) + 硬件降序 Sort + Muls(-1)（不反转名次）：
//     硬件降序 Sort 把 NaN 放 rank0 → 升序还原后 NaN 仍在 rank0（开头）。
//   故 NaN 在升序、降序下「都落开头」。
//
//   有限值/±Inf 区间：升序单调不减、降序单调不增；+Inf > 有限 > -Inf；±0 视为相等 ties。
// ============================================================================

// 有限值/±Inf 的严格弱序（不含 NaN，调用方已保证两侧非 NaN）。返回 a < b。
static bool FiniteLess(float a, float b)
{
    return a < b;  // ±0 互相 a<b 与 b<a 均 false → 视为相等 ties
}

// ============================================================================
// 通用 CPU Golden（910B 语义）：
//   1) NaN 元素稳定放序列开头（保持原始位置升序）；
//   2) 其余有限值/±Inf 按 ascending/descending 稳定排序；
//   3) value 位 / index 位按同一 permutation 跟随。
// value 以原始位宽存储（fp16/bf16: uint16；fp32: uint32；int32: int32）。
// projectKey 把存储位投影为排序键 float。
// ============================================================================

template <typename ValueStore, typename IndexStore>
static void ComputeGolden910b(const std::vector<ValueStore>& x, const std::vector<IndexStore>& index,
                              const std::vector<int64_t>& shape, bool descending,
                              const std::function<float(ValueStore)>& projectKey,
                              std::vector<ValueStore>& y, std::vector<IndexStore>& sortedIndex)
{
    int64_t total = GetShapeSize(shape);
    int64_t axisLen = GetSortAxisLen(shape);
    int64_t rows = GetRowCount(shape);
    y.assign(static_cast<size_t>(total), ValueStore{});
    sortedIndex.assign(static_cast<size_t>(total), IndexStore{});
    if (total == 0 || axisLen == 0) {
        return;  // 空 tensor：返回空
    }
    for (int64_t r = 0; r < rows; ++r) {
        int64_t base = r * axisLen;
        std::vector<int64_t> perm(static_cast<size_t>(axisLen));
        std::iota(perm.begin(), perm.end(), 0);
        // std::stable_sort：NaN 先于一切（落开头，原始位置升序）；非 NaN 按 finite order。
        std::stable_sort(perm.begin(), perm.end(), [&](int64_t ia, int64_t ib) {
            float va = projectKey(x[static_cast<size_t>(base + ia)]);
            float vb = projectKey(x[static_cast<size_t>(base + ib)]);
            bool na = std::isnan(va);
            bool nb = std::isnan(vb);
            if (na || nb) {
                // NaN 落开头：a 排在 b 前（a<b）当且仅当 a 是 NaN 且 b 非 NaN。
                // 两侧均 NaN → 返回 false（相等，stable 保持原始位置升序）。
                return na && !nb;
            }
            return descending ? FiniteLess(vb, va) : FiniteLess(va, vb);
        });
        for (int64_t k = 0; k < axisLen; ++k) {
            int64_t src = base + perm[static_cast<size_t>(k)];
            y[static_cast<size_t>(base + k)] = x[static_cast<size_t>(src)];
            sortedIndex[static_cast<size_t>(base + k)] = index[static_cast<size_t>(src)];
        }
    }
}

// ============================================================================
// 精度比对（重排算子：values bitwise + NaN 行 isnan、indices 精确；rtol=atol=0）
//
//   NaN 行不按 memcmp（升序 Muls(-1) 翻 NaN 符号位，bit 不一致但仍是 NaN）。
//   value 比对：projectKey 投影后两侧均 NaN → 视为相等；否则 bitwise(memcmp)。
// ============================================================================

template <typename ValueStore>
static bool CompareValuesIsnanAware(const std::vector<ValueStore>& golden, const std::vector<ValueStore>& actual,
                                    const std::function<float(ValueStore)>& projectKey)
{
    if (golden.size() != actual.size()) {
        LOG_PRINT("  [FAIL] values size 不一致: golden=%zu actual=%zu", golden.size(), actual.size());
        return false;
    }
    int mismatch = 0;
    for (size_t i = 0; i < golden.size(); ++i) {
        bool gn = std::isnan(projectKey(golden[i]));
        bool an = std::isnan(projectKey(actual[i]));
        bool eq;
        if (gn || an) {
            eq = gn && an;  // NaN 行：两侧均 NaN 视为相等（isnan，不比 bit）
        } else {
            eq = (std::memcmp(&golden[i], &actual[i], sizeof(ValueStore)) == 0);  // 有限值 bitwise
        }
        if (!eq) {
            mismatch++;
            if (mismatch <= 5) {
                LOG_PRINT("  values 不匹配 [%zu]: golden_key=%.6f(nan=%d) actual_key=%.6f(nan=%d)", i,
                          projectKey(golden[i]), (int)gn, projectKey(actual[i]), (int)an);
            }
        }
    }
    if (mismatch == 0) {
        LOG_PRINT("  [PASS] values bitwise(isnan-aware) 一致（%zu 个元素）", golden.size());
        return true;
    }
    LOG_PRINT("  [FAIL] values 发现 %d 个不匹配", mismatch);
    return false;
}

template <typename IndexStore>
static bool CompareIndicesExact(const std::vector<IndexStore>& golden, const std::vector<IndexStore>& actual)
{
    if (golden.size() != actual.size()) {
        LOG_PRINT("  [FAIL] indices size 不一致: golden=%zu actual=%zu", golden.size(), actual.size());
        return false;
    }
    int mismatch = 0;
    for (size_t i = 0; i < golden.size(); ++i) {
        if (golden[i] != actual[i]) {
            mismatch++;
            if (mismatch <= 5) {
                LOG_PRINT("  indices 不匹配 [%zu]: golden=%lld actual=%lld", i,
                          (long long)golden[i], (long long)actual[i]);
            }
        }
    }
    if (mismatch == 0) {
        LOG_PRINT("  [PASS] indices 精确一致（%zu 个元素）", golden.size());
        return true;
    }
    LOG_PRINT("  [FAIL] indices 发现 %d 个不匹配", mismatch);
    return false;
}

// ============================================================================
// 结构性不变量：
//   - y 是 x 沿最后一维的置换（multiset，NaN↔NaN 视为相等 isnan，非 NaN 按 bit）
//   - sorted_index 是 index 沿最后一维的置换
//   - y 沿最后一维单调（升序不减 / 降序不增）；NaN 排除在单调性校验之外（NaN 落开头）
// ============================================================================

// NaN-aware multiset 比较 key：把 NaN 归一为同一个 canonical key，使 NaN↔NaN 等价。
template <typename ValueStore>
static uint64_t MultisetKey(const ValueStore& v, const std::function<float(ValueStore)>& projectKey)
{
    float f = projectKey(v);
    if (std::isnan(f)) {
        return 0xFFFFFFFFFFFFFFFFULL;  // 所有 NaN 归一为同一 key（isnan 等价）
    }
    // 非 NaN：用原始存储位（bitwise 区分有限值，含 ±0 区分位型）。
    uint64_t bits = 0;
    std::memcpy(&bits, &v, sizeof(ValueStore));
    return bits;  // 高位为 0（ValueStore <= 8 字节）；NaN 的 all-ones key 不会与有限值碰撞
}

template <typename ValueStore, typename IndexStore>
static bool CheckInvariants910b(const std::vector<ValueStore>& x, const std::vector<IndexStore>& index,
                                const std::vector<ValueStore>& y, const std::vector<IndexStore>& si,
                                const std::vector<int64_t>& shape, bool descending,
                                const std::function<float(ValueStore)>& projectKey)
{
    int64_t axisLen = GetSortAxisLen(shape);
    int64_t rows = GetRowCount(shape);
    if (axisLen == 0 || rows == 0) {
        // 空 tensor：x/y、index/si 均空
        bool ok = (x.size() == y.size()) && (index.size() == si.size());
        if (!ok) {
            LOG_PRINT("  [FAIL] 空 tensor 输出非空");
        }
        return ok;
    }
    for (int64_t r = 0; r < rows; ++r) {
        int64_t base = r * axisLen;
        // value multiset（NaN↔NaN isnan 等价）
        std::vector<uint64_t> xa(static_cast<size_t>(axisLen));
        std::vector<uint64_t> ya(static_cast<size_t>(axisLen));
        for (int64_t k = 0; k < axisLen; ++k) {
            xa[static_cast<size_t>(k)] = MultisetKey<ValueStore>(x[static_cast<size_t>(base + k)], projectKey);
            ya[static_cast<size_t>(k)] = MultisetKey<ValueStore>(y[static_cast<size_t>(base + k)], projectKey);
        }
        std::sort(xa.begin(), xa.end());
        std::sort(ya.begin(), ya.end());
        if (xa != ya) {
            LOG_PRINT("  [FAIL] 不变量 y_is_permutation_of_x 失败（row=%ld）", static_cast<long>(r));
            return false;
        }
        // index multiset
        std::vector<IndexStore> ia(index.begin() + base, index.begin() + base + axisLen);
        std::vector<IndexStore> sa(si.begin() + base, si.begin() + base + axisLen);
        std::sort(ia.begin(), ia.end());
        std::sort(sa.begin(), sa.end());
        if (ia != sa) {
            LOG_PRINT("  [FAIL] 不变量 sorted_index_is_permutation_of_index 失败（row=%ld）", static_cast<long>(r));
            return false;
        }
        // 单调：跳过 NaN 元素（NaN 落开头，不计入单调性校验）。
        float prev = 0.0f;
        bool havePrev = false;
        for (int64_t k = 0; k < axisLen; ++k) {
            float cur = projectKey(y[static_cast<size_t>(base + k)]);
            if (std::isnan(cur)) {
                continue;  // NaN 排除在单调性之外
            }
            if (havePrev) {
                bool bad = descending ? FiniteLess(prev, cur) : FiniteLess(cur, prev);
                if (bad) {
                    LOG_PRINT("  [FAIL] 不变量 y_sorted_monotone 失败（row=%ld k=%ld prev=%.6f cur=%.6f）",
                              static_cast<long>(r), static_cast<long>(k), prev, cur);
                    return false;
                }
            }
            prev = cur;
            havePrev = true;
        }
    }
    LOG_PRINT("  [PASS] 3 个结构性不变量成立");
    return true;
}

// ============================================================================
// 数据填充模式（extreme / boundary）
// ============================================================================

enum class DataPattern {
    RANDOM_UNIQUE,       // 无重复有限值（正向精度用例：indices 可精确比对）
    SCALAR,              // 单元素
    INJECT_NAN,          // 含 1 个 NaN（升序落开头）
    SINGLE_POS_INF,      // 含 1 个 +Inf（升序落末尾）
    SINGLE_NEG_INF,      // 含 1 个 -Inf（升序落开头）
    NAN_INF_COMBO,       // NaN + ±Inf 综合（验证 NaN 开头 / -Inf 开头 / +Inf 末尾）
    ALL_ZERO,            // 全 0（ties，stable index 升序）
    ALL_SAME,            // 全相等（ties，stable index 升序）
    SIGNED_ZERO_MIX,     // ±0 混合（视为相等 ties）
    WITH_TIES,           // 含重复值（确定性 / stable 验证）
};

// ============================================================================
// 通用用例描述（覆盖全 4 dtype + 边界 + extreme）
// ============================================================================

struct CaseSpec {
    const char* case_id;
    const char* value_dtype;  // "float16" / "float32" / "bfloat16" / "int32"
    std::vector<int64_t> shape;
    int64_t axis;
    bool descending;
    bool stable;
    DataPattern pattern;
    bool is_empty;
    const char* note;
};

// ============================================================================
// 输入构造：按 value_dtype 与 pattern 生成存储位 value + int32 index。
//   value 统一以「位」存进 uint64 缓冲再投影；为简化 Mock/Real 共享，按 dtype 分派。
// ============================================================================

// 确定性 LCG（保证 Mock/Real 可复现，不依赖随机种子）。
struct Lcg {
    uint32_t state;
    explicit Lcg(uint32_t seed) : state(seed) {}
    uint32_t next()
    {
        state = state * 1664525U + 1013904223U;
        return state;
    }
    // [-range/2, range/2) 的整数
    int32_t nextInt(uint32_t range)
    {
        return static_cast<int32_t>((next() >> 9) % range) - static_cast<int32_t>(range / 2);
    }
};

// 生成一行的「float 语义值」序列（按 pattern），随后由各 dtype 量化为存储位。
// 返回 float 值，且对 RANDOM_UNIQUE/WITH_TIES 保证落在该 dtype 可精确表示的值域。
static void BuildRowFloats(const std::string& dtype, DataPattern pattern, int64_t axisLen, int64_t rowSeed,
                           std::vector<float>& out)
{
    out.assign(static_cast<size_t>(axisLen), 0.0f);
    if (axisLen == 0) {
        return;
    }
    Lcg lcg(static_cast<uint32_t>(0x9E3779B9U ^ (rowSeed * 2654435761U)));
    float nanv = std::numeric_limits<float>::quiet_NaN();
    float pinf = std::numeric_limits<float>::infinity();
    float ninf = -std::numeric_limits<float>::infinity();

    // 该 dtype 可精确表示且【严格唯一】的有限值生成器（关键：大 N 下也无 ties，indices 可精确比对）。
    //
    //   ★ 浮点 dtype（fp16/bf16）：整数线性映射在大量级处会被舍入致碰撞（bf16 仅 7 位尾数，|v|>256
    //     即非全部整数可精确表示 → (k-N/2) 大 N 时相邻 k 截断成同一 bf16 → ties，破坏 indices 精确）。
    //     故改为【直接走存储位空间】：正有限 fp16/bf16 的 uint16 位型随 bit 值单调递增。取 base 为一个
    //     正规小正数的位型，令第 k 个元素位型 = base + k（再投影回 float），保证位型互不相同 →
    //     bf16/fp16 值【严格唯一且 round-trip 精确】（QuantizeOne 还原即原位型）。N 上界 < 该 binade 余量，
    //     本套最大 N=3008，base 取离上溢足够远，无 Inf/NaN 越界。
    //   ★ fp32：float 尾数 23 位，(k-N/2) 整数 |v|<=2048 远在精确范围，唯一。
    //   ★ int32：(k-N/2) 整数 |x|<=2^24 恒成立（N<=3008），Cast RINT 精确且唯一。
    auto genFinite = [&](int64_t k) -> float {
        (void)lcg;
        if (dtype == "float16") {
            // fp16 正规正数位型 0x3C00=1.0；+k 单调递增、互异（k<N<=3008 不越到 Inf 0x7C00）。
            uint16_t bits = static_cast<uint16_t>(0x3C00U + static_cast<uint16_t>(k));
            return Fp16ToFloat(bits);
        } else if (dtype == "bfloat16") {
            // bf16 正规正数位型 0x3F80=1.0；+k 单调递增、互异（0x3F80+3008=0x4B40 << Inf 0x7F80）。
            uint16_t bits = static_cast<uint16_t>(0x3F80U + static_cast<uint16_t>(k));
            return Bf16ToFloat(bits);
        }
        // fp32 / int32：唯一整数（含负，覆盖有符号）。
        int64_t v = k - axisLen / 2;
        return static_cast<float>(v);
    };

    switch (pattern) {
        case DataPattern::SCALAR:
            out[0] = (dtype == "int32") ? 42.0f : 13.5f;
            if (dtype == "int32") out[0] = 42.0f;
            break;
        case DataPattern::ALL_ZERO:
            for (int64_t k = 0; k < axisLen; ++k) out[static_cast<size_t>(k)] = 0.0f;
            break;
        case DataPattern::ALL_SAME:
            for (int64_t k = 0; k < axisLen; ++k) out[static_cast<size_t>(k)] = (dtype == "int32") ? 5.0f : 1.0f;
            break;
        case DataPattern::SIGNED_ZERO_MIX:
            for (int64_t k = 0; k < axisLen; ++k) out[static_cast<size_t>(k)] = (k % 2 == 0) ? 0.0f : -0.0f;
            break;
        case DataPattern::WITH_TIES:
            // 成对重复值（ties），int32/浮点通用。
            for (int64_t k = 0; k < axisLen; ++k) {
                out[static_cast<size_t>(k)] = static_cast<float>((k / 2) + 1);
            }
            break;
        case DataPattern::INJECT_NAN:
            for (int64_t k = 0; k < axisLen; ++k) out[static_cast<size_t>(k)] = genFinite(k);
            out[static_cast<size_t>(axisLen / 2)] = nanv;  // 中间注入 1 个 NaN
            break;
        case DataPattern::SINGLE_POS_INF:
            for (int64_t k = 0; k < axisLen; ++k) out[static_cast<size_t>(k)] = genFinite(k);
            out[static_cast<size_t>(axisLen / 3)] = pinf;
            break;
        case DataPattern::SINGLE_NEG_INF:
            for (int64_t k = 0; k < axisLen; ++k) out[static_cast<size_t>(k)] = genFinite(k);
            out[static_cast<size_t>(axisLen * 2 / 3)] = ninf;
            break;
        case DataPattern::NAN_INF_COMBO:
            for (int64_t k = 0; k < axisLen; ++k) out[static_cast<size_t>(k)] = genFinite(k);
            if (axisLen >= 4) {
                out[0] = ninf;
                out[static_cast<size_t>(axisLen / 3)] = pinf;
                out[static_cast<size_t>(axisLen / 2)] = nanv;
                out[static_cast<size_t>(axisLen - 1)] = -0.0f;
            }
            break;
        case DataPattern::RANDOM_UNIQUE:
        default:
            for (int64_t k = 0; k < axisLen; ++k) out[static_cast<size_t>(k)] = genFinite(k);
            break;
    }
}

// 量化 float 语义值到各 dtype 的存储位，写入 uint64 缓冲（统一容器）。
// 同时记录每元素的 projectKey 用 float（与 NPU bitwise 比对一致）。
template <typename ValueStore>
static ValueStore QuantizeOne(const std::string& dtype, float f);

template <>
uint16_t QuantizeOne<uint16_t>(const std::string& dtype, float f)
{
    if (dtype == "bfloat16") {
        return FloatToBf16Truncate(f);
    }
    return FloatToFp16(f);  // float16
}
template <>
uint32_t QuantizeOne<uint32_t>(const std::string& dtype, float f)
{
    uint32_t b;
    std::memcpy(&b, &f, sizeof(b));
    return b;  // float32
}
template <>
int32_t QuantizeOne<int32_t>(const std::string& dtype, float f)
{
    (void)dtype;
    return static_cast<int32_t>(std::lround(f));  // int32 value
}

// projectKey（存储位 -> 排序键 float），按 dtype。
template <typename ValueStore>
static std::function<float(ValueStore)> MakeProjectKey(const std::string& dtype);

template <>
std::function<float(uint16_t)> MakeProjectKey<uint16_t>(const std::string& dtype)
{
    if (dtype == "bfloat16") {
        return [](uint16_t b) { return Bf16ToFloat(b); };
    }
    return [](uint16_t b) { return Fp16ToFloat(b); };
}
template <>
std::function<float(uint32_t)> MakeProjectKey<uint32_t>(const std::string& /*dtype*/)
{
    return [](uint32_t b) { float f; std::memcpy(&f, &b, sizeof(f)); return f; };
}
template <>
std::function<float(int32_t)> MakeProjectKey<int32_t>(const std::string& /*dtype*/)
{
    return [](int32_t v) { return static_cast<float>(v); };
}

// 构造一个用例的 value 存储位 + int32 index（每行 0..N-1）。
template <typename ValueStore>
static void BuildCaseInputs(const CaseSpec& cs, std::vector<ValueStore>& x, std::vector<int32_t>& index)
{
    int64_t total = GetShapeSize(cs.shape);
    int64_t axisLen = GetSortAxisLen(cs.shape);
    int64_t rows = GetRowCount(cs.shape);
    std::string dtype = cs.value_dtype;
    x.assign(static_cast<size_t>(total), ValueStore{});
    index.assign(static_cast<size_t>(total), 0);
    if (total == 0 || axisLen == 0) {
        return;
    }
    for (int64_t r = 0; r < rows; ++r) {
        int64_t base = r * axisLen;
        std::vector<float> rowF;
        BuildRowFloats(dtype, cs.pattern, axisLen, r + 1, rowF);
        for (int64_t k = 0; k < axisLen; ++k) {
            x[static_cast<size_t>(base + k)] = QuantizeOne<ValueStore>(dtype, rowF[static_cast<size_t>(k)]);
            index[static_cast<size_t>(base + k)] = static_cast<int32_t>(k);  // 每行 0..N-1
        }
    }
}

// ============================================================================
// 统一用例执行：Mock（golden 自验证）/ Real（NPU vs golden）共享同一 golden + 比对。
//   对 extreme/ties 用例（ties 存在）：只断言 permutation 不变量 + value(isnan-aware)，
//   indices 精确比对仅对「无 ties 输入」启用（exactIndices 标志）。
// ============================================================================

#ifndef USE_MOCK_ACLNN
template <typename ValueStore>
static aclDataType ValueAclType(const std::string& dtype);

template <>
aclDataType ValueAclType<uint16_t>(const std::string& dtype)
{
    return (dtype == "bfloat16") ? ACL_BF16 : ACL_FLOAT16;
}
template <>
aclDataType ValueAclType<uint32_t>(const std::string& /*dtype*/)
{
    return ACL_FLOAT;
}
template <>
aclDataType ValueAclType<int32_t>(const std::string& /*dtype*/)
{
    return ACL_INT32;
}

static std::vector<int64_t> ComputeStrides(const std::vector<int64_t>& shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = (int64_t)shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    return strides;
}

template <typename T>
static int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                           aclDataType dataType, aclTensor** tensor)
{
    size_t size = (size_t)GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size == 0 ? sizeof(T) : size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    if (size > 0) {
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
            aclrtFree(*deviceAddr);
            return ret;
        }
    }
    auto strides = ComputeStrides(shape);
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return ACL_SUCCESS;
}
#endif  // USE_MOCK_ACLNN

// indices 是否可精确比对（仅无 ties 的正向用例）。
static bool ExactIndices(DataPattern p)
{
    switch (p) {
        case DataPattern::RANDOM_UNIQUE:
        case DataPattern::SCALAR:
        case DataPattern::SINGLE_POS_INF:
        case DataPattern::SINGLE_NEG_INF:
            return true;  // 无重复有限值（含单个 Inf，仍唯一）
        default:
            return false;  // ALL_ZERO/ALL_SAME/SIGNED_ZERO_MIX/WITH_TIES/NAN_INF_COMBO/INJECT_NAN：有 ties
    }
}

#ifndef USE_MOCK_ACLNN
// Real：执行 NPU + 比对。返回 PASS/FAIL。
template <typename ValueStore>
static bool RunCaseReal(const CaseSpec& cs, const std::vector<ValueStore>& x, const std::vector<int32_t>& index,
                        aclrtStream stream)
{
    std::string dtype = cs.value_dtype;
    auto projectKey = MakeProjectKey<ValueStore>(dtype);
    int64_t total = GetShapeSize(cs.shape);
    size_t n = static_cast<size_t>(total);

    void* xDev = nullptr;
    void* idxDev = nullptr;
    void* yDev = nullptr;
    void* siDev = nullptr;
    aclTensor* xT = nullptr;
    aclTensor* idxT = nullptr;
    aclTensor* yT = nullptr;
    aclTensor* siT = nullptr;
    void* workspace = nullptr;
    auto cleanup = [&]() {
        if (workspace) aclrtFree(workspace);
        if (xT) aclDestroyTensor(xT);
        if (idxT) aclDestroyTensor(idxT);
        if (yT) aclDestroyTensor(yT);
        if (siT) aclDestroyTensor(siT);
        if (xDev) aclrtFree(xDev);
        if (idxDev) aclrtFree(idxDev);
        if (yDev) aclrtFree(yDev);
        if (siDev) aclrtFree(siDev);
    };

    std::vector<ValueStore> yHost(n, ValueStore{});
    std::vector<int32_t> siHost(n, 0);
    aclDataType vType = ValueAclType<ValueStore>(dtype);

    if (CreateAclTensor(x, cs.shape, &xDev, vType, &xT) != ACL_SUCCESS ||
        CreateAclTensor(index, cs.shape, &idxDev, ACL_INT32, &idxT) != ACL_SUCCESS ||
        CreateAclTensor(yHost, cs.shape, &yDev, vType, &yT) != ACL_SUCCESS ||
        CreateAclTensor(siHost, cs.shape, &siDev, ACL_INT32, &siT) != ACL_SUCCESS) {
        LOG_PRINT("  [FAIL] 创建 aclTensor 失败");
        cleanup();
        return false;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnSortWithIndexGetWorkspaceSize(xT, idxT, cs.axis, cs.descending, cs.stable, yT, siT,
                                                  &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  [FAIL] GetWorkspaceSize 失败: %d", ret);
        cleanup();
        return false;
    }
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("  [FAIL] workspace 分配失败: %d (size=%lu)", ret, workspaceSize);
            cleanup();
            return false;
        }
    }
    ret = aclnnSortWithIndex(workspace, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  [FAIL] aclnnSortWithIndex 执行失败: %d", ret);
        cleanup();
        return false;
    }
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  [FAIL] 流同步失败: %d", ret);
        cleanup();
        return false;
    }
    if (n > 0) {
        ret = aclrtMemcpy(yHost.data(), n * sizeof(ValueStore), yDev, n * sizeof(ValueStore),
                          ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret == ACL_SUCCESS) {
            ret = aclrtMemcpy(siHost.data(), n * sizeof(int32_t), siDev, n * sizeof(int32_t),
                              ACL_MEMCPY_DEVICE_TO_HOST);
        }
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("  [FAIL] D2H 拷贝失败: %d", ret);
            cleanup();
            return false;
        }
    }

    std::vector<ValueStore> goldenY;
    std::vector<int32_t> goldenSi;
    ComputeGolden910b<ValueStore, int32_t>(x, index, cs.shape, cs.descending, projectKey, goldenY, goldenSi);

    bool valOk = CompareValuesIsnanAware<ValueStore>(goldenY, yHost, projectKey);
    bool idxOk = true;
    if (ExactIndices(cs.pattern)) {
        idxOk = CompareIndicesExact<int32_t>(goldenSi, siHost);
    } else {
        LOG_PRINT("  [INFO] ties 用例：跳过 indices 精确比对，仅断言 permutation 不变量");
    }
    bool invOk = CheckInvariants910b<ValueStore, int32_t>(x, index, yHost, siHost, cs.shape, cs.descending,
                                                          projectKey);
    cleanup();
    return valOk && idxOk && invOk;
}

// Real：负向用例（预期 aclnn 拒绝）。indexShapeOverride 非空时构造 x≠index shape。
template <typename ValueStore>
static bool RunNegativeReal(const char* caseId, const std::string& dtype, const std::vector<int64_t>& xShape,
                            const std::vector<int64_t>& indexShape, int64_t axis, aclrtStream stream)
{
    auto projectKey = MakeProjectKey<ValueStore>(dtype);
    (void)projectKey;
    std::vector<ValueStore> x(static_cast<size_t>(GetShapeSize(xShape)), ValueStore{});
    std::vector<int32_t> index(static_cast<size_t>(GetShapeSize(indexShape)), 0);

    void* xDev = nullptr;
    void* idxDev = nullptr;
    void* yDev = nullptr;
    void* siDev = nullptr;
    aclTensor* xT = nullptr;
    aclTensor* idxT = nullptr;
    aclTensor* yT = nullptr;
    aclTensor* siT = nullptr;
    void* workspace = nullptr;
    auto cleanup = [&]() {
        if (workspace) aclrtFree(workspace);
        if (xT) aclDestroyTensor(xT);
        if (idxT) aclDestroyTensor(idxT);
        if (yT) aclDestroyTensor(yT);
        if (siT) aclDestroyTensor(siT);
        if (xDev) aclrtFree(xDev);
        if (idxDev) aclrtFree(idxDev);
        if (yDev) aclrtFree(yDev);
        if (siDev) aclrtFree(siDev);
    };
    std::vector<ValueStore> yHost(static_cast<size_t>(GetShapeSize(xShape)), ValueStore{});
    std::vector<int32_t> siHost(static_cast<size_t>(GetShapeSize(indexShape)), 0);
    aclDataType vType = ValueAclType<ValueStore>(dtype);

    if (CreateAclTensor(x, xShape, &xDev, vType, &xT) != ACL_SUCCESS ||
        CreateAclTensor(index, indexShape, &idxDev, ACL_INT32, &idxT) != ACL_SUCCESS ||
        CreateAclTensor(yHost, xShape, &yDev, vType, &yT) != ACL_SUCCESS ||
        CreateAclTensor(siHost, indexShape, &siDev, ACL_INT32, &siT) != ACL_SUCCESS) {
        LOG_PRINT("  [FAIL] 创建 aclTensor 失败");
        cleanup();
        return false;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnSortWithIndexGetWorkspaceSize(xT, idxT, axis, false, false, yT, siT, &workspaceSize, &executor);
    bool rejected = (ret != ACL_SUCCESS);
    if (rejected) {
        LOG_PRINT("  [PASS] 负向用例 %s 被算子优雅拒绝（ret=%d，符合预期）", caseId, ret);
    } else {
        // 部分实现可能在 GetWorkspaceSize 通过后于 execute 阶段拒绝；继续执行验证。
        if (workspaceSize > 0) {
            aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        }
        ret = aclnnSortWithIndex(workspace, workspaceSize, executor, stream);
        if (ret != ACL_SUCCESS) {
            rejected = true;
            LOG_PRINT("  [PASS] 负向用例 %s 于 execute 阶段被拒绝（ret=%d，符合预期）", caseId, ret);
        } else {
            aclrtSynchronizeStream(stream);
            LOG_PRINT("  [FAIL] 负向用例 %s 未被拒绝（GetWorkspaceSize+execute 均成功）", caseId);
        }
    }
    cleanup();
    return rejected;
}
#endif  // USE_MOCK_ACLNN

#ifdef USE_MOCK_ACLNN
// Mock：golden 充当 NPU 输出，验证比对/不变量闭环（无 NPU 依赖）。
template <typename ValueStore>
static bool RunCaseMock(const CaseSpec& cs, const std::vector<ValueStore>& x, const std::vector<int32_t>& index)
{
    std::string dtype = cs.value_dtype;
    auto projectKey = MakeProjectKey<ValueStore>(dtype);

    std::vector<ValueStore> goldenY;
    std::vector<int32_t> goldenSi;
    ComputeGolden910b<ValueStore, int32_t>(x, index, cs.shape, cs.descending, projectKey, goldenY, goldenSi);

    if (cs.is_empty) {
        bool emptyOk = goldenY.empty() && goldenSi.empty();
        LOG_PRINT("  %s 空 tensor 返回空（y.size=%zu si.size=%zu）", emptyOk ? "[PASS]" : "[FAIL]", goldenY.size(),
                  goldenSi.size());
        return emptyOk;
    }

    // Mock 输出 = golden（验证比对/不变量逻辑闭环）。
    std::vector<ValueStore> mockY = goldenY;
    std::vector<int32_t> mockSi = goldenSi;

    bool valOk = CompareValuesIsnanAware<ValueStore>(goldenY, mockY, projectKey);
    bool idxOk = true;
    if (ExactIndices(cs.pattern)) {
        idxOk = CompareIndicesExact<int32_t>(goldenSi, mockSi);
    } else {
        LOG_PRINT("  [INFO] ties 用例：跳过 indices 精确比对，仅断言 permutation 不变量");
    }
    bool invOk = CheckInvariants910b<ValueStore, int32_t>(x, index, mockY, mockSi, cs.shape, cs.descending,
                                                          projectKey);
    return valOk && idxOk && invOk;
}
#endif  // USE_MOCK_ACLNN

// dtype 分派：按 value_dtype 选择存储位类型并执行。
static bool DispatchRunCase(const CaseSpec& cs,
#ifndef USE_MOCK_ACLNN
                            aclrtStream stream
#else
                            int /*unused*/
#endif
)
{
    std::string dtype = cs.value_dtype;
    if (dtype == "float16" || dtype == "bfloat16") {
        std::vector<uint16_t> x;
        std::vector<int32_t> index;
        BuildCaseInputs<uint16_t>(cs, x, index);
#ifdef USE_MOCK_ACLNN
        return RunCaseMock<uint16_t>(cs, x, index);
#else
        return RunCaseReal<uint16_t>(cs, x, index, stream);
#endif
    } else if (dtype == "float32") {
        std::vector<uint32_t> x;
        std::vector<int32_t> index;
        BuildCaseInputs<uint32_t>(cs, x, index);
#ifdef USE_MOCK_ACLNN
        return RunCaseMock<uint32_t>(cs, x, index);
#else
        return RunCaseReal<uint32_t>(cs, x, index, stream);
#endif
    } else {  // int32
        std::vector<int32_t> x;
        std::vector<int32_t> index;
        BuildCaseInputs<int32_t>(cs, x, index);
#ifdef USE_MOCK_ACLNN
        return RunCaseMock<int32_t>(cs, x, index);
#else
        return RunCaseReal<int32_t>(cs, x, index, stream);
#endif
    }
}

// ============================================================================
// CPU Golden 正确性自测（不依赖 NPU；Mock/Real 均执行）—— 910B NaN-开头 语义
// ============================================================================

static bool TestGoldenCorrectness()
{
    LOG_PRINT("\n========================================");
    LOG_PRINT("CPU Golden 正确性自测（910B NaN-开头 语义）");
    LOG_PRINT("========================================");
    bool allPass = true;
    auto keyFp16 = MakeProjectKey<uint16_t>("float16");

    // 自测 1：升序 + ties stable（shape=[5]）
    {
        std::vector<float> xf = {3.0f, 1.0f, 4.0f, 1.0f, 2.0f};
        std::vector<uint16_t> x(xf.size());
        for (size_t i = 0; i < xf.size(); ++i) x[i] = FloatToFp16(xf[i]);
        std::vector<int32_t> idx = {0, 1, 2, 3, 4};
        std::vector<uint16_t> y;
        std::vector<int32_t> si;
        ComputeGolden910b<uint16_t, int32_t>(x, idx, {5}, false, keyFp16, y, si);
        std::vector<int32_t> expectIdx = {1, 3, 4, 0, 2};  // 1,1,2,3,4 -> idx 1,3 在前
        std::vector<float> expectVal = {1.0f, 1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<uint16_t> expectY(expectVal.size());
        for (size_t i = 0; i < expectVal.size(); ++i) expectY[i] = FloatToFp16(expectVal[i]);
        LOG_PRINT("\n自测 1: 升序 + ties stable（shape=[5]）");
        bool p = CompareValuesIsnanAware<uint16_t>(expectY, y, keyFp16) && CompareIndicesExact<int32_t>(expectIdx, si);
        p &= CheckInvariants910b<uint16_t, int32_t>(x, idx, y, si, {5}, false, keyFp16);
        LOG_PRINT("  结果: %s", p ? "PASS" : "FAIL");
        allPass &= p;
    }

    // 自测 2：降序
    {
        std::vector<float> xf = {3.0f, 1.0f, 4.0f, 1.0f, 2.0f};
        std::vector<uint16_t> x(xf.size());
        for (size_t i = 0; i < xf.size(); ++i) x[i] = FloatToFp16(xf[i]);
        std::vector<int32_t> idx = {0, 1, 2, 3, 4};
        std::vector<uint16_t> y;
        std::vector<int32_t> si;
        ComputeGolden910b<uint16_t, int32_t>(x, idx, {5}, true, keyFp16, y, si);
        std::vector<int32_t> expectIdx = {2, 0, 4, 1, 3};  // 4,3,2,1,1
        std::vector<float> expectVal = {4.0f, 3.0f, 2.0f, 1.0f, 1.0f};
        std::vector<uint16_t> expectY(expectVal.size());
        for (size_t i = 0; i < expectVal.size(); ++i) expectY[i] = FloatToFp16(expectVal[i]);
        LOG_PRINT("\n自测 2: 降序 + ties stable（shape=[5]）");
        bool p = CompareValuesIsnanAware<uint16_t>(expectY, y, keyFp16) && CompareIndicesExact<int32_t>(expectIdx, si);
        p &= CheckInvariants910b<uint16_t, int32_t>(x, idx, y, si, {5}, true, keyFp16);
        LOG_PRINT("  结果: %s", p ? "PASS" : "FAIL");
        allPass &= p;
    }

    // 自测 3：多行独立排序（shape=[2,3]）
    {
        std::vector<float> xf = {3.0f, 1.0f, 2.0f, 9.0f, 7.0f, 8.0f};
        std::vector<uint16_t> x(xf.size());
        for (size_t i = 0; i < xf.size(); ++i) x[i] = FloatToFp16(xf[i]);
        std::vector<int32_t> idx = {0, 1, 2, 0, 1, 2};
        std::vector<uint16_t> y;
        std::vector<int32_t> si;
        ComputeGolden910b<uint16_t, int32_t>(x, idx, {2, 3}, false, keyFp16, y, si);
        std::vector<int32_t> expectIdx = {1, 2, 0, 1, 2, 0};
        LOG_PRINT("\n自测 3: 多行独立排序（shape=[2,3]）");
        bool p = CompareIndicesExact<int32_t>(expectIdx, si);
        p &= CheckInvariants910b<uint16_t, int32_t>(x, idx, y, si, {2, 3}, false, keyFp16);
        LOG_PRINT("  结果: %s", p ? "PASS" : "FAIL");
        allPass &= p;
    }

    // 自测 4：★ 910B NaN 升序落「开头」（非 torch 末尾）
    {
        float nanv = std::numeric_limits<float>::quiet_NaN();
        std::vector<float> xf = {2.0f, nanv, 1.0f, 3.0f};
        std::vector<uint16_t> x(xf.size());
        for (size_t i = 0; i < xf.size(); ++i) x[i] = FloatToFp16(xf[i]);
        std::vector<int32_t> idx = {0, 1, 2, 3};
        std::vector<uint16_t> y;
        std::vector<int32_t> si;
        ComputeGolden910b<uint16_t, int32_t>(x, idx, {4}, false, keyFp16, y, si);
        // 910B 升序: NaN(开头),1,2,3 -> idx 1,2,0,3
        std::vector<int32_t> expectIdx = {1, 2, 0, 3};
        LOG_PRINT("\n自测 4: ★910B NaN 升序落开头（shape=[4]，expect idx=1,2,0,3）");
        bool p = CompareIndicesExact<int32_t>(expectIdx, si);
        // y[0] 必须是 NaN
        p &= std::isnan(Fp16ToFloat(y[0]));
        if (!std::isnan(Fp16ToFloat(y[0]))) {
            LOG_PRINT("  [FAIL] y[0] 不是 NaN（910B 升序 NaN 应落开头）");
        }
        p &= CheckInvariants910b<uint16_t, int32_t>(x, idx, y, si, {4}, false, keyFp16);
        LOG_PRINT("  结果: %s", p ? "PASS" : "FAIL");
        allPass &= p;
    }

    // 自测 5：降序 NaN 也落开头
    {
        float nanv = std::numeric_limits<float>::quiet_NaN();
        std::vector<float> xf = {2.0f, nanv, 1.0f, 3.0f};
        std::vector<uint16_t> x(xf.size());
        for (size_t i = 0; i < xf.size(); ++i) x[i] = FloatToFp16(xf[i]);
        std::vector<int32_t> idx = {0, 1, 2, 3};
        std::vector<uint16_t> y;
        std::vector<int32_t> si;
        ComputeGolden910b<uint16_t, int32_t>(x, idx, {4}, true, keyFp16, y, si);
        // 910B 降序: NaN(开头),3,2,1 -> idx 1,3,0,2
        std::vector<int32_t> expectIdx = {1, 3, 0, 2};
        LOG_PRINT("\n自测 5: 降序 NaN 落开头（shape=[4]，expect idx=1,3,0,2）");
        bool p = CompareIndicesExact<int32_t>(expectIdx, si) && std::isnan(Fp16ToFloat(y[0]));
        p &= CheckInvariants910b<uint16_t, int32_t>(x, idx, y, si, {4}, true, keyFp16);
        LOG_PRINT("  结果: %s", p ? "PASS" : "FAIL");
        allPass &= p;
    }

    // 自测 6：±Inf 落位（升序 -Inf 开头、+Inf 末尾）
    {
        float pinf = std::numeric_limits<float>::infinity();
        float ninf = -pinf;
        std::vector<float> xf = {2.0f, pinf, ninf, 1.0f};
        std::vector<uint16_t> x(xf.size());
        for (size_t i = 0; i < xf.size(); ++i) x[i] = FloatToFp16(xf[i]);
        std::vector<int32_t> idx = {0, 1, 2, 3};
        std::vector<uint16_t> y;
        std::vector<int32_t> si;
        ComputeGolden910b<uint16_t, int32_t>(x, idx, {4}, false, keyFp16, y, si);
        // 升序: -Inf,1,2,+Inf -> idx 2,3,0,1
        std::vector<int32_t> expectIdx = {2, 3, 0, 1};
        LOG_PRINT("\n自测 6: ±Inf 升序（-Inf 开头/+Inf 末尾，shape=[4]）");
        bool p = CompareIndicesExact<int32_t>(expectIdx, si);
        p &= CheckInvariants910b<uint16_t, int32_t>(x, idx, y, si, {4}, false, keyFp16);
        LOG_PRINT("  结果: %s", p ? "PASS" : "FAIL");
        allPass &= p;
    }

    // 自测 7：NaN 位型 isnan 比较（升序 NaN bit 翻转，golden vs «翻位» actual 仍相等）
    {
        std::vector<uint16_t> golden = {0x7E00, FloatToFp16(1.0f), FloatToFp16(2.0f)};   // canonical NaN
        std::vector<uint16_t> actual = {0x7FFF, FloatToFp16(1.0f), FloatToFp16(2.0f)};   // sign-flipped NaN (910B 升序)
        LOG_PRINT("\n自测 7: NaN 位型 isnan 比较（0x7E00 vs 0x7FFF 仍视为相等）");
        bool p = CompareValuesIsnanAware<uint16_t>(golden, actual, keyFp16);
        LOG_PRINT("  结果: %s", p ? "PASS" : "FAIL");
        allPass &= p;
    }

    // 自测 8：空 tensor
    {
        std::vector<uint16_t> x, y;
        std::vector<int32_t> idx, si;
        ComputeGolden910b<uint16_t, int32_t>(x, idx, {0}, false, keyFp16, y, si);
        LOG_PRINT("\n自测 8: 空 tensor（shape=[0]）");
        bool p = y.empty() && si.empty();
        LOG_PRINT("  %s 空输出（y.size=%zu si.size=%zu）", p ? "[PASS]" : "[FAIL]", y.size(), si.size());
        allPass &= p;
    }

    // 自测 9：int32 value golden（Cast 语义，|x|<=2^24）
    {
        auto keyI32 = MakeProjectKey<int32_t>("int32");
        std::vector<int32_t> x = {30, -10, 20, -10, 5};
        std::vector<int32_t> idx = {0, 1, 2, 3, 4};
        std::vector<int32_t> y, si;
        ComputeGolden910b<int32_t, int32_t>(x, idx, {5}, false, keyI32, y, si);
        std::vector<int32_t> expectY = {-10, -10, 5, 20, 30};
        std::vector<int32_t> expectIdx = {1, 3, 4, 2, 0};
        LOG_PRINT("\n自测 9: int32 value 升序 + ties（shape=[5]）");
        bool p = CompareIndicesExact<int32_t>(expectIdx, si) && CompareIndicesExact<int32_t>(expectY, y);
        p &= CheckInvariants910b<int32_t, int32_t>(x, idx, y, si, {5}, false, keyI32);
        LOG_PRINT("  结果: %s", p ? "PASS" : "FAIL");
        allPass &= p;
    }

    LOG_PRINT("\nGolden 自测汇总: %s", allPass ? "全部 PASS" : "存在 FAIL");
    LOG_PRINT("========================================");
    return allPass;
}

// ============================================================================
// 全量用例表（4 dtype × shape/属性/边界/extreme）
// ============================================================================

static std::vector<CaseSpec> GetFullCases()
{
    using DP = DataPattern;
    std::vector<CaseSpec> cases;
    const char* dtypes[] = {"float16", "float32", "bfloat16", "int32"};

    // ---- A. 每个 dtype：基础 shape × descending × stable 组合 ----
    int idA = 0;
    for (const char* dt : dtypes) {
        // 1D 小轴：升/降 × stable
        cases.push_back({"", dt, {8}, -1, false, false, DP::RANDOM_UNIQUE, false, "1D 升序 unstable"});
        cases.push_back({"", dt, {8}, -1, false, true, DP::RANDOM_UNIQUE, false, "1D 升序 stable"});
        cases.push_back({"", dt, {8}, -1, true, false, DP::RANDOM_UNIQUE, false, "1D 降序 unstable"});
        cases.push_back({"", dt, {8}, -1, true, true, DP::RANDOM_UNIQUE, false, "1D 降序 stable"});
        // 多行分核 [4,32]：升/降
        cases.push_back({"", dt, {4, 32}, -1, false, true, DP::RANDOM_UNIQUE, false, "多行分核 升序"});
        cases.push_back({"", dt, {4, 32}, -1, true, true, DP::RANDOM_UNIQUE, false, "多行分核 降序"});
        // rank2 中轴长 [3,64]
        cases.push_back({"", dt, {3, 64}, -1, false, true, DP::RANDOM_UNIQUE, false, "rank2 轴长64"});
        (void)idA;
    }

    // ---- B. shape 多 rank（rank0–8）+ 单 tile 内多轴长（每 dtype 选一个代表，避免组合爆炸）----
    // rank0 标量
    cases.push_back({"", "float16", {}, -1, false, false, DP::SCALAR, false, "rank0 标量"});
    cases.push_back({"", "float32", {}, -1, true, true, DP::SCALAR, false, "rank0 标量 降序"});
    cases.push_back({"", "int32", {}, -1, false, false, DP::SCALAR, false, "rank0 标量 int32"});
    // rank1 轴长1（拷贝路径）
    cases.push_back({"", "float16", {1}, -1, false, true, DP::RANDOM_UNIQUE, false, "轴长=1 拷贝"});
    cases.push_back({"", "bfloat16", {1}, -1, true, true, DP::RANDOM_UNIQUE, false, "轴长=1 拷贝 bf16"});
    // rank2 轴长1（多行单元素拷贝）
    cases.push_back({"", "float32", {4, 1}, -1, false, true, DP::RANDOM_UNIQUE, false, "rank2 轴长=1"});
    cases.push_back({"", "int32", {4, 1}, -1, true, true, DP::RANDOM_UNIQUE, false, "rank2 轴长=1 int32"});
    // rank3
    cases.push_back({"", "float16", {2, 3, 8}, -1, false, false, DP::RANDOM_UNIQUE, false, "rank3"});
    // rank4
    cases.push_back({"", "bfloat16", {2, 2, 2, 16}, -1, false, true, DP::RANDOM_UNIQUE, false, "rank4"});
    // rank8（最大 rank）axis=-1
    cases.push_back({"", "float16", {2, 2, 2, 2, 2, 2, 2, 4}, -1, false, true, DP::RANDOM_UNIQUE, false, "rank8 axis=-1"});
    // rank8 axis=rank-1=7（等价最后一维）
    cases.push_back({"", "float32", {2, 2, 2, 2, 2, 2, 2, 4}, 7, false, true, DP::RANDOM_UNIQUE, false, "rank8 axis=rank-1"});
    // axis=rank-1（rank2 显式正轴）
    cases.push_back({"", "bfloat16", {3, 8}, 1, true, true, DP::RANDOM_UNIQUE, false, "rank2 axis=rank-1 降序"});

    // ---- C. 单 tile 内多轴长（轴长 <= 单 tile 上限；不超界）----
    cases.push_back({"", "float16", {256}, -1, false, true, DP::RANDOM_UNIQUE, false, "fp16 轴长256"});
    cases.push_back({"", "float16", {2048}, -1, false, true, DP::RANDOM_UNIQUE, false, "fp16 轴长2048"});
    cases.push_back({"", "float16", {3008}, -1, false, true, DP::RANDOM_UNIQUE, false, "fp16 轴长3008(近上限)"});
    cases.push_back({"", "float16", {3008}, -1, true, true, DP::RANDOM_UNIQUE, false, "fp16 轴长3008 降序"});
    cases.push_back({"", "float32", {2048}, -1, false, true, DP::RANDOM_UNIQUE, false, "fp32 轴长2048"});
    cases.push_back({"", "float32", {2816}, -1, false, true, DP::RANDOM_UNIQUE, false, "fp32 轴长2816(近上限)"});
    cases.push_back({"", "bfloat16", {2048}, -1, false, true, DP::RANDOM_UNIQUE, false, "bf16 轴长2048"});
    cases.push_back({"", "bfloat16", {2816}, -1, false, true, DP::RANDOM_UNIQUE, false, "bf16 轴长2816(近上限)"});
    cases.push_back({"", "int32", {2048}, -1, false, true, DP::RANDOM_UNIQUE, false, "int32 轴长2048"});
    cases.push_back({"", "int32", {2560}, -1, false, true, DP::RANDOM_UNIQUE, false, "int32 轴长2560(近上限)"});
    // 多行 + 较大轴
    cases.push_back({"", "float16", {16, 256}, -1, false, true, DP::RANDOM_UNIQUE, false, "16行×256 多行较大轴"});

    // ---- D. 边界：空 tensor ----
    cases.push_back({"", "float16", {0}, -1, false, false, DP::RANDOM_UNIQUE, true, "空 tensor [0]"});
    cases.push_back({"", "float32", {0}, -1, false, false, DP::RANDOM_UNIQUE, true, "空 tensor [0]"});
    cases.push_back({"", "bfloat16", {0}, -1, false, false, DP::RANDOM_UNIQUE, true, "空 tensor [0]"});
    cases.push_back({"", "int32", {0}, -1, false, false, DP::RANDOM_UNIQUE, true, "空 tensor [0]"});
    cases.push_back({"", "float16", {3, 0}, -1, false, false, DP::RANDOM_UNIQUE, true, "空 tensor 多维 [3,0]"});
    cases.push_back({"", "float32", {0, 8}, -1, false, false, DP::RANDOM_UNIQUE, true, "空 tensor 多维 [0,8]"});

    // ---- E. extreme（仅浮点 dtype 适用 NaN/Inf/±0；int32 适用 全零/全相等）----
    for (const char* dt : {"float16", "float32", "bfloat16"}) {
        cases.push_back({"", dt, {8}, -1, false, true, DP::INJECT_NAN, false, "NaN 升序落开头"});
        cases.push_back({"", dt, {8}, -1, true, true, DP::INJECT_NAN, false, "NaN 降序落开头"});
        cases.push_back({"", dt, {8}, -1, false, true, DP::SINGLE_POS_INF, false, "+Inf 升序末尾"});
        cases.push_back({"", dt, {8}, -1, false, true, DP::SINGLE_NEG_INF, false, "-Inf 升序开头"});
        cases.push_back({"", dt, {8}, -1, false, true, DP::NAN_INF_COMBO, false, "NaN+±Inf 综合"});
        cases.push_back({"", dt, {8}, -1, false, true, DP::ALL_ZERO, false, "全零 ties"});
        cases.push_back({"", dt, {8}, -1, false, true, DP::ALL_SAME, false, "全相等 ties stable"});
        cases.push_back({"", dt, {8}, -1, false, true, DP::SIGNED_ZERO_MIX, false, "±0 视为相等 ties"});
    }
    // int32 extreme（无 NaN/Inf）
    cases.push_back({"", "int32", {8}, -1, false, true, DP::ALL_ZERO, false, "int32 全零 ties"});
    cases.push_back({"", "int32", {8}, -1, false, true, DP::ALL_SAME, false, "int32 全相等 ties stable"});

    // ---- F. 确定性（含 ties 的重复执行，golden 兜底；Real 模式由不变量重复成立验证）----
    cases.push_back({"", "float16", {16}, -1, false, true, DP::WITH_TIES, false, "确定性 ties D1"});
    cases.push_back({"", "float32", {16, 64}, -1, false, true, DP::WITH_TIES, false, "确定性 多行分核 D2"});

    // 填充自动 case_id
    static std::vector<std::string> ids;
    ids.clear();
    ids.reserve(cases.size());
    for (size_t i = 0; i < cases.size(); ++i) {
        ids.push_back("FULL-" + std::string(cases[i].value_dtype) + "-" + std::to_string(i + 1));
    }
    for (size_t i = 0; i < cases.size(); ++i) {
        cases[i].case_id = ids[i].c_str();
    }
    return cases;
}

// ============================================================================
// 主函数
// ============================================================================

int main(int /*argc*/, char* /*argv*/[])
{
    LOG_PRINT("\n========================================");
    LOG_PRINT("SortWithIndex 算子 ST 测试（全量：4 组 dtype + 边界 + extreme，910B NaN-开头）");
    LOG_PRINT("========================================");
#ifdef USE_MOCK_ACLNN
    LOG_PRINT("模式: Mock (CPU golden 自验证，无需 NPU)");
#else
    LOG_PRINT("模式: Real (NPU 执行 + bitwise(isnan-aware)/精确比对)");
#endif

    bool goldenOk = TestGoldenCorrectness();

    int passed = 0;
    int failed = 0;

#ifndef USE_MOCK_ACLNN
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
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

    LOG_PRINT("\n执行全量正向用例（4 dtype × shape/属性/边界/extreme）...");
    auto cases = GetFullCases();
    for (const auto& cs : cases) {
        LOG_PRINT("\n用例 %s [%s]: shape=%s axis=%ld desc=%d stable=%d (%s)", cs.case_id, cs.value_dtype,
                  ShapeToStr(cs.shape).c_str(), (long)cs.axis, (int)cs.descending, (int)cs.stable, cs.note);
        bool ok = DispatchRunCase(cs,
#ifndef USE_MOCK_ACLNN
                                  stream
#else
                                  0
#endif
        );
        if (ok) {
            passed++;
        } else {
            failed++;
        }
        LOG_PRINT("  用例 %s: %s", cs.case_id, ok ? "PASS" : "FAIL");
    }

#ifndef USE_MOCK_ACLNN
    // ---- 负向用例（仅 Real 模式：验证算子优雅拒绝；Mock 无 acl 依赖故跳过）----
    LOG_PRINT("\n========================================");
    LOG_PRINT("负向用例（预期算子优雅拒绝）");
    LOG_PRINT("========================================");
    {
        // N1: x 与 index shape 不一致 → shape_mismatch
        LOG_PRINT("\n负向 N1: x.shape=[2,3] index.shape=[2,4]（shape 不一致）");
        bool ok = RunNegativeReal<uint16_t>("N1-shape-mismatch", "float16", {2, 3}, {2, 4}, -1, stream);
        ok ? passed++ : failed++;
    }
    {
        // N2: axis 非最后一维（axis=0, rank2）→ attribute_value_out_of_range
        LOG_PRINT("\n负向 N2: shape=[4,8] axis=0（非最后一维）");
        bool ok = RunNegativeReal<uint16_t>("N2-axis-not-last", "float16", {4, 8}, {4, 8}, 0, stream);
        ok ? passed++ : failed++;
    }
    {
        // N3: axis 越界（axis=5, rank2）
        LOG_PRINT("\n负向 N3: shape=[4,8] axis=5（越界）");
        bool ok = RunNegativeReal<uint16_t>("N3-axis-oob", "float16", {4, 8}, {4, 8}, 5, stream);
        ok ? passed++ : failed++;
    }
    {
        // N4: 轴长超单 tile 上限（fp16 N=8192 >> ~3008）→ tiling 优雅拒绝
        LOG_PRINT("\n负向 N4: fp16 轴长 N=8192（超单 tile 上限，预期 GRAPH_FAILED 拒绝）");
        bool ok = RunNegativeReal<uint16_t>("N4-axis-too-long", "float16", {8192}, {8192}, -1, stream);
        ok ? passed++ : failed++;
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
#endif

    LOG_PRINT("\n========================================");
    LOG_PRINT("测试报告");
    LOG_PRINT("========================================");
    LOG_PRINT("Golden 自测: %s", goldenOk ? "PASS" : "FAIL");
    LOG_PRINT("用例总计: %d", passed + failed);
    LOG_PRINT("通过: %d", passed);
    LOG_PRINT("失败: %d", failed);
    LOG_PRINT("========================================\n");

    return (goldenOk && failed == 0) ? 0 : 1;
}
