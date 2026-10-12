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
 * \file test_tensor_redirect_tiling_arch35.cpp
 * \brief TensorRedirect op_host Tiling UT
 */

#include <cstdint>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

// 被测代码的真值源：直接 include，避免 UT 侧结构体定义漂移。
// 路径相对于 OP_TILING_INCLUDE 中的 ${OPS_MATH_DIR}（cmake/variables.cmake），
// 与本仓 UT 既有约定一致（无 per-op include dir）。
#include "conversion/tensor_redirect/op_host/arch35/tensor_redirect_tiling_arch35.h"
#include "conversion/tensor_redirect/op_kernel/arch35/tensor_redirect_tiling_data.h"

using namespace std;
using namespace ge;

namespace {

// ---- 平台常量 ----
constexpr int64_t PLAT_CORE_NUM = 64;         // vector_core_cnt 实测值
constexpr int64_t PLAT_UB_SIZE = 253952;      // GetCoreMemSize(UB)：已扣 8KB 预留的**可用**值
constexpr int64_t N_BUFFER = 2;               // double buffer
constexpr uint64_t EXPECT_TILING_KEY = 0;     // TPL_SCH_MODE_0（唯一调度模式）
constexpr size_t EXPECT_WORKSPACE = 16777216; // GetLibApiWorkSpaceSize() 实测值

// fp16 下的 UB 单块上界：253952 / 2 / 2 = 63488（100% UB 档位）
constexpr int64_t MAX_UB_FP16 = PLAT_UB_SIZE / N_BUFFER / 2;
// 100% UB 门槛
constexpr int64_t UB_FULL_THRESHOLD_FP16 = PLAT_CORE_NUM * MAX_UB_FP16;

// ---- 4 桶矩阵所需的 dtype 无关量 ----
constexpr int64_t UB_FACTOR_MIN_BETY = 2048; // UB 单块下界（字节）
constexpr int64_t ONE_BLK_BYTE = 32;         // ubblock_size

// UT 侧独立实现的向上对齐（**刻意不复用**被测代码的 Ops::Base::CeilAlign —— 若复用，
// 对齐逻辑本身出错时 UT 会跟着一起错，失去判据独立性）。
constexpr int64_t CeilAlignRef(int64_t value, int64_t align)
{
    return (align == 0) ? value : ((value + align - 1) / align) * align;
}

// 每桶的派生量（全部为「字节宽的函数」，故必须逐桶验证，不能只验 fp16）
constexpr int64_t MaxUbFor(int64_t bytes) { return PLAT_UB_SIZE / N_BUFFER / bytes; }
constexpr int64_t UbFactorMinFor(int64_t bytes) { return UB_FACTOR_MIN_BETY / bytes; }
constexpr int64_t ElemsPer32BFor(int64_t bytes) { return ONE_BLK_BYTE / bytes; }
constexpr int64_t UbFullThresholdFor(int64_t bytes) { return PLAT_CORE_NUM * MaxUbFor(bytes); }

// binary.json 的 4 个 DtypeByte 桶（bin_filename: TensorRedirect_{1,2,4,8}_BYTES）。
// 每桶取一个代表 dtype —— 同桶内 dtype 的等价性另由「同桶一致性」用例锁定
// （tiling_bf16_same_bucket_as_fp16 / tiling_uint16_same_bucket_as_int16）。
struct BucketCase {
    const char* name;
    ge::DataType dtype;
    int64_t bytes;
};

const BucketCase kBuckets[] = {
    {"1_BYTES(int8)", ge::DT_INT8, 1},
    {"2_BYTES(fp16)", ge::DT_FLOAT16, 2},
    {"4_BYTES(fp32)", ge::DT_FLOAT, 4},
    {"8_BYTES(int64)", ge::DT_INT64, 8},
};

// 经 CompileInfo 注入平台参数
optiling::TensorRedirectCompileInfo g_compileInfo{PLAT_CORE_NUM, PLAT_UB_SIZE, static_cast<int64_t>(EXPECT_WORKSPACE)};
// 单核守卫分支专用
optiling::TensorRedirectCompileInfo g_compileInfoSingleCore{1, PLAT_UB_SIZE, static_cast<int64_t>(EXPECT_WORKSPACE)};

// 构造 x / output_x 同 shape 同 dtype 的标准用例参数
gert::TilingContextPara MakePara(const gert::StorageShape& shape, ge::DataType dtype,
                                 void* compileInfo = &g_compileInfo)
{
    return gert::TilingContextPara("TensorRedirect", {{shape, dtype, ge::FORMAT_ND}}, {{shape, dtype, ge::FORMAT_ND}},
                                   compileInfo, static_cast<uint64_t>(PLAT_CORE_NUM),
                                   static_cast<uint64_t>(PLAT_UB_SIZE));
}

// 从 TilingInfo 还原 TensorRedirectTilingData（POD struct，直接按字节取回）
const TensorRedirectTilingData* AsTilingData(const TilingInfo& info)
{
    EXPECT_EQ(info.tilingDataSize, sizeof(TensorRedirectTilingData));
    return reinterpret_cast<const TensorRedirectTilingData*>(info.tilingData.get());
}

bool SafeExecuteTiling(const gert::TilingContextPara& para, TilingInfo& info)
{
    try {
        return ExecuteTiling(para, info);
    } catch (const std::invalid_argument& e) {
        return false;
    }
}

// 公共不变量校验
void CheckCommonInvariants(const TilingInfo& info, int64_t numel, int64_t bytesForOneData)
{
    const auto* td = AsTilingData(info);
    ASSERT_NE(td, nullptr);

    // TilingKey / workspace
    EXPECT_EQ(static_cast<uint64_t>(info.tilingKey), EXPECT_TILING_KEY);
    ASSERT_EQ(info.workspaceSizes.size(), 1U);
    EXPECT_EQ(static_cast<size_t>(info.workspaceSizes[0]), EXPECT_WORKSPACE);

    // SetBlockDim 恒 >= 1
    EXPECT_GE(info.blockNum, 1U) << "SetBlockDim(0) 会导致 LaunchKernelV2 107000/EE1003 静默失败";
    EXPECT_EQ(static_cast<int64_t>(info.blockNum), td->usedCoreNum) << "blockDim 必须等于 usedCoreNum";

    // 核数边界
    EXPECT_GE(td->usedCoreNum, 1);
    EXPECT_LE(td->usedCoreNum, PLAT_CORE_NUM);

    // UB 上界：ubFactor 不得超 UB 预算
    EXPECT_GE(td->ubFactor, 1);
    EXPECT_LE(td->ubFactor * bytesForOneData * N_BUFFER, PLAT_UB_SIZE)
        << "ubFactor 超 UB 预算：无运行时兜底，将表征为 507035 向量核异常 + 静默数据错误";

    // 32B 对齐
    const int64_t elemsPer32B = 32 / bytesForOneData;
    EXPECT_TRUE(td->ubFactor == numel || td->ubFactor % elemsPer32B == 0)
        << "ubFactor=" << td->ubFactor << " 既不等于 numel 也非 32B 对齐";

    // 循环/尾块恒等式
    EXPECT_GE(td->blockFactor, 1);
    EXPECT_GE(td->tailBlockFactor, 1);
    EXPECT_GE(td->tailBlockTailUbFactor, 1);
    EXPECT_LE(td->tailBlockTailUbFactor, td->ubFactor);

    // 覆盖总量 == numel：前 (used-1) 核各 blockFactor 个满块，尾核 (tailBlockFactor-1) 个满块 + 1 个尾块
    const int64_t covered = (td->usedCoreNum - 1) * td->blockFactor * td->ubFactor +
                            (td->tailBlockFactor - 1) * td->ubFactor + td->tailBlockTailUbFactor;
    EXPECT_EQ(covered, numel) << "各核区间并集必须恰为 [0, numel)，否则数据漏搬/重搬";
}

} // namespace

class TensorRedirectTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "TensorRedirectTilingTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "TensorRedirectTilingTest TearDown" << std::endl; }
};

// 一、空 Tensor 防护

// [0,3]：numel=0 → 早返回，BlockDim=1（**绝不可为 0**），workspace 仍为 EXPECT_WORKSPACE
TEST_F(TensorRedirectTilingTest, tiling_empty_tensor_0x3_blockdim_must_not_be_zero)
{
    gert::StorageShape shape = {{0, 3}, {0, 3}};
    TilingInfo info;
    ASSERT_TRUE(SafeExecuteTiling(MakePara(shape, ge::DT_FLOAT16), info));

    // SetBlockDim(0) 反回归锚点
    EXPECT_GE(info.blockNum, 1U) << "numel==0 时 usedCoreNum 退化为 0 → SetBlockDim(0) → 107000/EE1003";
    EXPECT_EQ(info.blockNum, 1U) << "空 Tensor 应固定 BlockDim=1（DESIGN §3.7.2）";

    EXPECT_EQ(static_cast<uint64_t>(info.tilingKey), EXPECT_TILING_KEY);
    ASSERT_EQ(info.workspaceSizes.size(), 1U);
    EXPECT_EQ(static_cast<size_t>(info.workspaceSizes[0]), EXPECT_WORKSPACE);

    // TilingData 全零（memset）→ kernel 不下发有效计算
    const auto* td = AsTilingData(info);
    ASSERT_NE(td, nullptr);
    EXPECT_EQ(td->usedCoreNum, 0);
    EXPECT_EQ(td->ubFactor, 0);
    EXPECT_EQ(td->blockFactor, 0);
    EXPECT_EQ(td->tailBlockFactor, 0);
    EXPECT_EQ(td->tailBlockTailUbFactor, 0);
}

// [2,0,3]：0 维度在中间 —— GetShapeSize() 同样为 0，须走同一早返回路径
TEST_F(TensorRedirectTilingTest, tiling_empty_tensor_2x0x3_blockdim_must_not_be_zero)
{
    gert::StorageShape shape = {{2, 0, 3}, {2, 0, 3}};
    TilingInfo info;
    ASSERT_TRUE(SafeExecuteTiling(MakePara(shape, ge::DT_FLOAT16), info));

    EXPECT_GE(info.blockNum, 1U) << "SetBlockDim(0) 反回归锚点";
    EXPECT_EQ(info.blockNum, 1U);
    ASSERT_EQ(info.workspaceSizes.size(), 1U);
    EXPECT_EQ(static_cast<size_t>(info.workspaceSizes[0]), EXPECT_WORKSPACE);
}

// 空 Tensor 早返回与 dtype 无关：换 8 字节宽 dtype 仍须 BlockDim>=1
TEST_F(TensorRedirectTilingTest, tiling_empty_tensor_int64_blockdim_must_not_be_zero)
{
    gert::StorageShape shape = {{0}, {0}};
    TilingInfo info;
    ASSERT_TRUE(SafeExecuteTiling(MakePara(shape, ge::DT_INT64), info));
    EXPECT_GE(info.blockNum, 1U);
    EXPECT_EQ(info.blockNum, 1U);
}

// 二、校验路径

// rank < 1：0 维标量 → shape_mismatch（spec inputs[0].rank_range = [1,8]）
TEST_F(TensorRedirectTilingTest, tiling_check_rank_below_min_scalar_failed)
{
    gert::StorageShape shape = {{}, {}};
    ExecuteTestCase(MakePara(shape, ge::DT_FLOAT16), ge::GRAPH_FAILED, EXPECT_TILING_KEY, std::vector<size_t>{});
}

// rank > 8：9 维 → shape_mismatch
TEST_F(TensorRedirectTilingTest, tiling_check_rank_above_max_9d_failed)
{
    gert::StorageShape shape = {{2, 1, 1, 1, 1, 1, 1, 1, 1}, {2, 1, 1, 1, 1, 1, 1, 1, 1}};
    ExecuteTestCase(MakePara(shape, ge::DT_FLOAT16), ge::GRAPH_FAILED, EXPECT_TILING_KEY, std::vector<size_t>{});
}

// rank == 8 边界内侧：合法，须成功
TEST_F(TensorRedirectTilingTest, tiling_check_rank_max_8d_success)
{
    gert::StorageShape shape = {{2, 2, 2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2, 2, 2}};
    TilingInfo info;
    ASSERT_TRUE(SafeExecuteTiling(MakePara(shape, ge::DT_FLOAT16), info));
    CheckCommonInvariants(info, 256, 2); // 2^8 = 256
}

// rank == 1 边界内侧：合法，须成功
TEST_F(TensorRedirectTilingTest, tiling_check_rank_min_1d_success)
{
    gert::StorageShape shape = {{128}, {128}};
    TilingInfo info;
    ASSERT_TRUE(SafeExecuteTiling(MakePara(shape, ge::DT_FLOAT16), info));
    CheckCommonInvariants(info, 128, 2);
}

// dtype 不支持（DT_DOUBLE 不在 11 种白名单内）→ dtype_not_supported
TEST_F(TensorRedirectTilingTest, tiling_check_dtype_unsupported_double_failed)
{
    gert::StorageShape shape = {{128}, {128}};
    ExecuteTestCase(MakePara(shape, ge::DT_DOUBLE), ge::GRAPH_FAILED, EXPECT_TILING_KEY, std::vector<size_t>{});
}

// dtype 不支持（DT_BOOL）→ dtype_not_supported
TEST_F(TensorRedirectTilingTest, tiling_check_dtype_unsupported_bool_failed)
{
    gert::StorageShape shape = {{128}, {128}};
    ExecuteTestCase(MakePara(shape, ge::DT_BOOL), ge::GRAPH_FAILED, EXPECT_TILING_KEY, std::vector<size_t>{});
}

// x.dtype != output_x.dtype → dtype 一致性校验失败
TEST_F(TensorRedirectTilingTest, tiling_check_dtype_mismatch_x_vs_output_failed)
{
    gert::StorageShape shape = {{128}, {128}};
    gert::TilingContextPara para("TensorRedirect", {{shape, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                 {{shape, ge::DT_FLOAT, ge::FORMAT_ND}}, &g_compileInfo,
                                 static_cast<uint64_t>(PLAT_CORE_NUM), static_cast<uint64_t>(PLAT_UB_SIZE));
    ExecuteTestCase(para, ge::GRAPH_FAILED, EXPECT_TILING_KEY, std::vector<size_t>{});
}

// x.shape != output_x.shape（维度值不同）→ shape_mismatch
TEST_F(TensorRedirectTilingTest, tiling_check_shape_mismatch_dim_value_failed)
{
    gert::StorageShape xShape = {{128}, {128}};
    gert::StorageShape yShape = {{64}, {64}};
    gert::TilingContextPara para("TensorRedirect", {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                 {{yShape, ge::DT_FLOAT16, ge::FORMAT_ND}}, &g_compileInfo,
                                 static_cast<uint64_t>(PLAT_CORE_NUM), static_cast<uint64_t>(PLAT_UB_SIZE));
    ExecuteTestCase(para, ge::GRAPH_FAILED, EXPECT_TILING_KEY, std::vector<size_t>{});
}

// x.shape != output_x.shape（rank 不同，但 numel 相同）→ 逐维严格相等校验须拦截
TEST_F(TensorRedirectTilingTest, tiling_check_shape_mismatch_rank_same_numel_failed)
{
    gert::StorageShape xShape = {{4, 32}, {4, 32}};
    gert::StorageShape yShape = {{128}, {128}};
    gert::TilingContextPara para("TensorRedirect", {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                 {{yShape, ge::DT_FLOAT16, ge::FORMAT_ND}}, &g_compileInfo,
                                 static_cast<uint64_t>(PLAT_CORE_NUM), static_cast<uint64_t>(PLAT_UB_SIZE));
    ExecuteTestCase(para, ge::GRAPH_FAILED, EXPECT_TILING_KEY, std::vector<size_t>{});
}

// concrete shape 含 -1（动态占位符穿透到 Tiling）→ 必须拦截，
// 否则 GetShapeSize() 返回负 numel，绕过 numel==0 守卫并算出 usedCoreNum==0
TEST_F(TensorRedirectTilingTest, tiling_check_negative_dim_minus1_failed)
{
    gert::StorageShape shape = {{-1, 32}, {-1, 32}};
    ExecuteTestCase(MakePara(shape, ge::DT_FLOAT16), ge::GRAPH_FAILED, EXPECT_TILING_KEY, std::vector<size_t>{});
}

// concrete shape 含 -2（unknown rank 占位符）→ 必须拦截
TEST_F(TensorRedirectTilingTest, tiling_check_negative_dim_minus2_failed)
{
    gert::StorageShape shape = {{-2}, {-2}};
    ExecuteTestCase(MakePara(shape, ge::DT_FLOAT16), ge::GRAPH_FAILED, EXPECT_TILING_KEY, std::vector<size_t>{});
}

// 负维出现在非首维 → 逐维校验须覆盖全部维度，而不只是 dim0
TEST_F(TensorRedirectTilingTest, tiling_check_negative_dim_at_last_axis_failed)
{
    gert::StorageShape shape = {{4, 8, -1}, {4, 8, -1}};
    ExecuteTestCase(MakePara(shape, ge::DT_FLOAT16), ge::GRAPH_FAILED, EXPECT_TILING_KEY, std::vector<size_t>{});
}

// 维度乘积溢出 int64_t：GetShapeSize() 返回 kInvalidDimValue(INT64_MIN)，
// 该哨兵不是错误码，必须由 Tiling 显式拦截
TEST_F(TensorRedirectTilingTest, tiling_check_shape_size_overflow_failed)
{
    gert::StorageShape shape = {{INT64_MAX, 2}, {INT64_MAX, 2}};
    ExecuteTestCase(MakePara(shape, ge::DT_FLOAT16), ge::GRAPH_FAILED, EXPECT_TILING_KEY, std::vector<size_t>{});
}

// 三、多核切分核心路径

// #4：[1048577] 触发提核优化
TEST_F(TensorRedirectTilingTest, tiling_fp16_1048577_core_boost_optimization)
{
    gert::StorageShape shape = {{1048577}, {1048577}};
    constexpr int64_t numel = 1048577;
    TilingInfo info;
    ASSERT_TRUE(SafeExecuteTiling(MakePara(shape, ge::DT_FLOAT16), info));
    CheckCommonInvariants(info, numel, 2);

    const auto* td = AsTilingData(info);
    // ubFactor = CeilAlign(FloorDiv(1048577, 63), 16) = 16656（numel % 64 != 0 → 走 totalCoreNum-1 分支）
    EXPECT_EQ(td->ubFactor, 16656);
    EXPECT_EQ(td->usedCoreNum, 63);
    EXPECT_EQ(td->blockFactor, 1) << "提核优化后每核仅一次循环";
    EXPECT_EQ(td->tailBlockFactor, 1);
    // 1048577 - 62 * 16656 = 15905
    EXPECT_EQ(td->tailBlockTailUbFactor, 15905);
    EXPECT_EQ(info.blockNum, 63U);

    EXPECT_LT(td->ubFactor, MAX_UB_FP16) << "[1048577] 不应取满 UB 上界（DESIGN §3.7.4 #4）";
}

// #4b：[16777217] 100% UB + 多核 + 多循环 + 非对齐尾块
TEST_F(TensorRedirectTilingTest, tiling_fp16_16777217_ub_upper_bound_multi_loop)
{
    gert::StorageShape shape = {{16777217}, {16777217}};
    constexpr int64_t numel = 16777217;
    TilingInfo info;
    ASSERT_TRUE(SafeExecuteTiling(MakePara(shape, ge::DT_FLOAT16), info));
    CheckCommonInvariants(info, numel, 2);

    const auto* td = AsTilingData(info);
    EXPECT_EQ(td->ubFactor, MAX_UB_FP16); // 63488
    EXPECT_EQ(td->ubFactor, 63488);
    EXPECT_EQ(td->usedCoreNum, 53);
    EXPECT_EQ(td->blockFactor, 5) << "多循环：double buffer 两个档位均被真正写入";
    EXPECT_EQ(td->tailBlockFactor, 5);
    // 16777217 - 264 * 63488 = 16385（非对齐尾块）
    EXPECT_EQ(td->tailBlockTailUbFactor, 16385);
    EXPECT_EQ(info.blockNum, 53U);

    // 100% UB：ubFactor * 2B * 2(double buffer) == 253952，正好用满且不越界
    EXPECT_EQ(td->ubFactor * 2 * N_BUFFER, PLAT_UB_SIZE) << "UB 上界档位（100%）";
    EXPECT_GT(td->blockFactor, 1) << "必须多循环，否则未覆盖 double buffer 第 2 档";
}

// 100% UB 门槛边界
TEST_F(TensorRedirectTilingTest, tiling_fp16_ub_full_threshold_exact)
{
    gert::StorageShape shape = {{UB_FULL_THRESHOLD_FP16}, {UB_FULL_THRESHOLD_FP16}};
    TilingInfo info;
    ASSERT_TRUE(SafeExecuteTiling(MakePara(shape, ge::DT_FLOAT16), info));
    CheckCommonInvariants(info, UB_FULL_THRESHOLD_FP16, 2);

    const auto* td = AsTilingData(info);
    EXPECT_EQ(UB_FULL_THRESHOLD_FP16, 4063232);
    EXPECT_EQ(td->ubFactor, MAX_UB_FP16) << "达到门槛即取满 UB 上界";
    EXPECT_EQ(td->usedCoreNum, PLAT_CORE_NUM) << "恰好用满 64 核";
    EXPECT_EQ(td->blockFactor, 1);
    EXPECT_EQ(td->tailBlockTailUbFactor, MAX_UB_FP16) << "整除 → 尾块取满 ubFactor";
}

// 非对齐尾块
TEST_F(TensorRedirectTilingTest, tiling_fp16_1023_unaligned_tail)
{
    gert::StorageShape shape = {{1023}, {1023}};
    TilingInfo info;
    ASSERT_TRUE(SafeExecuteTiling(MakePara(shape, ge::DT_FLOAT16), info));
    CheckCommonInvariants(info, 1023, 2);

    const auto* td = AsTilingData(info);
    EXPECT_EQ(td->usedCoreNum, 1);
    EXPECT_EQ(td->blockFactor, 1);
    EXPECT_EQ(td->tailBlockTailUbFactor, 1023) << "非对齐尾块由 DataCopyPad 补齐/丢弃";
    // 提核优化后被下界钳制到 UB_FACTOR_MIN_BETY(2048) / 2 = 1024
    EXPECT_EQ(td->ubFactor, 1024);
}

// #2：单元素 [1]，ubFactor 被下界钳制到 1024
TEST_F(TensorRedirectTilingTest, tiling_fp16_single_element)
{
    gert::StorageShape shape = {{1}, {1}};
    TilingInfo info;
    ASSERT_TRUE(SafeExecuteTiling(MakePara(shape, ge::DT_FLOAT16), info));
    CheckCommonInvariants(info, 1, 2);

    const auto* td = AsTilingData(info);
    EXPECT_EQ(td->usedCoreNum, 1);
    EXPECT_EQ(td->blockFactor, 1);
    EXPECT_EQ(td->tailBlockTailUbFactor, 1) << "实际只搬 1 个元素";
    EXPECT_EQ(td->ubFactor, 1024) << "下界钳制 UB_FACTOR_MIN_BETY(2048)/2";
}

// #3 小 shape [128]：提核优化触发但被下界钳制
TEST_F(TensorRedirectTilingTest, tiling_fp16_small_shape_128)
{
    gert::StorageShape shape = {{128}, {128}};
    TilingInfo info;
    ASSERT_TRUE(SafeExecuteTiling(MakePara(shape, ge::DT_FLOAT16), info));
    CheckCommonInvariants(info, 128, 2);

    const auto* td = AsTilingData(info);
    EXPECT_EQ(td->ubFactor, 1024);
    EXPECT_EQ(td->usedCoreNum, 1);
    EXPECT_EQ(td->tailBlockTailUbFactor, 128);
}

// #3 小 shape [1,1,64]：多维小 shape 展平为 1D
TEST_F(TensorRedirectTilingTest, tiling_fp16_small_shape_1x1x64)
{
    gert::StorageShape shape = {{1, 1, 64}, {1, 1, 64}};
    TilingInfo info;
    ASSERT_TRUE(SafeExecuteTiling(MakePara(shape, ge::DT_FLOAT16), info));
    CheckCommonInvariants(info, 64, 2);

    const auto* td = AsTilingData(info);
    EXPECT_EQ(td->tailBlockTailUbFactor, 64) << "rank 不进入 kernel，一律展平为 numel";
    EXPECT_EQ(td->usedCoreNum, 1);
}

// ubFactor 下界
TEST_F(TensorRedirectTilingTest, tiling_fp16_ub_factor_lower_bound_clamp)
{
    gert::StorageShape shape = {{4096}, {4096}};
    TilingInfo info;
    ASSERT_TRUE(SafeExecuteTiling(MakePara(shape, ge::DT_FLOAT16), info));
    CheckCommonInvariants(info, 4096, 2);

    const auto* td = AsTilingData(info);
    // FloorDiv(4096, 64) = 64 → CeilAlign(64,16) = 64 → < 1024 → 钳制为 1024
    EXPECT_EQ(td->ubFactor, 1024) << "ubFactor 下界 = UB_FACTOR_MIN_BETY(2048) / 2B";
    EXPECT_EQ(td->usedCoreNum, 4);
    EXPECT_EQ(td->blockFactor, 1);
}

// 四、单核守卫分支

// 单核 + 小 shape：不得因 FloorDiv(numel, 0) 得到错误 ubFactor
TEST_F(TensorRedirectTilingTest, tiling_single_core_guard_small_shape)
{
    gert::StorageShape shape = {{128}, {128}};
    TilingInfo info;
    ASSERT_TRUE(SafeExecuteTiling(MakePara(shape, ge::DT_FLOAT16, &g_compileInfoSingleCore), info));

    const auto* td = AsTilingData(info);
    ASSERT_NE(td, nullptr);
    EXPECT_EQ(td->usedCoreNum, 1);
    EXPECT_GE(info.blockNum, 1U);
    // 守卫生效的判据：ubFactor 保持 min(numel, maxUbAvailable)，**未**被 FloorDiv(numel,0)=numel 污染
    EXPECT_EQ(td->ubFactor, 128) << "单核场景 ubFactor 应保持 min(numel, maxUb)";
    EXPECT_EQ(td->tailBlockTailUbFactor, 128);
}

// 单核 + 大 shape：ubFactor 须严格钳在 UB 上界内（这正是「错误 ubFactor」会 UB 超限的场景）
TEST_F(TensorRedirectTilingTest, tiling_single_core_guard_large_shape_ub_bounded)
{
    gert::StorageShape shape = {{1048577}, {1048577}};
    constexpr int64_t numel = 1048577;
    TilingInfo info;
    ASSERT_TRUE(SafeExecuteTiling(MakePara(shape, ge::DT_FLOAT16, &g_compileInfoSingleCore), info));

    const auto* td = AsTilingData(info);
    ASSERT_NE(td, nullptr);
    EXPECT_EQ(td->usedCoreNum, 1);
    // 守卫失效将导致 UB 超限
    EXPECT_EQ(td->ubFactor, MAX_UB_FP16);
    EXPECT_LE(td->ubFactor * 2 * N_BUFFER, PLAT_UB_SIZE) << "单核下 ubFactor 仍不得超 UB 预算";
    EXPECT_EQ(td->blockFactor, 17); // CeilDiv(1048577, 63488) = 17
    EXPECT_EQ(td->tailBlockTailUbFactor, 32769);

    const int64_t covered = (td->tailBlockFactor - 1) * td->ubFactor + td->tailBlockTailUbFactor;
    EXPECT_EQ(covered, numel);
}

// 五、dtype 字节宽对 UB 切分的影响

// 1 BYTES 桶（int8）：maxUbAvailable = 253952/2/1 = 126976
TEST_F(TensorRedirectTilingTest, tiling_int8_1byte_bucket_ub_bound)
{
    gert::StorageShape shape = {{8388608}, {8388608}};
    TilingInfo info;
    ASSERT_TRUE(SafeExecuteTiling(MakePara(shape, ge::DT_INT8), info));
    CheckCommonInvariants(info, 8388608, 1);

    const auto* td = AsTilingData(info);
    EXPECT_EQ(td->ubFactor, PLAT_UB_SIZE / N_BUFFER / 1); // 126976
    EXPECT_EQ(td->ubFactor, 126976);
}

// 4 BYTES 桶（fp32）：maxUbAvailable = 253952/2/4 = 31744
TEST_F(TensorRedirectTilingTest, tiling_fp32_4byte_bucket)
{
    gert::StorageShape shape = {{1048577}, {1048577}};
    TilingInfo info;
    ASSERT_TRUE(SafeExecuteTiling(MakePara(shape, ge::DT_FLOAT), info));
    CheckCommonInvariants(info, 1048577, 4);

    const auto* td = AsTilingData(info);
    // 提核优化：CeilAlign(FloorDiv(1048577,63), 32/4=8) = CeilAlign(16644,8) = 16648
    EXPECT_EQ(td->ubFactor, 16648);
    EXPECT_EQ(td->usedCoreNum, 63);
    EXPECT_EQ(td->tailBlockTailUbFactor, 16401);
    EXPECT_EQ(td->ubFactor % (32 / 4), 0) << "4B dtype 的 32B 对齐 = 8 元素对齐";
}

// 8 BYTES 桶（int64）：maxUbAvailable = 253952/2/8 = 15872，多核多循环
TEST_F(TensorRedirectTilingTest, tiling_int64_8byte_bucket_multi_loop)
{
    gert::StorageShape shape = {{8388608}, {8388608}};
    TilingInfo info;
    ASSERT_TRUE(SafeExecuteTiling(MakePara(shape, ge::DT_INT64), info));
    CheckCommonInvariants(info, 8388608, 8);

    const auto* td = AsTilingData(info);
    EXPECT_EQ(td->ubFactor, PLAT_UB_SIZE / N_BUFFER / 8); // 15872
    EXPECT_EQ(td->ubFactor, 15872);
    EXPECT_EQ(td->usedCoreNum, 59);
    EXPECT_EQ(td->blockFactor, 9);
    EXPECT_GT(td->blockFactor, 1) << "8B dtype UB 块最小 → 循环次数最多";
}

// bf16（2 BYTES 桶，与 fp16 同桶）：须与 fp16 得到完全一致的切分
TEST_F(TensorRedirectTilingTest, tiling_bf16_same_bucket_as_fp16)
{
    gert::StorageShape shape = {{1048577}, {1048577}};
    TilingInfo infoBf16;
    TilingInfo infoFp16;
    ASSERT_TRUE(SafeExecuteTiling(MakePara(shape, ge::DT_BF16), infoBf16));
    ASSERT_TRUE(SafeExecuteTiling(MakePara(shape, ge::DT_FLOAT16), infoFp16));

    const auto* bf16 = AsTilingData(infoBf16);
    const auto* fp16 = AsTilingData(infoFp16);
    EXPECT_EQ(bf16->ubFactor, fp16->ubFactor) << "bf16/fp16 同属 2 BYTES 桶，切分必须一致";
    EXPECT_EQ(bf16->usedCoreNum, fp16->usedCoreNum);
    EXPECT_EQ(bf16->tailBlockTailUbFactor, fp16->tailBlockTailUbFactor);
    EXPECT_EQ(infoBf16.blockNum, infoFp16.blockNum);
}

// 无符号 dtype 与同宽有符号 dtype 同桶（uint16 vs int16）
TEST_F(TensorRedirectTilingTest, tiling_uint16_same_bucket_as_int16)
{
    gert::StorageShape shape = {{1048577}, {1048577}};
    TilingInfo infoU16;
    TilingInfo infoI16;
    ASSERT_TRUE(SafeExecuteTiling(MakePara(shape, ge::DT_UINT16), infoU16));
    ASSERT_TRUE(SafeExecuteTiling(MakePara(shape, ge::DT_INT16), infoI16));

    EXPECT_EQ(AsTilingData(infoU16)->ubFactor, AsTilingData(infoI16)->ubFactor);
    EXPECT_EQ(AsTilingData(infoU16)->usedCoreNum, AsTilingData(infoI16)->usedCoreNum);
}

// 无符号 dtype 与同宽有符号 dtype 同桶（uint8 vs int8）
TEST_F(TensorRedirectTilingTest, tiling_uint8_same_bucket_as_int8)
{
    gert::StorageShape shape = {{1048577}, {1048577}};
    TilingInfo infoU8;
    TilingInfo infoI8;
    ASSERT_TRUE(SafeExecuteTiling(MakePara(shape, ge::DT_UINT8), infoU8));
    ASSERT_TRUE(SafeExecuteTiling(MakePara(shape, ge::DT_INT8), infoI8));

    EXPECT_EQ(AsTilingData(infoU8)->ubFactor, AsTilingData(infoI8)->ubFactor);
    EXPECT_EQ(AsTilingData(infoU8)->usedCoreNum, AsTilingData(infoI8)->usedCoreNum);
}

// 无符号 dtype 与同宽有符号 dtype 同桶（uint32 vs int32）
TEST_F(TensorRedirectTilingTest, tiling_uint32_same_bucket_as_int32)
{
    gert::StorageShape shape = {{1048577}, {1048577}};
    TilingInfo infoU32;
    TilingInfo infoI32;
    ASSERT_TRUE(SafeExecuteTiling(MakePara(shape, ge::DT_UINT32), infoU32));
    ASSERT_TRUE(SafeExecuteTiling(MakePara(shape, ge::DT_INT32), infoI32));

    EXPECT_EQ(AsTilingData(infoU32)->ubFactor, AsTilingData(infoI32)->ubFactor);
    EXPECT_EQ(AsTilingData(infoU32)->usedCoreNum, AsTilingData(infoI32)->usedCoreNum);
}

// 无符号 dtype 与同宽有符号 dtype 同桶（uint64 vs int64）
TEST_F(TensorRedirectTilingTest, tiling_uint64_same_bucket_as_int64)
{
    gert::StorageShape shape = {{1048577}, {1048577}};
    TilingInfo infoU64;
    TilingInfo infoI64;
    ASSERT_TRUE(SafeExecuteTiling(MakePara(shape, ge::DT_UINT64), infoU64));
    ASSERT_TRUE(SafeExecuteTiling(MakePara(shape, ge::DT_INT64), infoI64));

    EXPECT_EQ(AsTilingData(infoU64)->ubFactor, AsTilingData(infoI64)->ubFactor);
    EXPECT_EQ(AsTilingData(infoU64)->usedCoreNum, AsTilingData(infoI64)->usedCoreNum);
}

// 六、平台参数双路径

// 回退路径可用性：CompileInfo 未填充时须回退查 platform，且**不得**失败或产出退化值
TEST_F(TensorRedirectTilingTest, tiling_platform_fallback_when_compileinfo_unpopulated)
{
    optiling::TensorRedirectCompileInfo emptyCompileInfo{0, 0};
    gert::StorageShape shape = {{1048577}, {1048577}};
    TilingInfo info;
    ASSERT_TRUE(SafeExecuteTiling(MakePara(shape, ge::DT_FLOAT16, &emptyCompileInfo), info))
        << "CompileInfo 未填充时必须回退直查 platform，而非失败";
    CheckCommonInvariants(info, 1048577, 2);

    const auto* td = AsTilingData(info);
    EXPECT_GE(info.blockNum, 1U);
}

// 双路径等价性：两路径须逐字段一致
TEST_F(TensorRedirectTilingTest, tiling_dual_path_equivalence_compileinfo_vs_platform)
{
    optiling::TensorRedirectCompileInfo emptyCompileInfo{0, 0};
    gert::StorageShape shape = {{1048577}, {1048577}};

    TilingInfo viaCompileInfo; // 路径 1：GE 图
    TilingInfo viaPlatform;    // 路径 2：platform 回退（ACLNN 单算子）
    ASSERT_TRUE(SafeExecuteTiling(MakePara(shape, ge::DT_FLOAT16, &g_compileInfo), viaCompileInfo));
    ASSERT_TRUE(SafeExecuteTiling(MakePara(shape, ge::DT_FLOAT16, &emptyCompileInfo), viaPlatform));

    const auto* a = AsTilingData(viaCompileInfo);
    const auto* b = AsTilingData(viaPlatform);
    EXPECT_EQ(a->usedCoreNum, b->usedCoreNum);
    EXPECT_EQ(a->ubFactor, b->ubFactor);
    EXPECT_EQ(a->blockFactor, b->blockFactor);
    EXPECT_EQ(a->tailBlockFactor, b->tailBlockFactor);
    EXPECT_EQ(a->tailBlockTailUbFactor, b->tailBlockTailUbFactor);
    EXPECT_EQ(viaCompileInfo.blockNum, viaPlatform.blockNum);
    EXPECT_EQ(viaCompileInfo.tilingKey, viaPlatform.tilingKey);
    EXPECT_EQ(memcmp(a, b, sizeof(TensorRedirectTilingData)), 0) << "双路径 TilingData 必须逐字节一致";
}

// 七、4 桶 × 分支维度全矩阵覆盖

// 7.1 小 shape × 4 桶：ubFactor 下界钳制
TEST_F(TensorRedirectTilingTest, tiling_bucket_matrix_small_shape_lower_bound_clamp)
{
    constexpr int64_t numel = 4096;
    for (const auto& bk : kBuckets) {
        SCOPED_TRACE(testing::Message() << "bucket=" << bk.name << " bytes=" << bk.bytes);
        gert::StorageShape shape = {{numel}, {numel}};
        TilingInfo info;
        ASSERT_TRUE(SafeExecuteTiling(MakePara(shape, bk.dtype), info));
        CheckCommonInvariants(info, numel, bk.bytes);

        const auto* td = AsTilingData(info);
        const int64_t expectUb = UbFactorMinFor(bk.bytes); // 2048 / 1024 / 512 / 256
        EXPECT_LT(numel, MaxUbFor(bk.bytes)) << "前置：该 shape 必须落在「小 shape」档位";
        EXPECT_EQ(td->ubFactor, expectUb) << "下界必须随字节宽变化，fp16 的 1024 不通用";
        EXPECT_EQ(td->usedCoreNum, numel / expectUb);
        EXPECT_EQ(td->blockFactor, 1);
        EXPECT_EQ(td->tailBlockFactor, 1);
        EXPECT_EQ(td->tailBlockTailUbFactor, expectUb) << "4096 被 ubFactor 整除 → 尾块取满";
    }
}

// 7.2 单元素 × 4 桶：下界钳制防 ubFactor=0
TEST_F(TensorRedirectTilingTest, tiling_bucket_matrix_single_element_clamp_guards_zero_ubfactor)
{
    constexpr int64_t numel = 1;
    for (const auto& bk : kBuckets) {
        SCOPED_TRACE(testing::Message() << "bucket=" << bk.name << " bytes=" << bk.bytes);
        gert::StorageShape shape = {{numel}, {numel}};
        TilingInfo info;
        ASSERT_TRUE(SafeExecuteTiling(MakePara(shape, bk.dtype), info));
        CheckCommonInvariants(info, numel, bk.bytes);

        const auto* td = AsTilingData(info);
        EXPECT_GT(td->ubFactor, 0) << "★ ubFactor==0 → CeilDiv(numel,0) 退化 → tiling 全错";
        EXPECT_EQ(td->ubFactor, UbFactorMinFor(bk.bytes)) << "钳制后恰为下界";
        EXPECT_EQ(td->usedCoreNum, 1);
        EXPECT_EQ(td->blockFactor, 1);
        EXPECT_EQ(td->tailBlockFactor, 1);
        EXPECT_EQ(td->tailBlockTailUbFactor, 1) << "实际只搬 1 个元素";
        EXPECT_GE(info.blockNum, 1U);
    }
}

// 7.3 大 shape × 4 桶：UB 上界 100%
TEST_F(TensorRedirectTilingTest, tiling_bucket_matrix_ub_upper_bound_100pct)
{
    for (const auto& bk : kBuckets) {
        SCOPED_TRACE(testing::Message() << "bucket=" << bk.name << " bytes=" << bk.bytes);
        const int64_t numel = UbFullThresholdFor(bk.bytes); // 8126464/4063232/2031616/1015808
        gert::StorageShape shape = {{numel}, {numel}};
        TilingInfo info;
        ASSERT_TRUE(SafeExecuteTiling(MakePara(shape, bk.dtype), info));
        CheckCommonInvariants(info, numel, bk.bytes);

        const auto* td = AsTilingData(info);
        EXPECT_EQ(td->ubFactor, MaxUbFor(bk.bytes)) << "达到门槛即取满 UB 上界";
        // 100% UB 且不越界
        EXPECT_EQ(td->ubFactor * bk.bytes * N_BUFFER, PLAT_UB_SIZE) << "UB 占用必须恰为 100%";
        EXPECT_EQ(td->usedCoreNum, PLAT_CORE_NUM) << "门槛处恰好用满 64 核";
        EXPECT_EQ(info.blockNum, static_cast<uint32_t>(PLAT_CORE_NUM));
        EXPECT_EQ(td->blockFactor, 1);
        EXPECT_EQ(td->tailBlockTailUbFactor, MaxUbFor(bk.bytes)) << "整除 → 尾块取满 ubFactor";
    }
}

// 7.4 【尾块 × 4 桶】门槛 +1 → 最极端尾块：tailBlockTailUbFactor == 1（单元素尾块）
//     该档位同时覆盖「100% UB + 多核 + blockFactor>1 + 极小非对齐尾块」。
//     尾块=1 是 DataCopyPad 非对齐搬运的最坏输入，必须逐桶验证不漏不重。
TEST_F(TensorRedirectTilingTest, tiling_bucket_matrix_tail_block_one_element)
{
    for (const auto& bk : kBuckets) {
        SCOPED_TRACE(testing::Message() << "bucket=" << bk.name << " bytes=" << bk.bytes);
        const int64_t numel = UbFullThresholdFor(bk.bytes) + 1;
        gert::StorageShape shape = {{numel}, {numel}};
        TilingInfo info;
        ASSERT_TRUE(SafeExecuteTiling(MakePara(shape, bk.dtype), info));
        // CheckCommonInvariants 内含「覆盖总量 == numel」恒等式 —— 尾块场景的核心判据
        CheckCommonInvariants(info, numel, bk.bytes);

        const auto* td = AsTilingData(info);
        EXPECT_EQ(td->ubFactor, MaxUbFor(bk.bytes)) << "超门槛 → 仍取满 UB 上界";
        EXPECT_EQ(td->tailBlockTailUbFactor, 1) << "★ 多出的 1 个元素独占一个尾块";
        EXPECT_EQ(td->usedCoreNum, 33);
        EXPECT_EQ(td->blockFactor, 2) << "uo=65 → 每核 2 个满块";
        EXPECT_EQ(td->tailBlockFactor, 1) << "尾核只剩 1 个块（即那个单元素尾块）";
        EXPECT_LT(td->tailBlockTailUbFactor, td->ubFactor) << "尾块必须严格小于满块，否则非尾块场景";
    }
}

// 7.5 提核优化 × 4 桶：整除分支
TEST_F(TensorRedirectTilingTest, tiling_bucket_matrix_core_boost_divisible_branch)
{
    constexpr int64_t numel = 131072; // 64 * 2048，且 131072 % 64 == 0
    static_assert(numel % PLAT_CORE_NUM == 0, "必须命中整除分支");
    for (const auto& bk : kBuckets) {
        SCOPED_TRACE(testing::Message() << "bucket=" << bk.name << " bytes=" << bk.bytes);
        gert::StorageShape shape = {{numel}, {numel}};
        TilingInfo info;
        ASSERT_TRUE(SafeExecuteTiling(MakePara(shape, bk.dtype), info));
        CheckCommonInvariants(info, numel, bk.bytes);

        const auto* td = AsTilingData(info);
        // 整除分支：ubFactor=2048
        EXPECT_EQ(td->ubFactor, 2048) << "整除分支必须用 totalCoreNum（64）而非 totalCoreNum-1（63）作除数";
        EXPECT_EQ(td->usedCoreNum, PLAT_CORE_NUM) << "提核优化的语义：把核用满";
        EXPECT_EQ(info.blockNum, static_cast<uint32_t>(PLAT_CORE_NUM));
        EXPECT_EQ(td->blockFactor, 1) << "提核优化后每核仅一次循环";
        EXPECT_EQ(td->tailBlockFactor, 1);
        EXPECT_EQ(td->tailBlockTailUbFactor, 2048) << "整除 → 尾块取满";
        EXPECT_EQ(td->ubFactor % ElemsPer32BFor(bk.bytes), 0) << "32B 对齐（DataCopyPad UB 侧硬约束）";

        if (bk.bytes > 1) {
            EXPECT_GT(td->ubFactor, UbFactorMinFor(bk.bytes))
                << "该桶 ubFactor 严格大于下界 → 值确由整除分支产生，非钳制兜底";
        }
    }
}

// 7.5b 提核优化 × 32B 对齐 × 4 桶
TEST_F(TensorRedirectTilingTest, tiling_bucket_matrix_core_boost_divisible_with_32b_align)
{
    constexpr int64_t numel = 131200; // 64 * 2050，整除；2050 在 4 个桶里均非 32B 对齐
    static_assert(numel % PLAT_CORE_NUM == 0, "必须命中整除分支");
    // 逐桶期望：CeilAlign(2050, 32/bytes)
    const int64_t expectUb[] = {2080, 2064, 2056, 2052};
    const int64_t expectTailUb[] = {160, 1168, 1672, 1924};
    for (size_t i = 0; i < sizeof(kBuckets) / sizeof(kBuckets[0]); ++i) {
        const auto& bk = kBuckets[i];
        SCOPED_TRACE(testing::Message() << "bucket=" << bk.name << " bytes=" << bk.bytes);
        gert::StorageShape shape = {{numel}, {numel}};
        TilingInfo info;
        ASSERT_TRUE(SafeExecuteTiling(MakePara(shape, bk.dtype), info));
        CheckCommonInvariants(info, numel, bk.bytes);

        const auto* td = AsTilingData(info);
        EXPECT_EQ(td->ubFactor, expectUb[i]) << "★ 对齐粒度必须随字节宽变化：CeilAlign(2050, 32/bytes)";
        EXPECT_EQ(td->ubFactor, CeilAlignRef(2050, ElemsPer32BFor(bk.bytes))) << "与公式一致";
        EXPECT_GT(td->ubFactor, 2050) << "对齐必须向上取整，否则 DataCopyPad UB 侧越界";
        EXPECT_EQ(td->ubFactor % ElemsPer32BFor(bk.bytes), 0);
        EXPECT_GT(td->ubFactor, UbFactorMinFor(bk.bytes)) << "未被下界钳制 → 值确由整除分支产生";
        EXPECT_EQ(td->usedCoreNum, PLAT_CORE_NUM);
        EXPECT_EQ(td->blockFactor, 1);
        EXPECT_EQ(td->tailBlockTailUbFactor, expectTailUb[i]);
    }
}

// 7.6 【提核优化：整除 vs 非整除 分支判别】同一量级下两分支必须产出**不同**的 ubFactor
//     若整除分支被删除（统一走 totalCoreNum-1），两者将相等 → 本用例失败。
TEST_F(TensorRedirectTilingTest, tiling_core_boost_divisible_vs_non_divisible_differ)
{
    gert::StorageShape divisibleShape = {{131072}, {131072}};    // 131072 % 64 == 0
    gert::StorageShape nonDivisibleShape = {{131073}, {131073}}; // 131073 % 64 == 1
    TilingInfo divInfo;
    TilingInfo nonDivInfo;
    ASSERT_TRUE(SafeExecuteTiling(MakePara(divisibleShape, ge::DT_FLOAT16), divInfo));
    ASSERT_TRUE(SafeExecuteTiling(MakePara(nonDivisibleShape, ge::DT_FLOAT16), nonDivInfo));
    CheckCommonInvariants(divInfo, 131072, 2);
    CheckCommonInvariants(nonDivInfo, 131073, 2);

    const auto* divTd = AsTilingData(divInfo);
    const auto* nonDivTd = AsTilingData(nonDivInfo);
    EXPECT_EQ(divTd->ubFactor, 2048) << "整除：FloorDiv(131072, 64) = 2048";
    // 非整除：FloorDiv(131073, 63) = 2080 → CeilAlign(2080, 16) = 2080
    EXPECT_EQ(nonDivTd->ubFactor, 2080) << "非整除：FloorDiv(131073, 63) = 2080";
    EXPECT_NE(divTd->ubFactor, nonDivTd->ubFactor)
        << "★ 两分支必须可区分；相等说明整除分支已失效（被统一成 totalCoreNum-1）";
}

// 7.7 空 Tensor × 补齐 1/4 BYTES 桶

// 1_BYTES 桶空 Tensor：早返回与 dtype 无关，但仍须逐桶锁定
TEST_F(TensorRedirectTilingTest, tiling_empty_tensor_int8_blockdim_must_not_be_zero)
{
    gert::StorageShape shape = {{0, 5}, {0, 5}};
    TilingInfo info;
    ASSERT_TRUE(SafeExecuteTiling(MakePara(shape, ge::DT_INT8), info));

    EXPECT_GE(info.blockNum, 1U) << "★ SetBlockDim(0) → LaunchKernelV2 107000/EE1003 静默失败";
    EXPECT_EQ(info.blockNum, 1U);
    ASSERT_EQ(info.workspaceSizes.size(), 1U);
    EXPECT_EQ(static_cast<size_t>(info.workspaceSizes[0]), EXPECT_WORKSPACE);
    const auto* td = AsTilingData(info);
    EXPECT_EQ(td->usedCoreNum, 0) << "TilingData 全零 → kernel 不下发有效计算";
    EXPECT_EQ(td->ubFactor, 0);
}

// 4_BYTES 桶空 Tensor
TEST_F(TensorRedirectTilingTest, tiling_empty_tensor_fp32_blockdim_must_not_be_zero)
{
    gert::StorageShape shape = {{3, 0}, {3, 0}};
    TilingInfo info;
    ASSERT_TRUE(SafeExecuteTiling(MakePara(shape, ge::DT_FLOAT), info));

    EXPECT_GE(info.blockNum, 1U) << "★ SetBlockDim(0) 反回归锚点";
    EXPECT_EQ(info.blockNum, 1U);
    EXPECT_EQ(static_cast<uint64_t>(info.tilingKey), EXPECT_TILING_KEY) << "空 Tensor 仍须下发 TilingKey";
    const auto* td = AsTilingData(info);
    EXPECT_EQ(td->usedCoreNum, 0);
}

// 空 Tensor 全 8 维（rank 上界 × 0 维）：GetShapeSize()==0 的路径与 rank 无关
TEST_F(TensorRedirectTilingTest, tiling_empty_tensor_8d_blockdim_must_not_be_zero)
{
    gert::StorageShape shape = {{2, 2, 2, 2, 2, 2, 2, 0}, {2, 2, 2, 2, 2, 2, 2, 0}};
    TilingInfo info;
    ASSERT_TRUE(SafeExecuteTiling(MakePara(shape, ge::DT_FLOAT16), info));
    EXPECT_GE(info.blockNum, 1U) << "★ SetBlockDim(0) 反回归锚点";
    EXPECT_EQ(info.blockNum, 1U);
}
