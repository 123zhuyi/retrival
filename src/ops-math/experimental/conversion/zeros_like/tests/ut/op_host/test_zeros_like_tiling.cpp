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
 * \file test_zeros_like_tiling.cpp
 * \brief experimental 自包含 ascend910b (DAV_2201) 标准 AscendC tiling 单元测试。
 *        被测对象：experimental/conversion/zeros_like/op_host/zeros_like_tiling.cpp
 *                  （IMPL_OP_OPTILING(ZerosLike) for 910b，按总字节数 32B 对齐块多核切分，
 *                   字节宽度 TilingKey 1/2/4/8，workspace=0，0 元素保护）。
 *        覆盖：
 *          1) 基础 shape 多核均衡切分（主线 fp16，blockDim/perCoreBytes/tailCoreNum/totalBytes）
 *          2) TilingKey 按字节宽度选择（2B 主路径 + 验证 1B/4B/8B 各 dtype 命中正确）
 *          3) 0 元素（空 tensor）保护 + 单元素
 *          4) workspace=0
 *          5) 不均衡切分（tailCoreNum>0）、大 shape 满载、shape/dtype 异常
 */

#include "../../../op_host/zeros_like_tiling.h"
#include "../../../op_kernel/zeros_like_tiling_data.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;
using ZerosLikeNs::ZerosLikeTilingData;

namespace {
// 复用 kernel/host 共享的 32B 对齐基本块常量（ZerosLikeNs::ZL_BLOCK_BYTES，uint32_t），
// 避免在 UT 内重复定义导致数值/类型不一致。
using ZerosLikeNs::ZL_BLOCK_BYTES;
// 字节宽度桶（= kernel template<int BYTE_KEY> 实参；与 zeros_like_tiling_key.h 的 ZL_KEY_* 一致）。
constexpr uint32_t ZL_BYTE_1B = 1;
constexpr uint32_t ZL_BYTE_2B = 2;
constexpr uint32_t ZL_BYTE_4B = 4;
constexpr uint32_t ZL_BYTE_8B = 8;

// 默认平台：910B3 风格，核数 / UB 由各用例 compileInfo 显式给定（禁止依赖隐式默认）。
constexpr uint64_t DEFAULT_UB = 262144;
constexpr uint32_t DEFAULT_CORE = 64;

uint64_t CeilDiv(uint64_t a, uint64_t b)
{
    return (b == 0) ? 0 : ((a + b - 1) / b);
}

// 运行时 GetTilingKey() 的语义说明：
//   tiling 内部用 ASCENDC_TPL_SEL_PARAM(context, bytesPerElem) 选择模板实参，
//   框架将 (byteKey) 在声明列表 ASCENDC_TPL_UINT_DECL(byteKey, ..., 1,2,4,8) 中的「下标」
//   作为运行时 TilingKey 返回（与 ndtri/is_finite 等 arch35 算子一致：UT 固化运行时实际值，
//   而非传入的字面值）。kernel 侧 template<int BYTE_KEY> 仍收到字面字节值 1/2/4/8。
// 因此：1B->index0, 2B->index1, 4B->index2, 8B->index3。
uint64_t ByteWidthToTilingKeyIndex(uint32_t bytesPerElem)
{
    switch (bytesPerElem) {
        case ZL_BYTE_1B:
            return 0;
        case ZL_BYTE_2B:
            return 1;
        case ZL_BYTE_4B:
            return 2;
        case ZL_BYTE_8B:
            return 3;
        default:
            return UINT64_MAX; // 不应出现
    }
}

// StorageShape 的构造函数仅接受 initializer_list，这里用 AppendDim 从 vector 动态构造
// （origin 与 storage 同形）。
gert::StorageShape MakeShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape ss;
    for (auto d : dims) {
        ss.MutableOriginShape().AppendDim(d);
        ss.MutableStorageShape().AppendDim(d);
    }
    return ss;
}

const ZerosLikeTilingData* AsTilingData(const TilingInfo& info)
{
    EXPECT_GE(info.tilingDataSize, sizeof(ZerosLikeTilingData));
    return reinterpret_cast<const ZerosLikeTilingData*>(info.tilingData.get());
}

// 构造一个 910b CompileInfo（{totalCoreNum, ubSize}）。
optiling::ZerosLikeCompileInfo MakeCompileInfo(int32_t coreNum, int64_t ubSize)
{
    optiling::ZerosLikeCompileInfo info;
    info.totalCoreNum = coreNum;
    info.ubSize = ubSize;
    return info;
}

// 期望的切分结果（与被测 tiling 逻辑独立重算，作为 oracle）。
struct ExpectSplit {
    uint64_t totalBytes;
    uint64_t perCoreBytes;
    uint64_t tailCoreNum;
    uint32_t usedCoreNum;
    uint32_t bytesPerElem;
};

ExpectSplit ComputeExpect(uint64_t elemNum, uint32_t bytesPerElem, uint32_t coreNum)
{
    ExpectSplit e{};
    e.bytesPerElem = bytesPerElem;
    e.totalBytes = elemNum * bytesPerElem;
    if (e.totalBytes == 0) {
        e.perCoreBytes = 0;
        e.tailCoreNum = 0;
        e.usedCoreNum = 1;
        return e;
    }
    uint64_t totalBlock = CeilDiv(e.totalBytes, ZL_BLOCK_BYTES);
    uint32_t usedCore = coreNum;
    if (totalBlock < usedCore) {
        usedCore = static_cast<uint32_t>(totalBlock);
    }
    e.usedCoreNum = usedCore;
    e.perCoreBytes = (totalBlock / usedCore) * ZL_BLOCK_BYTES;
    e.tailCoreNum = totalBlock % usedCore;
    return e;
}

// 跑一次 tiling 并对结果做完整字段校验（成功路径）。
// expectBytesPerElem：该 dtype 归一后的字节宽度（1/2/4/8），用于校验 TilingData.bytesPerElem，
//                     并据此推导运行时 GetTilingKey() 的期望下标。
void RunAndCheck(
    const std::vector<int64_t>& shape, ge::DataType dtype, uint32_t coreNum, uint64_t ubSize,
    uint32_t expectBytesPerElem)
{
    optiling::ZerosLikeCompileInfo compileInfo =
        MakeCompileInfo(static_cast<int32_t>(coreNum), static_cast<int64_t>(ubSize));
    gert::StorageShape ss = MakeShape(shape);
    gert::TilingContextPara para(
        "ZerosLike", {{ss, dtype, ge::FORMAT_ND}}, {{ss, dtype, ge::FORMAT_ND}}, &compileInfo, coreNum, ubSize);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info)) << "tiling should return GRAPH_SUCCESS";

    // 元素总数
    uint64_t elemNum = 1;
    bool isEmpty = false;
    for (auto d : shape) {
        if (d == 0) {
            isEmpty = true;
        }
        elemNum *= static_cast<uint64_t>(d);
    }
    if (isEmpty) {
        elemNum = 0;
    }

    ExpectSplit exp = ComputeExpect(elemNum, expectBytesPerElem, coreNum);
    const ZerosLikeTilingData* td = AsTilingData(info);

    // 运行时 TilingKey == 字节宽度桶在声明列表中的下标（框架编码，见 ByteWidthToTilingKeyIndex 注释）
    EXPECT_EQ(static_cast<uint64_t>(info.tilingKey), ByteWidthToTilingKeyIndex(expectBytesPerElem));
    // blockDim == usedCoreNum
    EXPECT_EQ(info.blockNum, exp.usedCoreNum);
    // workspace == 0
    ASSERT_EQ(info.workspaceSizes.size(), 1u);
    EXPECT_EQ(info.workspaceSizes[0], 0);
    // TilingData 字段
    EXPECT_EQ(td->totalBytes, exp.totalBytes);
    EXPECT_EQ(td->perCoreBytes, exp.perCoreBytes);
    EXPECT_EQ(td->tailCoreNum, exp.tailCoreNum);
    EXPECT_EQ(td->usedCoreNum, exp.usedCoreNum);
    // bytesPerElem（kernel template<int BYTE_KEY> 实参的来源）保留字面字节宽度 1/2/4/8
    EXPECT_EQ(td->bytesPerElem, expectBytesPerElem);
    // tileBytes 32B 对齐且非 0
    EXPECT_EQ(td->tileBytes % ZL_BLOCK_BYTES, 0u);
    EXPECT_GT(td->tileBytes, 0u);
}
} // namespace

class test_zeros_like_tiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "test_zeros_like_tiling SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "test_zeros_like_tiling TearDown" << std::endl;
    }
};

// ============================================================================
// 1) 基础 shape 多核均衡切分（主线 fp16，2B 桶）
//    shape {1,64,2,32}=4096 elem × 2B = 8192B；totalBlock=256；coreNum=64 → 整除：
//    usedCore=64, perCoreBytes=4*32=128, tailCoreNum=0。
// ============================================================================
TEST_F(test_zeros_like_tiling, test_tiling_fp16_basic_even_split)
{
    RunAndCheck({1, 64, 2, 32}, ge::DT_FLOAT16, DEFAULT_CORE, DEFAULT_UB, ZL_BYTE_2B);
}

// ============================================================================
// 2) 主线 fp16 不均衡切分（tailCoreNum > 0）
//    shape {1000} × 2B = 2000B；totalBlock=ceil(2000/32)=63；coreNum=8 →
//    usedCore=8, blockPerCore=7, tailCoreNum=63%8=7, perCoreBytes=7*32=224。
// ============================================================================
TEST_F(test_zeros_like_tiling, test_tiling_fp16_uneven_split_tailcore)
{
    RunAndCheck({1000}, ge::DT_FLOAT16, 8, DEFAULT_UB, ZL_BYTE_2B);
}

// ============================================================================
// 3) TilingKey 按字节宽度选择 —— 1B 桶（int8 / uint8 / bool 均归 1）
// ============================================================================
TEST_F(test_zeros_like_tiling, test_tiling_int8_key1b)
{
    RunAndCheck({64}, ge::DT_INT8, DEFAULT_CORE, DEFAULT_UB, ZL_BYTE_1B);
}

TEST_F(test_zeros_like_tiling, test_tiling_uint8_key1b)
{
    RunAndCheck({128}, ge::DT_UINT8, DEFAULT_CORE, DEFAULT_UB, ZL_BYTE_1B);
}

TEST_F(test_zeros_like_tiling, test_tiling_bool_key1b)
{
    RunAndCheck({256}, ge::DT_BOOL, DEFAULT_CORE, DEFAULT_UB, ZL_BYTE_1B);
}

// ============================================================================
// TilingKey 2B 桶（fp16 已在主线验证，这里补 bf16）
// ============================================================================
TEST_F(test_zeros_like_tiling, test_tiling_bf16_key2b)
{
    RunAndCheck({32}, ge::DT_BF16, DEFAULT_CORE, DEFAULT_UB, ZL_BYTE_2B);
}

// ============================================================================
// TilingKey 4B 桶（fp32 / int32 均归 4）
// ============================================================================
TEST_F(test_zeros_like_tiling, test_tiling_fp32_key4b)
{
    RunAndCheck({256}, ge::DT_FLOAT, DEFAULT_CORE, DEFAULT_UB, ZL_BYTE_4B);
}

TEST_F(test_zeros_like_tiling, test_tiling_int32_key4b)
{
    RunAndCheck({256}, ge::DT_INT32, DEFAULT_CORE, DEFAULT_UB, ZL_BYTE_4B);
}

// ============================================================================
// TilingKey 8B 桶（int64 归 8）
// ============================================================================
TEST_F(test_zeros_like_tiling, test_tiling_int64_key8b)
{
    RunAndCheck({16}, ge::DT_INT64, DEFAULT_CORE, DEFAULT_UB, ZL_BYTE_8B);
}

// ============================================================================
// 4) 0 元素（空 tensor）保护：shape {0} → totalBytes=0,
//    blockDim=1, perCoreBytes=0, tailCoreNum=0, usedCoreNum=1, TilingKey 仍按 dtype 选。
// ============================================================================
TEST_F(test_zeros_like_tiling, test_tiling_empty_tensor_fp16)
{
    RunAndCheck({0}, ge::DT_FLOAT16, DEFAULT_CORE, DEFAULT_UB, ZL_BYTE_2B);
}

TEST_F(test_zeros_like_tiling, test_tiling_empty_tensor_multidim)
{
    // {4, 0, 8} 的元素数也为 0
    RunAndCheck({4, 0, 8}, ge::DT_INT32, DEFAULT_CORE, DEFAULT_UB, ZL_BYTE_4B);
}

// ============================================================================
// 单元素保护：shape {1} fp16 → 2B；totalBlock=ceil(2/32)=1；usedCore=min(64,1)=1。
// ============================================================================
TEST_F(test_zeros_like_tiling, test_tiling_single_element_fp16)
{
    RunAndCheck({1}, ge::DT_FLOAT16, DEFAULT_CORE, DEFAULT_UB, ZL_BYTE_2B);
}

TEST_F(test_zeros_like_tiling, test_tiling_single_element_int64)
{
    // {1} int64 → 8B；totalBlock=1；usedCore=1。
    RunAndCheck({1}, ge::DT_INT64, DEFAULT_CORE, DEFAULT_UB, ZL_BYTE_8B);
}

// ============================================================================
// rank=0 标量 [] —— Shape::GetShapeSize() 对 dim_num_==0 返回 1（单元素退化）。
//    fp32(4B)：totalBytes=4，totalBlock=ceil(4/32)=1，usedCore=min(64,1)=1，
//    perCoreBytes=32，tailCoreNum=0，blockDim=1。验证标量路径不崩、TilingKey/字段正确。
// ============================================================================
TEST_F(test_zeros_like_tiling, test_tiling_rank0_scalar_fp32)
{
    RunAndCheck({}, ge::DT_FLOAT, DEFAULT_CORE, DEFAULT_UB, ZL_BYTE_4B);

    // 显式复核 blockDim==1（标量单元素只用 1 核）
    optiling::ZerosLikeCompileInfo compileInfo = MakeCompileInfo(DEFAULT_CORE, static_cast<int64_t>(DEFAULT_UB));
    gert::StorageShape ss = MakeShape({}); // rank-0：无 AppendDim，dim_num_=0
    gert::TilingContextPara para(
        "ZerosLike", {{ss, ge::DT_FLOAT, ge::FORMAT_ND}}, {{ss, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo,
        DEFAULT_CORE, DEFAULT_UB);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info)) << "rank-0 scalar tiling should succeed";
    EXPECT_EQ(info.blockNum, 1u);
    const ZerosLikeTilingData* td = AsTilingData(info);
    EXPECT_EQ(td->totalBytes, 4u); // 1 元素 × 4B
    EXPECT_EQ(td->usedCoreNum, 1u);
    EXPECT_EQ(td->tailCoreNum, 0u);
}

// ============================================================================
// 8 维 rank 上限 [2,1,2,1,2,1,2,3]=48 elem，验证多核切分 + 尾块路径。
//    int64(8B)：totalBytes=48×8=384，totalBlock=ceil(384/32)=12；coreNum=5 →
//    usedCore=5, blockPerCore=12/5=2, tailCoreNum=12%5=2(>0 尾块), perCoreBytes=2*32=64。
//    8 维 shape 经 GetShapeSize 折算元素数，tiling 不依赖维度数 → 路径与低维等价，
//    本用例固化 rank=MAX_DIM_LEN(8) 下 tiling 不崩 + 不均衡切分(tailCoreNum>0)正确。
// ============================================================================
TEST_F(test_zeros_like_tiling, test_tiling_8dim_int64_multicore_tail)
{
    const uint32_t coreNum = 5;
    RunAndCheck({2, 1, 2, 1, 2, 1, 2, 3}, ge::DT_INT64, coreNum, DEFAULT_UB, ZL_BYTE_8B);

    // 显式复核 tailCoreNum>0（不均衡切分）+ blockDim==usedCore + 8 维元素折算
    optiling::ZerosLikeCompileInfo compileInfo = MakeCompileInfo(static_cast<int32_t>(coreNum), DEFAULT_UB);
    gert::StorageShape ss = MakeShape({2, 1, 2, 1, 2, 1, 2, 3});
    gert::TilingContextPara para(
        "ZerosLike", {{ss, ge::DT_INT64, ge::FORMAT_ND}}, {{ss, ge::DT_INT64, ge::FORMAT_ND}}, &compileInfo, coreNum,
        DEFAULT_UB);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info)) << "8-dim tiling should succeed";
    const ZerosLikeTilingData* td = AsTilingData(info);
    EXPECT_EQ(td->totalBytes, 48u * 8u); // 48 元素 × 8B
    EXPECT_EQ(td->usedCoreNum, 5u);
    EXPECT_EQ(info.blockNum, 5u);
    EXPECT_GT(td->tailCoreNum, 0u); // 不均衡：前 tailCoreNum 核各多 1 个 32B 块
    EXPECT_EQ(td->tailCoreNum, 2u);
    EXPECT_EQ(td->perCoreBytes, 64u);
}

// ============================================================================
// 5) 大 shape 多核满载 + 数据搬运守恒检查（各核字节范围之和覆盖 totalBytes）
//    shape {200000} fp32 → 800000B；totalBlock=25000；coreNum=48 → usedCore=48。
//    blockPerCore=25000/48=520, tailCoreNum=25000%48=40, perCoreBytes=520*32=16640。
//    覆盖守恒：前 tailCoreNum 核各 (blockPerCore+1) 块，其余核 blockPerCore 块，
//    合计字节 = totalBlock*32 >= totalBytes（尾零头由 kernel DataCopyPad 处理）。
// ============================================================================
TEST_F(test_zeros_like_tiling, test_tiling_large_shape_full_multicore)
{
    const uint32_t coreNum = 48;
    RunAndCheck({200000}, ge::DT_FLOAT, coreNum, DEFAULT_UB, ZL_BYTE_4B);

    // 额外做覆盖守恒校验
    optiling::ZerosLikeCompileInfo compileInfo =
        MakeCompileInfo(static_cast<int32_t>(coreNum), static_cast<int64_t>(DEFAULT_UB));
    gert::StorageShape ss = MakeShape({200000});
    gert::TilingContextPara para(
        "ZerosLike", {{ss, ge::DT_FLOAT, ge::FORMAT_ND}}, {{ss, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo, coreNum,
        DEFAULT_UB);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    const ZerosLikeTilingData* td = AsTilingData(info);

    uint64_t totalBlock = CeilDiv(td->totalBytes, ZL_BLOCK_BYTES);
    // 各核块数之和 = totalBlock（前 tailCoreNum 核 +1 块）
    uint64_t blockPerCore = td->perCoreBytes / ZL_BLOCK_BYTES;
    uint64_t sumBlock = blockPerCore * td->usedCoreNum + td->tailCoreNum;
    EXPECT_EQ(sumBlock, totalBlock);
    // 32B 对齐块总字节 >= totalBytes，且不足一块的零头 < 32B
    EXPECT_GE(totalBlock * ZL_BLOCK_BYTES, td->totalBytes);
    EXPECT_LT(totalBlock * ZL_BLOCK_BYTES - td->totalBytes, ZL_BLOCK_BYTES);
}

// ============================================================================
// 多核块数少于核数：小 shape 时 usedCore=totalBlock，避免空转核。
//    shape {64} int8 → 64B；totalBlock=2；coreNum=64 → usedCore=2。
// ============================================================================
TEST_F(test_zeros_like_tiling, test_tiling_smallshape_usedcore_clamp)
{
    RunAndCheck({64}, ge::DT_INT8, DEFAULT_CORE, DEFAULT_UB, ZL_BYTE_1B);

    optiling::ZerosLikeCompileInfo compileInfo = MakeCompileInfo(64, static_cast<int64_t>(DEFAULT_UB));
    gert::StorageShape ss = MakeShape({64});
    gert::TilingContextPara para(
        "ZerosLike", {{ss, ge::DT_INT8, ge::FORMAT_ND}}, {{ss, ge::DT_INT8, ge::FORMAT_ND}}, &compileInfo, DEFAULT_CORE,
        DEFAULT_UB);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.blockNum, 2u); // totalBlock=2 < coreNum=64
}

// ============================================================================
// 异常路径：tiling 防御性失败
// ============================================================================
// 不支持的 dtype（def 仅 8 种，但 tiling 应对未知字节宽度返回 GRAPH_FAILED）
TEST_F(test_zeros_like_tiling, test_tiling_unsupported_dtype_fail)
{
    optiling::ZerosLikeCompileInfo compileInfo = MakeCompileInfo(64, static_cast<int64_t>(DEFAULT_UB));
    gert::StorageShape ss = MakeShape({16});
    gert::TilingContextPara para(
        "ZerosLike", {{ss, ge::DT_DOUBLE, ge::FORMAT_ND}}, {{ss, ge::DT_DOUBLE, ge::FORMAT_ND}}, &compileInfo,
        DEFAULT_CORE, DEFAULT_UB);
    ExecuteTestCase(para, ge::GRAPH_FAILED, 0, std::vector<size_t>{});
}

// 输入/输出 dtype 不一致 → GRAPH_FAILED
TEST_F(test_zeros_like_tiling, test_tiling_dtype_mismatch_fail)
{
    optiling::ZerosLikeCompileInfo compileInfo = MakeCompileInfo(64, static_cast<int64_t>(DEFAULT_UB));
    gert::StorageShape ss = MakeShape({16});
    gert::TilingContextPara para(
        "ZerosLike", {{ss, ge::DT_FLOAT16, ge::FORMAT_ND}}, {{ss, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo,
        DEFAULT_CORE, DEFAULT_UB);
    ExecuteTestCase(para, ge::GRAPH_FAILED, 0, std::vector<size_t>{});
}

// 输入/输出 shape 元素数不一致 → GRAPH_FAILED
TEST_F(test_zeros_like_tiling, test_tiling_shape_mismatch_fail)
{
    optiling::ZerosLikeCompileInfo compileInfo = MakeCompileInfo(64, static_cast<int64_t>(DEFAULT_UB));
    gert::StorageShape ssIn = MakeShape({16});
    gert::StorageShape ssOut = MakeShape({32});
    gert::TilingContextPara para(
        "ZerosLike", {{ssIn, ge::DT_FLOAT16, ge::FORMAT_ND}}, {{ssOut, ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo,
        DEFAULT_CORE, DEFAULT_UB);
    ExecuteTestCase(para, ge::GRAPH_FAILED, 0, std::vector<size_t>{});
}
