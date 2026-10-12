/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <gtest/gtest.h>
#include "tiling_case_executor.h"
#include "../../../op_kernel/bitwise_not_tiling_data.h"

// 平台基线：与 tiling_context_faker 默认一致（coreNum=64, ubSize=262144），
// 期望值由离线模拟 bitwise_not_tiling.cpp 的整数运算精确给出（见 test-report.md）。
// 期望 workspace = 0（逐元素无用户 workspace，host 写 currentWorkspace[0]=0）。

// 逐字段比对（含整合新增的 lastCoreId / lastCoreTailDataNum 尾块对齐字段）。
static bool operator==(const BitwiseNotTilingData& l, const BitwiseNotTilingData& r)
{
    return l.smallCoreDataNum == r.smallCoreDataNum && l.bigCoreDataNum == r.bigCoreDataNum &&
           l.bigCoreLoopNum == r.bigCoreLoopNum && l.smallCoreLoopNum == r.smallCoreLoopNum &&
           l.ubPartDataNum == r.ubPartDataNum && l.smallCoreTailDataNum == r.smallCoreTailDataNum &&
           l.bigCoreTailDataNum == r.bigCoreTailDataNum && l.tailBlockNum == r.tailBlockNum && l.isBool == r.isBool &&
           l.lastCoreId == r.lastCoreId && l.lastCoreTailDataNum == r.lastCoreTailDataNum;
}

static std::ostream& operator<<(std::ostream& os, const BitwiseNotTilingData& t)
{
    os << "{smallCoreDataNum=" << t.smallCoreDataNum << ", bigCoreDataNum=" << t.bigCoreDataNum
       << ", bigCoreLoopNum=" << t.bigCoreLoopNum << ", smallCoreLoopNum=" << t.smallCoreLoopNum
       << ", ubPartDataNum=" << t.ubPartDataNum << ", smallCoreTailDataNum=" << t.smallCoreTailDataNum
       << ", bigCoreTailDataNum=" << t.bigCoreTailDataNum << ", tailBlockNum=" << t.tailBlockNum
       << ", isBool=" << t.isBool << ", lastCoreId=" << t.lastCoreId
       << ", lastCoreTailDataNum=" << t.lastCoreTailDataNum << "}";
    return os;
}

class BitwiseNotTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "BitwiseNotTiling SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "BitwiseNotTiling TearDown" << std::endl;
    }
};

// 从运行时 vector 构造 gert::StorageShape（StorageShape 仅提供 initializer_list 构造，故用 AppendDim）。
static gert::StorageShape MakeStorageShape(const std::vector<int64_t>& shape)
{
    gert::StorageShape s;
    for (int64_t d : shape) {
        s.MutableShape().AppendDim(d);
        s.MutableStorageShape().AppendDim(d);
    }
    return s;
}

// 通用执行：跑 tiling，校验 GRAPH 状态 / tilingKey / workspace / 全字段 tilingData。
static void RunTilingCase(
    const std::vector<int64_t>& shape, ge::DataType dtype, uint64_t expectTilingKey,
    const BitwiseNotTilingData& expectTilingData, size_t expectBlockDim)
{
    struct BitwiseNotCompileInfo {
    } compileInfo;
    gert::StorageShape storageShape = MakeStorageShape(shape);
    gert::TilingContextPara tilingContextPara(
        "BitwiseNot",
        {
            {storageShape, dtype, ge::FORMAT_ND},
        },
        {
            {storageShape, dtype, ge::FORMAT_ND},
        },
        &compileInfo);

    TilingInfo tilingInfo;
    bool ok = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(ok) << "tiling expected GRAPH_SUCCESS";

    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
    EXPECT_EQ(tilingInfo.blockNum, expectBlockDim);
    // workspace 恒为 0（逐元素无用户 workspace）。
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1u);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], 0);

    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(BitwiseNotTilingData));
    const BitwiseNotTilingData& got = *reinterpret_cast<BitwiseNotTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(got, expectTilingData) << "got=" << got << " expect=" << expectTilingData;
}

// 构造期望 TilingData 的便捷工具。
static BitwiseNotTilingData MakeTiling(
    uint32_t smallCoreDataNum, uint32_t bigCoreDataNum, uint32_t bigCoreLoopNum, uint32_t smallCoreLoopNum,
    uint32_t ubPartDataNum, uint32_t smallCoreTailDataNum, uint32_t bigCoreTailDataNum, uint32_t tailBlockNum,
    uint32_t isBool, uint32_t lastCoreId, uint32_t lastCoreTailDataNum)
{
    BitwiseNotTilingData t{};
    t.smallCoreDataNum = smallCoreDataNum;
    t.bigCoreDataNum = bigCoreDataNum;
    t.bigCoreLoopNum = bigCoreLoopNum;
    t.smallCoreLoopNum = smallCoreLoopNum;
    t.ubPartDataNum = ubPartDataNum;
    t.smallCoreTailDataNum = smallCoreTailDataNum;
    t.bigCoreTailDataNum = bigCoreTailDataNum;
    t.tailBlockNum = tailBlockNum;
    t.isBool = isBool;
    t.lastCoreId = lastCoreId;
    t.lastCoreTailDataNum = lastCoreTailDataNum;
    return t;
}

// ============================================================================
// 1) dtype 分支 / isBool 标志：6 dtype，整型 isBool=0、BOOL isBool=1（小 shape 单核 key0）
//    小 shape [8]：每 dtype 单核单 tile，验证 isBool 标志与 ubPartDataNum 随 dtype 字节数变化。
// ============================================================================
TEST_F(BitwiseNotTiling, ascend910b_dtype_int8_isbool0)
{
    // int8 [8]: ubPartDataNum=43648, small=32(1块对齐), pad=24, lastTail=8
    RunTilingCase({8}, ge::DT_INT8, 0, MakeTiling(32, 0, 0, 1, 43648, 32, 0, 0, 0, 0, 8), 1);
}

TEST_F(BitwiseNotTiling, ascend910b_dtype_int16_isbool0)
{
    // int16 [8]: ubPartDataNum=21824, small=16(1块=16元素), pad=8, lastTail=8
    RunTilingCase({8}, ge::DT_INT16, 0, MakeTiling(16, 0, 0, 1, 21824, 16, 0, 0, 0, 0, 8), 1);
}

TEST_F(BitwiseNotTiling, ascend910b_dtype_int32_isbool0)
{
    // int32 [8]: ubPartDataNum=10912, small=8(1块=8元素), pad=0, lastTail=8
    RunTilingCase({8}, ge::DT_INT32, 0, MakeTiling(8, 0, 0, 1, 10912, 8, 0, 0, 0, 0, 8), 1);
}

TEST_F(BitwiseNotTiling, ascend910b_dtype_int64_isbool0)
{
    // int64 [8]: ubPartDataNum=5456, small=8(1块=4元素→2块), pad=0, lastTail=8
    RunTilingCase({8}, ge::DT_INT64, 0, MakeTiling(8, 0, 0, 1, 5456, 8, 0, 0, 0, 0, 8), 1);
}

TEST_F(BitwiseNotTiling, ascend910b_dtype_uint8_isbool0)
{
    // uint8 [8]: 与 int8 同（1 字节），ubPartDataNum=43648, small=32, pad=24, lastTail=8
    RunTilingCase({8}, ge::DT_UINT8, 0, MakeTiling(32, 0, 0, 1, 43648, 32, 0, 0, 0, 0, 8), 1);
}

TEST_F(BitwiseNotTiling, ascend910b_dtype_bool_isbool1)
{
    // bool [8]: 与 int8 同字节但 isBool=1（区分 INT8 位翻转 vs BOOL 逻辑非）
    RunTilingCase({8}, ge::DT_BOOL, 0, MakeTiling(32, 0, 0, 1, 43648, 32, 0, 0, 1, 0, 8), 1);
}

// ============================================================================
// 2) 空 Tensor [0]：workspace=0 提前返回，blockDim=1，tilingKey=0，全字段清零
// ============================================================================
TEST_F(BitwiseNotTiling, ascend910b_empty_tensor_int16)
{
    RunTilingCase({0}, ge::DT_INT16, 0, MakeTiling(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), 1);
}

TEST_F(BitwiseNotTiling, ascend910b_empty_tensor_bool)
{
    // 空 tensor 也写 isBool（host 在 0 元素分支设置 isBool 后返回）
    RunTilingCase({0}, ge::DT_BOOL, 0, MakeTiling(0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0), 1);
}

// ============================================================================
// 3) 非 32B 对齐尾块（lastCoreId / lastCoreTailDataNum）：每 dtype 至少一例，
//    覆盖 [17]/[33]/[100]/[641]，lastCoreTailDataNum = 真实元素数（≠ 32B 对齐 smallCoreTailDataNum）。
// ============================================================================
TEST_F(BitwiseNotTiling, ascend910b_tail_int8_17)
{
    // int8 [17]: small=32, pad=15, lastTail=17
    RunTilingCase({17}, ge::DT_INT8, 0, MakeTiling(32, 0, 0, 1, 43648, 32, 0, 0, 0, 0, 17), 1);
}

TEST_F(BitwiseNotTiling, ascend910b_tail_uint8_17)
{
    // uint8 [17]: 同 int8（1 字节）
    RunTilingCase({17}, ge::DT_UINT8, 0, MakeTiling(32, 0, 0, 1, 43648, 32, 0, 0, 0, 0, 17), 1);
}

TEST_F(BitwiseNotTiling, ascend910b_tail_int16_17)
{
    // int16 [17]: small=32(2块*16), pad=15, lastTail=17
    RunTilingCase({17}, ge::DT_INT16, 0, MakeTiling(32, 0, 0, 1, 21824, 32, 0, 0, 0, 0, 17), 1);
}

TEST_F(BitwiseNotTiling, ascend910b_tail_int32_17)
{
    // int32 [17]: small=24(3块*8), pad=7, lastTail=17
    RunTilingCase({17}, ge::DT_INT32, 0, MakeTiling(24, 0, 0, 1, 10912, 24, 0, 0, 0, 0, 17), 1);
}

TEST_F(BitwiseNotTiling, ascend910b_tail_int64_17)
{
    // int64 [17]: small=20(5块*4), pad=3, lastTail=17
    RunTilingCase({17}, ge::DT_INT64, 0, MakeTiling(20, 0, 0, 1, 5456, 20, 0, 0, 0, 0, 17), 1);
}

TEST_F(BitwiseNotTiling, ascend910b_tail_bool_17)
{
    // bool [17]: 同 int8 字节但 isBool=1, lastTail=17
    RunTilingCase({17}, ge::DT_BOOL, 0, MakeTiling(32, 0, 0, 1, 43648, 32, 0, 0, 1, 0, 17), 1);
}

TEST_F(BitwiseNotTiling, ascend910b_tail_int8_33)
{
    // int8 [33]: small=64(2块*32), pad=31, lastTail=33
    RunTilingCase({33}, ge::DT_INT8, 0, MakeTiling(64, 0, 0, 1, 43648, 64, 0, 0, 0, 0, 33), 1);
}

TEST_F(BitwiseNotTiling, ascend910b_tail_int16_33)
{
    // int16 [33]: small=48(3块*16), pad=15, lastTail=33
    RunTilingCase({33}, ge::DT_INT16, 0, MakeTiling(48, 0, 0, 1, 21824, 48, 0, 0, 0, 0, 33), 1);
}

TEST_F(BitwiseNotTiling, ascend910b_tail_int32_100)
{
    // int32 [100]: small=104(13块*8), pad=4, lastTail=100
    RunTilingCase({100}, ge::DT_INT32, 0, MakeTiling(104, 0, 0, 1, 10912, 104, 0, 0, 0, 0, 100), 1);
}

TEST_F(BitwiseNotTiling, ascend910b_tail_uint8_100)
{
    // uint8 [100]: small=128(4块*32), pad=28, lastTail=100
    RunTilingCase({100}, ge::DT_UINT8, 0, MakeTiling(128, 0, 0, 1, 43648, 128, 0, 0, 0, 0, 100), 1);
}

TEST_F(BitwiseNotTiling, ascend910b_tail_int64_641)
{
    // int64 [641]: small=644(161块*4), pad=3, lastTail=641
    RunTilingCase({641}, ge::DT_INT64, 0, MakeTiling(644, 0, 0, 1, 5456, 644, 0, 0, 0, 0, 641), 1);
}

TEST_F(BitwiseNotTiling, ascend910b_tail_bool_33)
{
    // bool [33]: small=64, pad=31, isBool=1, lastTail=33
    RunTilingCase({33}, ge::DT_BOOL, 0, MakeTiling(64, 0, 0, 1, 43648, 64, 0, 0, 1, 0, 33), 1);
}

// ============================================================================
// 4) big/small core 切分（tilingKey=1）：大 shape 触发 tailBlockNum!=0
// ============================================================================
TEST_F(BitwiseNotTiling, ascend910b_bigsmall_int8_1000000)
{
    // int8 [1000000]: 64 核, tailBlockNum=18, small=15616, big=15648, key1, pad=0, lastTail=15616
    RunTilingCase({1000000}, ge::DT_INT8, 1, MakeTiling(15616, 15648, 1, 1, 43648, 15616, 15648, 18, 0, 63, 15616), 64);
}

TEST_F(BitwiseNotTiling, ascend910b_bigsmall_int64_500000)
{
    // int64 [500000]: 64 核, tailBlockNum=8, small=7812, big=7816, loop=2, key1, pad=0,
    // smallTail=2356, bigTail=2360, lastTail=2356（ubPartDataNum=5456 决定尾 tile 元素数）
    RunTilingCase({500000}, ge::DT_INT64, 1, MakeTiling(7812, 7816, 2, 2, 5456, 2356, 2360, 8, 0, 63, 2356), 64);
}

// ============================================================================
// 5) 单核大 tile（tilingKey=0，多元素对齐，pad=0）：标准 elementwise 主路径
// ============================================================================
TEST_F(BitwiseNotTiling, ascend910b_singlecore_int16_1Mi)
{
    // int16 [1048576]: 64 核全 small（无 tailBlock）, small=16384, key0, pad=0, lastTail=16384
    RunTilingCase({1024, 1024}, ge::DT_INT16, 0, MakeTiling(16384, 0, 0, 1, 21824, 16384, 0, 0, 0, 63, 16384), 64);
}

// rank=2 标准小 shape [2,3]（spec boundary：整型按位非主路径，pad=10, lastTail=6）
TEST_F(BitwiseNotTiling, ascend910b_rank2_int16_2x3)
{
    RunTilingCase({2, 3}, ge::DT_INT16, 0, MakeTiling(16, 0, 0, 1, 21824, 16, 0, 0, 0, 0, 6), 1);
}

// rank=0 标量 []（element=1，pad=15, lastTail=1）
TEST_F(BitwiseNotTiling, ascend910b_rank0_scalar_int16)
{
    RunTilingCase({}, ge::DT_INT16, 0, MakeTiling(16, 0, 0, 1, 21824, 16, 0, 0, 0, 0, 1), 1);
}
