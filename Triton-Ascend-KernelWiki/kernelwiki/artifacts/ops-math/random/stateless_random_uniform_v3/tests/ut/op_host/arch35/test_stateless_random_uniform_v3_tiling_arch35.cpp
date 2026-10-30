/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_stateless_random_uniform_v3_tiling_arch35.cpp
 * \brief StatelessRandomUniformV3 Tiling UT（arch35）
 *
 * TilingData 布局（RandomUnifiedTilingDataStruct，sizeof=96 字节，12 个 int64_t）：
 *   [0]  usedCoreNum
 *   [1]  normalCoreProNum
 *   [2]  tailCoreProNum
 *   [3]  singleBufferSize
 *   [4]  key[0]|(key[1]<<32)         -- Mod 16 后由 kernel 从 GM 读取，tiling 恒为 0
 *   [5]  counter[0]|(counter[1]<<32) -- Mod 16 后由 kernel 从 GM 读取，tiling 恒为 0
 *   [6]  counter[2]|(counter[3]<<32) -- Mod 16 后由 kernel 从 GM 读取，tiling 恒为 0
 *   [7]  outputSize
 *   [8]  probTensorSize              -- 固定为 0
 *   [9]  sharedTmpBufSize            -- 固定为 0
 *   [10] keepProb_bits|(v3KernelMode<<32)
 *   [11] reserved|padding             -- 固定为 0
 *
 * 平台参数（TilingContextPara 默认值，由 platformInfo 路径获取）：
 *   coreNum  = 64
 *   ubSize   = 262144 (256 KB)
 *
 * bufNum 计算（由 config.getBufferNum 回调，BUFFER_NUM=2）：
 *   float32:  outputDtypeSize=4, coefVal=1, bufNum=4*(2+1)=12
 *   float16:  outputDtypeSize=2, coefVal=2, bufNum=2*(2+2)=8
 *   bfloat16: outputDtypeSize=2, coefVal=2, bufNum=2*(2+2)=8
 *
 * singleBufferSize = ubSize / bufNum / GetUbBlockSize(=32) * 32：
 *   float32:  262144/12/32*32 = 21845/32*32 = 682*32 = 21824
 *   float16:  262144/8/32*32  = 32768/32*32 = 32768
 *   bfloat16: 262144/8/32*32  = 32768
 *
 * coreAlignSize = 512，normalCoreProNum = max(align512(ceil(outputSize/coreNum)), 256)：
 *   outputSize=16384, ceil(16384/64)=256 → align512=512, max(512,256)=512
 *   usedCoreNum = ceil(16384/512) = 32, tailCoreProNum = 16384 - 512*31 = 512
 *
 *   outputSize=32768（全核）: ceil(32768/64)=512 → align512=512
 *   usedCoreNum=64, tailCoreProNum=32768-512*63=512
 *
 *   outputSize=20000（非对齐尾块）: ceil(20000/64)=313 → align512=512
 *   usedCoreNum=ceil(20000/512)=40, tailCoreProNum=20000-512*39=32
 *
 *   outputSize=19500（非对齐尾块）: ceil(19500/64)=305 → align512=512
 *   usedCoreNum=ceil(19500/512)=39, tailCoreProNum=19500-512*38=44
 *
 *   outputSize=16（极小）: ceil(16/64)=1 → align512=512, max(512,256)=512
 *   usedCoreNum=1, tailCoreProNum=16
 *
 * tilingData[10] 编码：keepProb(0.0f)=0 | (v3KernelMode<<32)
 *   sm0: 0
 *   sm1: 1LL<<32 = 4294967296
 *
 * 分支覆盖矩阵（dtype × v3KernelMode）：
 *   float32  × sm0: test_0
 *   float32  × sm1: test_3
 *   float16  × sm0: test_4
 *   float16  × sm1: test_1
 *   bfloat16 × sm0: test_2
 *   bfloat16 × sm1: test_5
 *
 * shape 维度覆盖：
 *   1D: test_6
 *   2D: test_0~5, test_9~12
 *   3D: test_7
 *   4D: test_8
 *
 * 边界覆盖：
 *   极小 shape（outputSize<512，单核）:     test_9
 *   全核（usedCoreNum=64，tail=normal）:    test_10
 *   非对齐尾块（tailCoreProNum≠normalCoreProNum）: test_11, test_12
 *
 * 非法用例：
 *   非法输出 dtype（DT_INT32）:  test_invalid_dtype
 *   非法输出 dtype（DT_DOUBLE）: test_invalid_float64
 *   非法输出 dtype（DT_INT64）:  test_invalid_int64
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../../op_host/arch35/stateless_random_uniform_v3_tiling_arch35.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class StatelessRandomUniformV3TilingTest : public testing::Test
{
protected:
    static void StatelessRandomUniformV3TestCase()
    {
        std::cout << "StatelessRandomUniformV3TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "StatelessRandomUniformV3TilingTest TearDown" << std::endl;
    }
};

// case 0: int32 shape, float32 output [32,512], v3KernelMode=0
// singleBufferSize=21824, tilingData[10]=0
TEST_F(StatelessRandomUniformV3TilingTest, stateless_random_uniform_v3_test_0)
{
    optiling::RandomOperatorCompileInfo compileInfo = {64, 262144};
    vector<int32_t> shapeValue = {32, 512};

    gert::TilingContextPara tilingContextPara(
        "StatelessRandomUniformV3",
    {
        {{{2},{2}}, ge::DT_INT32, ge::FORMAT_ND, true, shapeValue.data()},
        {{{1},{1}}, ge::DT_UINT64, ge::FORMAT_ND},
        {{{2},{2}}, ge::DT_UINT64, ge::FORMAT_ND},
        {{{1},{1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{1},{1}}, ge::DT_FLOAT, ge::FORMAT_ND},
    },
    {
        {{{32,512},{32,512}}, ge::DT_FLOAT, ge::FORMAT_ND},
    },
    {
        gert::TilingContextPara::OpAttr("dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
        gert::TilingContextPara::OpAttr("v3KernelMode", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
    },
    &compileInfo);
    uint64_t expectTilingKey = 100;
    string expectTilingData = "32 512 512 21824 0 0 0 16384 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// case 1: int64 shape, float16 output [64,256], v3KernelMode=1
// singleBufferSize=32768, tilingData[10]=v3KernelMode<<32=4294967296
TEST_F(StatelessRandomUniformV3TilingTest, stateless_random_uniform_v3_test_1)
{
    optiling::RandomOperatorCompileInfo compileInfo = {64, 262144};
    vector<int64_t> shapeValue = {64, 256};

    gert::TilingContextPara tilingContextPara(
        "StatelessRandomUniformV3",
    {
        {{{2},{2}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
        {{{1},{1}}, ge::DT_UINT64, ge::FORMAT_ND},
        {{{2},{2}}, ge::DT_UINT64, ge::FORMAT_ND},
        {{{1},{1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{1},{1}}, ge::DT_FLOAT, ge::FORMAT_ND},
    },
    {
        {{{64,256},{64,256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        gert::TilingContextPara::OpAttr("dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
        gert::TilingContextPara::OpAttr("v3KernelMode", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
    },
    &compileInfo);
    uint64_t expectTilingKey = 100;
    // tilingData[10]: keepProb(0.0f)=0x00000000, v3KernelMode=1 → int64 = (1LL<<32) = 4294967296
    string expectTilingData = "32 512 512 32768 0 0 0 16384 0 0 0 1 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// case 2: int32 shape, bfloat16 output [32,512], v3KernelMode=0
// singleBufferSize=32768, tilingData[10]=0
TEST_F(StatelessRandomUniformV3TilingTest, stateless_random_uniform_v3_test_2)
{
    optiling::RandomOperatorCompileInfo compileInfo = {64, 262144};
    vector<int32_t> shapeValue = {32, 512};

    gert::TilingContextPara tilingContextPara(
        "StatelessRandomUniformV3",
    {
        {{{2},{2}}, ge::DT_INT32, ge::FORMAT_ND, true, shapeValue.data()},
        {{{1},{1}}, ge::DT_UINT64, ge::FORMAT_ND},
        {{{2},{2}}, ge::DT_UINT64, ge::FORMAT_ND},
        {{{1},{1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{1},{1}}, ge::DT_FLOAT, ge::FORMAT_ND},
    },
    {
        {{{32,512},{32,512}}, ge::DT_BF16, ge::FORMAT_ND},
    },
    {
        gert::TilingContextPara::OpAttr("dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(3)),
        gert::TilingContextPara::OpAttr("v3KernelMode", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
    },
    &compileInfo);
    uint64_t expectTilingKey = 100;
    string expectTilingData = "32 512 512 32768 0 0 0 16384 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// ===== dtype × v3KernelMode 覆盖补全 =====

// case 3: int32 shape, float32 output [32,512], v3KernelMode=1
// singleBufferSize=21824, tilingData[10]=1LL<<32=4294967296
TEST_F(StatelessRandomUniformV3TilingTest, stateless_random_uniform_v3_test_3)
{
    optiling::RandomOperatorCompileInfo compileInfo = {64, 262144};
    vector<int32_t> shapeValue = {32, 512};

    gert::TilingContextPara tilingContextPara(
        "StatelessRandomUniformV3",
    {
        {{{2},{2}}, ge::DT_INT32, ge::FORMAT_ND, true, shapeValue.data()},
        {{{1},{1}}, ge::DT_UINT64, ge::FORMAT_ND},
        {{{2},{2}}, ge::DT_UINT64, ge::FORMAT_ND},
        {{{1},{1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{1},{1}}, ge::DT_FLOAT, ge::FORMAT_ND},
    },
    {
        {{{32,512},{32,512}}, ge::DT_FLOAT, ge::FORMAT_ND},
    },
    {
        gert::TilingContextPara::OpAttr("dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
        gert::TilingContextPara::OpAttr("v3KernelMode", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
    },
    &compileInfo);
    uint64_t expectTilingKey = 100;
    string expectTilingData = "32 512 512 21824 0 0 0 16384 0 0 0 1 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// case 4: int32 shape, float16 output [32,512], v3KernelMode=0
// singleBufferSize=32768, tilingData[10]=0
TEST_F(StatelessRandomUniformV3TilingTest, stateless_random_uniform_v3_test_4)
{
    optiling::RandomOperatorCompileInfo compileInfo = {64, 262144};
    vector<int32_t> shapeValue = {32, 512};

    gert::TilingContextPara tilingContextPara(
        "StatelessRandomUniformV3",
    {
        {{{2},{2}}, ge::DT_INT32, ge::FORMAT_ND, true, shapeValue.data()},
        {{{1},{1}}, ge::DT_UINT64, ge::FORMAT_ND},
        {{{2},{2}}, ge::DT_UINT64, ge::FORMAT_ND},
        {{{1},{1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{1},{1}}, ge::DT_FLOAT, ge::FORMAT_ND},
    },
    {
        {{{32,512},{32,512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        gert::TilingContextPara::OpAttr("dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
        gert::TilingContextPara::OpAttr("v3KernelMode", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
    },
    &compileInfo);
    uint64_t expectTilingKey = 100;
    string expectTilingData = "32 512 512 32768 0 0 0 16384 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// case 5: int64 shape, bfloat16 output [32,512], v3KernelMode=1
// singleBufferSize=32768, tilingData[10]=4294967296
TEST_F(StatelessRandomUniformV3TilingTest, stateless_random_uniform_v3_test_5)
{
    optiling::RandomOperatorCompileInfo compileInfo = {64, 262144};
    vector<int64_t> shapeValue = {32, 512};

    gert::TilingContextPara tilingContextPara(
        "StatelessRandomUniformV3",
    {
        {{{2},{2}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
        {{{1},{1}}, ge::DT_UINT64, ge::FORMAT_ND},
        {{{2},{2}}, ge::DT_UINT64, ge::FORMAT_ND},
        {{{1},{1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{1},{1}}, ge::DT_FLOAT, ge::FORMAT_ND},
    },
    {
        {{{32,512},{32,512}}, ge::DT_BF16, ge::FORMAT_ND},
    },
    {
        gert::TilingContextPara::OpAttr("dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(3)),
        gert::TilingContextPara::OpAttr("v3KernelMode", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
    },
    &compileInfo);
    uint64_t expectTilingKey = 100;
    string expectTilingData = "32 512 512 32768 0 0 0 16384 0 0 0 1 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// ===== shape 维度覆盖 =====

// case 6: 1D int64 shape [16384], float32 output, v3KernelMode=0
// 覆盖 1D shape 解析路径
TEST_F(StatelessRandomUniformV3TilingTest, stateless_random_uniform_v3_test_6)
{
    optiling::RandomOperatorCompileInfo compileInfo = {64, 262144};
    vector<int64_t> shapeValue = {16384};

    gert::TilingContextPara tilingContextPara(
        "StatelessRandomUniformV3",
    {
        {{{1},{1}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
        {{{1},{1}}, ge::DT_UINT64, ge::FORMAT_ND},
        {{{2},{2}}, ge::DT_UINT64, ge::FORMAT_ND},
        {{{1},{1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{1},{1}}, ge::DT_FLOAT, ge::FORMAT_ND},
    },
    {
        {{{16384},{16384}}, ge::DT_FLOAT, ge::FORMAT_ND},
    },
    {
        gert::TilingContextPara::OpAttr("dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
        gert::TilingContextPara::OpAttr("v3KernelMode", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
    },
    &compileInfo);
    uint64_t expectTilingKey = 100;
    string expectTilingData = "32 512 512 21824 0 0 0 16384 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// case 7: 3D int32 shape [4,64,64]=16384, float16 output, v3KernelMode=0
// 覆盖 3D shape 解析路径
TEST_F(StatelessRandomUniformV3TilingTest, stateless_random_uniform_v3_test_7)
{
    optiling::RandomOperatorCompileInfo compileInfo = {64, 262144};
    vector<int32_t> shapeValue = {4, 64, 64};

    gert::TilingContextPara tilingContextPara(
        "StatelessRandomUniformV3",
    {
        {{{3},{3}}, ge::DT_INT32, ge::FORMAT_ND, true, shapeValue.data()},
        {{{1},{1}}, ge::DT_UINT64, ge::FORMAT_ND},
        {{{2},{2}}, ge::DT_UINT64, ge::FORMAT_ND},
        {{{1},{1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{1},{1}}, ge::DT_FLOAT, ge::FORMAT_ND},
    },
    {
        {{{4,64,64},{4,64,64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        gert::TilingContextPara::OpAttr("dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
        gert::TilingContextPara::OpAttr("v3KernelMode", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
    },
    &compileInfo);
    uint64_t expectTilingKey = 100;
    string expectTilingData = "32 512 512 32768 0 0 0 16384 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// case 8: 4D int64 shape [2,8,32,32]=16384, bfloat16 output, v3KernelMode=1
// 覆盖 4D shape 解析路径
TEST_F(StatelessRandomUniformV3TilingTest, stateless_random_uniform_v3_test_8)
{
    optiling::RandomOperatorCompileInfo compileInfo = {64, 262144};
    vector<int64_t> shapeValue = {2, 8, 32, 32};

    gert::TilingContextPara tilingContextPara(
        "StatelessRandomUniformV3",
    {
        {{{4},{4}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
        {{{1},{1}}, ge::DT_UINT64, ge::FORMAT_ND},
        {{{2},{2}}, ge::DT_UINT64, ge::FORMAT_ND},
        {{{1},{1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{1},{1}}, ge::DT_FLOAT, ge::FORMAT_ND},
    },
    {
        {{{2,8,32,32},{2,8,32,32}}, ge::DT_BF16, ge::FORMAT_ND},
    },
    {
        gert::TilingContextPara::OpAttr("dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(3)),
        gert::TilingContextPara::OpAttr("v3KernelMode", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
    },
    &compileInfo);
    uint64_t expectTilingKey = 100;
    string expectTilingData = "32 512 512 32768 0 0 0 16384 0 0 0 1 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// ===== 边界 shape 覆盖 =====

// case 9: 极小 shape，1D int32 [16]，float32，v3KernelMode=0
// outputSize=16 < MIN_CORE_PRO=256 → usedCoreNum=1, normalCoreProNum=512, tailCoreProNum=16
TEST_F(StatelessRandomUniformV3TilingTest, stateless_random_uniform_v3_test_9)
{
    optiling::RandomOperatorCompileInfo compileInfo = {64, 262144};
    vector<int32_t> shapeValue = {16};

    gert::TilingContextPara tilingContextPara(
        "StatelessRandomUniformV3",
    {
        {{{1},{1}}, ge::DT_INT32, ge::FORMAT_ND, true, shapeValue.data()},
        {{{1},{1}}, ge::DT_UINT64, ge::FORMAT_ND},
        {{{2},{2}}, ge::DT_UINT64, ge::FORMAT_ND},
        {{{1},{1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{1},{1}}, ge::DT_FLOAT, ge::FORMAT_ND},
    },
    {
        {{{16},{16}}, ge::DT_FLOAT, ge::FORMAT_ND},
    },
    {
        gert::TilingContextPara::OpAttr("dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
        gert::TilingContextPara::OpAttr("v3KernelMode", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
    },
    &compileInfo);
    uint64_t expectTilingKey = 100;
    // ceil(16/64)=1 → align512=512, max(512,256)=512 → normalCoreProNum=512
    // usedCoreNum=ceil(16/512)=1, tailCoreProNum=16
    string expectTilingData = "1 512 16 21824 0 0 0 16 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// case 10: 全核场景，2D int64 [64,512]=32768，float32，v3KernelMode=1
// usedCoreNum=64（全部64核），normalCoreProNum=512，tailCoreProNum=512（整除）
TEST_F(StatelessRandomUniformV3TilingTest, stateless_random_uniform_v3_test_10)
{
    optiling::RandomOperatorCompileInfo compileInfo = {64, 262144};
    vector<int64_t> shapeValue = {64, 512};

    gert::TilingContextPara tilingContextPara(
        "StatelessRandomUniformV3",
    {
        {{{2},{2}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
        {{{1},{1}}, ge::DT_UINT64, ge::FORMAT_ND},
        {{{2},{2}}, ge::DT_UINT64, ge::FORMAT_ND},
        {{{1},{1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{1},{1}}, ge::DT_FLOAT, ge::FORMAT_ND},
    },
    {
        {{{64,512},{64,512}}, ge::DT_FLOAT, ge::FORMAT_ND},
    },
    {
        gert::TilingContextPara::OpAttr("dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
        gert::TilingContextPara::OpAttr("v3KernelMode", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
    },
    &compileInfo);
    uint64_t expectTilingKey = 100;
    // ceil(32768/64)=512 → align512=512, usedCoreNum=64, tailCoreProNum=32768-512*63=512
    string expectTilingData = "64 512 512 21824 0 0 0 32768 0 0 0 1 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// case 11: 非对齐尾块，2D int32 [100,200]=20000，float16，v3KernelMode=0
// usedCoreNum=40，normalCoreProNum=512，tailCoreProNum=32（20000-512*39=32）
TEST_F(StatelessRandomUniformV3TilingTest, stateless_random_uniform_v3_test_11)
{
    optiling::RandomOperatorCompileInfo compileInfo = {64, 262144};
    vector<int32_t> shapeValue = {100, 200};

    gert::TilingContextPara tilingContextPara(
        "StatelessRandomUniformV3",
    {
        {{{2},{2}}, ge::DT_INT32, ge::FORMAT_ND, true, shapeValue.data()},
        {{{1},{1}}, ge::DT_UINT64, ge::FORMAT_ND},
        {{{2},{2}}, ge::DT_UINT64, ge::FORMAT_ND},
        {{{1},{1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{1},{1}}, ge::DT_FLOAT, ge::FORMAT_ND},
    },
    {
        {{{100,200},{100,200}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        gert::TilingContextPara::OpAttr("dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
        gert::TilingContextPara::OpAttr("v3KernelMode", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
    },
    &compileInfo);
    uint64_t expectTilingKey = 100;
    // ceil(20000/64)=313 → align512=512, usedCoreNum=ceil(20000/512)=40
    // tailCoreProNum=20000-512*39=20000-19968=32
    string expectTilingData = "40 512 32 32768 0 0 0 20000 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// case 12: 非对齐尾块，2D int64 [150,130]=19500，bfloat16，v3KernelMode=1
// usedCoreNum=39，normalCoreProNum=512，tailCoreProNum=44（19500-512*38=44）
TEST_F(StatelessRandomUniformV3TilingTest, stateless_random_uniform_v3_test_12)
{
    optiling::RandomOperatorCompileInfo compileInfo = {64, 262144};
    vector<int64_t> shapeValue = {150, 130};

    gert::TilingContextPara tilingContextPara(
        "StatelessRandomUniformV3",
    {
        {{{2},{2}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
        {{{1},{1}}, ge::DT_UINT64, ge::FORMAT_ND},
        {{{2},{2}}, ge::DT_UINT64, ge::FORMAT_ND},
        {{{1},{1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{1},{1}}, ge::DT_FLOAT, ge::FORMAT_ND},
    },
    {
        {{{150,130},{150,130}}, ge::DT_BF16, ge::FORMAT_ND},
    },
    {
        gert::TilingContextPara::OpAttr("dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(3)),
        gert::TilingContextPara::OpAttr("v3KernelMode", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
    },
    &compileInfo);
    uint64_t expectTilingKey = 100;
    // ceil(19500/64)=305 → align512=512, usedCoreNum=ceil(19500/512)=39
    // tailCoreProNum=19500-512*38=19500-19456=44
    string expectTilingData = "39 512 44 32768 0 0 0 19500 0 0 0 1 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// ===== 非法用例 =====

// case 13: unsupported output dtype (DT_INT32) → GRAPH_FAILED
TEST_F(StatelessRandomUniformV3TilingTest, stateless_random_uniform_v3_test_invalid_dtype)
{
    optiling::RandomOperatorCompileInfo compileInfo = {64, 262144};
    vector<int32_t> shapeValue = {32, 512};

    gert::TilingContextPara tilingContextPara(
        "StatelessRandomUniformV3",
    {
        {{{2},{2}}, ge::DT_INT32, ge::FORMAT_ND, true, shapeValue.data()},
        {{{1},{1}}, ge::DT_UINT64, ge::FORMAT_ND},
        {{{2},{2}}, ge::DT_UINT64, ge::FORMAT_ND},
        {{{1},{1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{1},{1}}, ge::DT_FLOAT, ge::FORMAT_ND},
    },
    {
        {{{32,512},{32,512}}, ge::DT_INT32, ge::FORMAT_ND},
    },
    {
        gert::TilingContextPara::OpAttr("dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
        gert::TilingContextPara::OpAttr("v3KernelMode", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
    },
    &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// case 14: unsupported output dtype (DT_DOUBLE) → GRAPH_FAILED
TEST_F(StatelessRandomUniformV3TilingTest, stateless_random_uniform_v3_test_invalid_float64)
{
    optiling::RandomOperatorCompileInfo compileInfo = {64, 262144};
    vector<int32_t> shapeValue = {32, 512};

    gert::TilingContextPara tilingContextPara(
        "StatelessRandomUniformV3",
    {
        {{{2},{2}}, ge::DT_INT32, ge::FORMAT_ND, true, shapeValue.data()},
        {{{1},{1}}, ge::DT_UINT64, ge::FORMAT_ND},
        {{{2},{2}}, ge::DT_UINT64, ge::FORMAT_ND},
        {{{1},{1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{1},{1}}, ge::DT_FLOAT, ge::FORMAT_ND},
    },
    {
        {{{32,512},{32,512}}, ge::DT_DOUBLE, ge::FORMAT_ND},
    },
    {
        gert::TilingContextPara::OpAttr("dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
        gert::TilingContextPara::OpAttr("v3KernelMode", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
    },
    &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// case 15: unsupported output dtype (DT_INT64) → GRAPH_FAILED
TEST_F(StatelessRandomUniformV3TilingTest, stateless_random_uniform_v3_test_invalid_int64)
{
    optiling::RandomOperatorCompileInfo compileInfo = {64, 262144};
    vector<int64_t> shapeValue = {32, 512};

    gert::TilingContextPara tilingContextPara(
        "StatelessRandomUniformV3",
    {
        {{{2},{2}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
        {{{1},{1}}, ge::DT_UINT64, ge::FORMAT_ND},
        {{{2},{2}}, ge::DT_UINT64, ge::FORMAT_ND},
        {{{1},{1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{1},{1}}, ge::DT_FLOAT, ge::FORMAT_ND},
    },
    {
        {{{32,512},{32,512}}, ge::DT_INT64, ge::FORMAT_ND},
    },
    {
        gert::TilingContextPara::OpAttr("dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
        gert::TilingContextPara::OpAttr("v3KernelMode", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
    },
    &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}