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
 * \file test_sort_with_index_tiling_arch35.cpp
 * \brief SortWithIndex算子arch35架构Tiling测试用例
 */

#include "../../../../op_host/arch35/sort_with_index_tiling.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class SortWithIndexTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SortWithIndexTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SortWithIndexTilingTest TearDown" << std::endl;
    }
};

/**
 * 测试用例：FLOAT16小规模归并排序
 * 
 * 测试目的：验证FLOAT16数据类型在排序轴长度<=512时触发SMALL_SIZE_OPTIM_MODE归并排序模式
 * 
 * 数据类型：DT_FLOAT16 (tilingKey基数=3002)
 * 
 * 输入形状：[4, 128] - 4行数据，每行128个元素需要排序
 *            unsortedDimNum=4（待排序行数），sortAxisNum=128（排序轴长度）
 * 
 * 算子属性：
 *   - axis: -1（最后一维，即axis=1）
 *   - descending: true（降序排序）
 *   - stable: true（稳定排序）
 * 
 * 排序模式：SMALL_SIZE_OPTIM_MODE（小规模优化模式，使用归并排序）
 *   - 触发条件：sortAxisNum <= 512 && dataType in [FLOAT, FLOAT16, BF16]
 *   - tilingKey计算：baseKey + MERGE_SORT_TILING_OFFSET = 3002 + 10000 = 13002
 * 
 */
TEST_F(SortWithIndexTilingTest, test_tiling_float16_small_size_merge_sort) {
    optiling::SortWithIndexCompileInfo compileInfo;
    compileInfo.core_num = 64;

    gert::TilingContextPara tilingContextPara("SortWithIndex",
        {
            {{{4, 128}, {4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 128}, {4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)},
            {"descending", Ops::Math::AnyValue::CreateFrom<bool>(true)},
            {"stable", Ops::Math::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 13002;
    string expectTilingData = "4294967297 1 4 4294967297 128 68719476736 4294967424 128 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

/**
 * 测试用例：FLOAT小规模归并排序
 * 
 * 测试目的：验证FLOAT数据类型在排序轴长度<=512时触发SMALL_SIZE_OPTIM_MODE归并排序模式
 * 
 * 数据类型：DT_FLOAT (tilingKey基数=3003)
 * 
 * 输入形状：[2, 256] - 2行数据，每行256个元素需要排序
 *            unsortedDimNum=2，sortAxisNum=256
 * 
 * 算子属性：
 *   - axis: -1（最后一维）
 *   - descending: true（降序排序）
 *   - stable: true（稳定排序）
 * 
 * 排序模式：SMALL_SIZE_OPTIM_MODE（归并排序）
 *   - 触发条件：sortAxisNum=256 <= 512 && dataType=FLOAT
 *   - tilingKey = 3003 + 10000 = 13003
 * 
 */
TEST_F(SortWithIndexTilingTest, test_tiling_float_small_size_merge_sort) {
    optiling::SortWithIndexCompileInfo compileInfo;
    compileInfo.core_num = 64;

    gert::TilingContextPara tilingContextPara("SortWithIndex",
        {
            {{{2, 256}, {2, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 256}, {2, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)},
            {"descending", Ops::Math::AnyValue::CreateFrom<bool>(true)},
            {"stable", Ops::Math::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 13003;
    string expectTilingData = "4294967297 1 2 4294967297 256 34359738368 4294967552 256 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

/**
 * 测试用例：BF16小规模归并排序
 * 
 * 测试目的：验证BF16数据类型在排序轴长度<=512时触发SMALL_SIZE_OPTIM_MODE归并排序模式
 *           并验证INT64索引输出类型的支持
 * 
 * 数据类型：DT_BF16 (tilingKey基数=4002)
 * 输出索引类型：DT_INT64
 * 
 * 输入形状：[2, 512] - 2行数据，每行512个元素需要排序
 *            unsortedDimNum=2，sortAxisNum=512（边界值，刚好触发归并排序）
 * 
 * 算子属性：
 *   - axis: -1（最后一维）
 *   - descending: false（升序排序）
 *   - stable: true（稳定排序）
 * 
 * 排序模式：SMALL_SIZE_OPTIM_MODE（归并排序）
 *   - 触发条件：sortAxisNum=512 <= 512 && dataType=BF16
 *   - tilingKey = 4002 + 10000 = 14002
 *   - BF16归并排序会在内部转换为FP32进行排序
 * 
 */
TEST_F(SortWithIndexTilingTest, test_tiling_bf16_small_size_merge_sort) {
    optiling::SortWithIndexCompileInfo compileInfo;
    compileInfo.core_num = 64;

    gert::TilingContextPara tilingContextPara("SortWithIndex",
        {
            {{{2, 512}, {2, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 512}, {2, 512}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)},
            {"descending", Ops::Math::AnyValue::CreateFrom<bool>(false)},
            {"stable", Ops::Math::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 14002;
    string expectTilingData = "4294967296 1 2 4294967297 512 17179869184 4294967808 512 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

/**
 * 测试用例：INT32小规模单块基数排序
 * 
 * 测试目的：验证INT32数据类型在小规模排序轴时触发SMALL_SIZE_MODE基数排序模式
 *           INT类型不支持归并排序优化，始终使用基数排序
 * 
 * 数据类型：DT_INT32 (tilingKey基数=1003)
 * 输出索引类型：DT_INT32
 * 
 * 输入形状：[2, 2] - 2行数据，每行仅2个元素需要排序（极小规模）
 *            unsortedDimNum=2，sortAxisNum=2
 * 
 * 算子属性：
 *   - axis: -1（最后一维）
 *   - descending: true（降序排序）
 *   - stable: true（稳定排序）
 * 
 * 排序模式：SMALL_SIZE_MODE（小规模基数排序）
 *   - 触发条件：sortAxisNum <= tileData（4096 for INT32）
 *   - 注意：INT32不满足归并排序条件（dataType不在optDataTypeBitMap中）
 *   - tilingKey = 1003（不加MERGE_SORT_TILING_OFFSET）
 * 
 */
TEST_F(SortWithIndexTilingTest, test_tiling_int32_small_size_single_block) {
    optiling::SortWithIndexCompileInfo compileInfo;
    compileInfo.core_num = 64;

    gert::TilingContextPara tilingContextPara("SortWithIndex",
        {
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)},
            {"descending", Ops::Math::AnyValue::CreateFrom<bool>(true)},
            {"stable", Ops::Math::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 1003;
    string expectTilingData = "4294967297 2 2 4294967297 2 4294967296 4294967298 2 0 0 0 4294967296 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

/**
 * 测试用例：INT64小规模单块基数排序
 * 
 * 测试目的：验证INT64数据类型在小规模排序轴时触发SMALL_SIZE_MODE基数排序模式
 *           并验证INT64数据类型的字节大小为8，tileData调整为2048
 * 
 * 数据类型：DT_INT64 (tilingKey基数=1004，字节大小=8)
 * 输出索引类型：DT_INT64
 * 
 * 
 * 算子属性：
 *   - axis: -1（最后一维）
 *   - descending: false（升序排序）
 *   - stable: true（稳定排序）
 * 
 * 排序模式：SMALL_SIZE_MODE（小规模基数排序）
 *   - 触发条件：sortAxisNum=2 <= tileData=2048（INT64专用tileData）
 *   - tilingKey = 1004
 * 
 */
TEST_F(SortWithIndexTilingTest, test_tiling_int64_small_size_single_block) {
    optiling::SortWithIndexCompileInfo compileInfo;
    compileInfo.core_num = 64;

    gert::TilingContextPara tilingContextPara("SortWithIndex",
        {
            {{{2, 2}, {2, 2}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{2, 2}, {2, 2}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)},
            {"descending", Ops::Math::AnyValue::CreateFrom<bool>(false)},
            {"stable", Ops::Math::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 1004;
    string expectTilingData = "4294967296 2 2 4294967297 2 4294967296 4294967298 2 0 0 0 4294967296 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

/**
 * 测试用例：UINT32小规模单块基数排序
 * 
 * 测试目的：验证UINT32数据类型在小规模排序轴时触发SMALL_SIZE_MODE基数排序模式
 *           UINT类型与INT类型一样不支持归并排序优化
 * 
 * 数据类型：DT_UINT32 (tilingKey基数=2003)
 * 输出索引类型：DT_UINT32
 * 
 * 
 * 排序模式：SMALL_SIZE_MODE（小规模基数排序）
 *   - 触发条件：sortAxisNum <= tileData=4096
 *   - tilingKey = 2003（UINT类型使用2000系列基数）
 * 
 */
TEST_F(SortWithIndexTilingTest, test_tiling_uint32_small_size_single_block) {
    optiling::SortWithIndexCompileInfo compileInfo;
    compileInfo.core_num = 64;

    gert::TilingContextPara tilingContextPara("SortWithIndex",
        {
            {{{2, 2}, {2, 2}}, ge::DT_UINT32, ge::FORMAT_ND},
            {{{2, 2}, {2, 2}}, ge::DT_UINT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_UINT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT32, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)},
            {"descending", Ops::Math::AnyValue::CreateFrom<bool>(false)},
            {"stable", Ops::Math::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 2003;
    string expectTilingData = "4294967296 2 2 4294967297 2 4294967296 4294967298 2 0 0 0 4294967296 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

/**
 * 测试用例：INT64大规模多块基数排序
 * 
 * 测试目的：验证INT64数据类型在大规模排序轴时触发MULT_CORE_MODE多核基数排序模式
 *           验证排序轴长度超过tileData时的多块切分策略
 * 
 * 数据类型：DT_INT64 (tilingKey基数=1004)
 * 输出索引类型：DT_INT64
 * 
 * 输入形状：[1, 8192] - 1行数据，8192个元素需要排序（大规模）
 *            unsortedDimNum=1，sortAxisNum=8192
 * 
 * 算子属性：
 *   - axis: -1（最后一维）
 *   - descending: true（降序排序）
 *   - stable: true（稳定排序）
 * 
 * 排序模式：MULT_CORE_MODE（多核基数排序）
 *   - 触发条件：sortAxisNum=8192 > tileData=2048（INT64）
 *   - 排序轴切分为多个块，每个块由一个核处理
 *   - tilingKey = 1004
 * 
 */
TEST_F(SortWithIndexTilingTest, test_tiling_int64_large_size_radix_multi_block) {
    optiling::SortWithIndexCompileInfo compileInfo;
    compileInfo.core_num = 64;

    gert::TilingContextPara tilingContextPara("SortWithIndex",
        {
            {{{1, 8192}, {1, 8192}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1, 8192}, {1, 8192}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)},
            {"descending", Ops::Math::AnyValue::CreateFrom<bool>(true)},
            {"stable", Ops::Math::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 1004;
    string expectTilingData = "4294967297 1 1 34359738376 1024 4294967296 4294975488 8192 34359738376 4294969344 8796093022464 8590137312 ";
    std::vector<size_t> expectWorkspaces = {16998400};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}