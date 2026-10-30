/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <vector>
#include "log/log.h"
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/top_k_v2_tiling_arch35.h"

using namespace std;
using namespace ge;

class TopKV2Tiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AscendTopKV2Test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "AscendTopKV2Test TearDown" << std::endl; }
};

/**
 * @brief 测试TopK V2算子的小规模归并排序模式
 *
 * 测试场景：
 * - 输入shape: [10, 32]，最后轴长度为32（小规模数据）
 * - K值: 8（取前8个最大值）
 * - 数据类型: FLOAT32
 * - 排序要求: sorted=true，需要输出排序结果
 * - 输出索引类型: INT64
 *
 * 测试目的：
 * - 验证小规模数据（lastAxisNum <= 1024）场景下，算子选择归并排序模式
 * - 验证TilingKey为13003（小规模归并排序模式标识）
 * - 验证UB空间分配和tile切分策略的正确性
 * - 验证workspace大小计算正确
 */
TEST_F(TopKV2Tiling, test_tiling_small_merge_sort_mode)
{
    optiling::TopKV2CompileInfo compileInfo = {64};
    vector<int64_t> k = {8};

    gert::TilingContextPara tilingContextPara(
        "TopKV2",
        {
            {{{10, 32}, {10, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND, true, k.data()},
        },
        {
            {{{10, 8}, {10, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{10, 8}, {10, 8}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("sorted", Ops::Math::AnyValue::CreateFrom<bool>(true)),
            gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)),
            gert::TilingContextPara::OpAttr("largest", Ops::Math::AnyValue::CreateFrom<bool>(true)),
            gert::TilingContextPara::OpAttr("indices_dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(9)),
        },
        &compileInfo);
    uint64_t expectTilingKey = 13003;
    string expectTilingData = "4294967297 1 10 10 137438953473 64 0 1 1 0 4294967296 32 8 1 8 1 0 0 0 0 0 0 0 0 0 0 0 "
                              "8589934592 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16787584};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

/**
 * @brief 测试TopK V2算子的单块模式（Single Block Mode）
 *
 * 测试场景：
 * - 输入shape: [10, 3234]，最后轴长度为3234
 * - K值: 258（取前258个最大值）
 * - 数据类型: FLOAT32
 * - 排序要求: sorted=true
 * - 输出索引类型: INT64
 *
 * 测试目的：
 * - 验证当lastAxisNum能够一次性装入UB时（tileSize >= lastAxisNum），选择单块模式
 * - 单块模式下，一个核可以处理多个batch（lastAxisNum），充分利用UB空间
 * - 验证TilingKey为3003（单块模式标识）
 * - 验证workspace大小为默认值+额外数据空间
 */
TEST_F(TopKV2Tiling, test_tiling_top_k_single_block_mode)
{
    optiling::TopKV2CompileInfo compileInfo = {64};
    vector<int64_t> k = {258};

    gert::TilingContextPara tilingContextPara(
        "TopKV2",
        {
            {{{10, 3234}, {10, 3234}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND, true, k.data()},
        },
        {
            {{{10, 258}, {10, 258}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{10, 258}, {10, 258}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("sorted", Ops::Math::AnyValue::CreateFrom<bool>(true)),
            gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)),
            gert::TilingContextPara::OpAttr("largest", Ops::Math::AnyValue::CreateFrom<bool>(true)),
            gert::TilingContextPara::OpAttr("indices_dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(9)),
        },
        &compileInfo);
    uint64_t expectTilingKey = 3003;
    /**
     * workspace = 16777216(默认) + 除了除了输入值、输出值、输出索引的worksape空间大小
     */
    string expectTilingData = "4294967297 1 64 10 13889924235265 64 0 1 4 10 4294967296 3234 258 1 258 1 0 0 0 0 0 0 0 "
                              "0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16843264};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

/**
 * @brief 测试TopK V2算子的FP32多核归并排序模式
 *
 * 测试场景：
 * - 输入shape: [1, 22340]，最后轴长度为22340（大规模数据）
 * - K值: 5258（取前5258个最大值）
 * - 数据类型: FLOAT32
 * - 排序要求: sorted=true，需要输出排序结果
 * - 输出索引类型: INT64
 * - 非尾轴维度: 1（单batch场景）
 *
 * 测试目的：
 * - 验证大规模数据场景下，选择FP32多核归并排序模式
 * - 验证splitCoreNum计算：ceil(22340/2048) = 11核
 * - 验证TilingKey为23003（FP32多核归并排序模式标识）
 * - 验证workspace大小包含输入数据、输出数据和索引的空间
 * - 验证核心参数：onceMaxElementsAlign、lastDimTileNumTimes等
 */
TEST_F(TopKV2Tiling, test_tiling_top_k_fp32_more_core_mode)
{
    optiling::TopKV2CompileInfo compileInfo = {64};
    vector<int64_t> k = {6258};

    gert::TilingContextPara tilingContextPara(
        "TopKV2",
        {
            {{{1, 22340}, {1, 22340}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND, true, k.data()},
        },
        {
            {{{1, 6258}, {1, 6258}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 6258}, {1, 6258}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("sorted", Ops::Math::AnyValue::CreateFrom<bool>(true)),
            gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)),
            gert::TilingContextPara::OpAttr("largest", Ops::Math::AnyValue::CreateFrom<bool>(true)),
            gert::TilingContextPara::OpAttr("indices_dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(9)),
        },
        &compileInfo);
    uint64_t expectTilingKey = 23003;
    string expectTilingData = "4294967297 1 1 1 8718783610891 64 0 1 0 0 4294967302 22340 6258 11 6258 1 0 0 0 0 0 0 0 "
                              "0 0 1792 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {17134656};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

/**
 * @brief 测试TopK V2算子的FP32核内归并排序模式
 *
 * 测试场景：
 * - 输入shape: [35, 52340]，最后轴长度为52340（超大规模数据）
 * - K值: 21321（取前21321个最大值）
 * - 数据类型: FLOAT32
 * - 排序要求: sorted=true，需要输出排序结果
 * - 输出索引类型: INT64
 * - 非尾轴维度: 35（多batch场景）
 *
 * 测试目的：
 * - 验证超大规模数据场景下，选择FP32核内归并排序模式
 * - 该模式适用于：splitCoreNum > 1 且 非尾轴数量 * splitCoreNum <= maxCoreNum 且 K > 0
 * - 验证splitCoreNum计算：ceil(52340/2048) ≈ 26核，splitCoreNum > 1
 * - 验证核内排序策略：单核内部进行多块数据的归并排序
 * - 验证TilingKey为33003（FP32核内归并排序模式标识）
 * - 验证workspace大小为46882816（包含大量输入数据的workspace空间）
 * - 验证核心参数：blocksPerRow=15、blockSortSize、extractChunkSize等核内排序参数
 * - 核内归并排序通过block级别排序后进行核内归并，充分利用单核资源
 */
TEST_F(TopKV2Tiling, test_tiling_top_k_fp32_intra_core_mode)
{
    optiling::TopKV2CompileInfo compileInfo = {64};
    vector<int64_t> k = {21321};

    gert::TilingContextPara tilingContextPara(
        "TopKV2",
        {
            {{{35, 52340}, {35, 52340}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND, true, k.data()},
        },
        {
            {{{35, 21321}, {35, 21321}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{35, 21321}, {35, 21321}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("sorted", Ops::Math::AnyValue::CreateFrom<bool>(true)),
            gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)),
            gert::TilingContextPara::OpAttr("largest", Ops::Math::AnyValue::CreateFrom<bool>(true)),
            gert::TilingContextPara::OpAttr("indices_dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(9)),
        },
        &compileInfo);
    uint64_t expectTilingKey = 33003;
    string expectTilingData = "4294967297 1 35 35 15393162788899 64 0 1 0 0 4294967303 52340 21321 15 21321 1 0 0 0 0 "
                              "0 0 0 0 0 1 461794883673088 20478404121088 599186 0 0 ";
    std::vector<size_t> expectWorkspaces = {46882816};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

/**
 * @brief 测试TopK V2算子的基数排序多核优化模式
 *
 * 测试场景：
 * - 输入shape: [10, 22340]，最后轴长度为22340
 * - K值: 2（取前2个最大值，K值较小）
 * - 数据类型: FLOAT32
 * - 排序要求: sorted=true
 * - 输出索引类型: INT64
 * - 非尾轴维度: 10（多batch场景）
 *
 * 测试目的：
 * - 验证多核优化模式的选择条件：K * (lastAxisNum / nowTileSize) <= nowTileSize
 * - 该模式适用场景：假设lastAxisNum需要N个核处理，每个核计算出Topk的值后，
 *   将前面每个核的Topk值集中到一个核（前提是这个核的UB能装得下），再进行一次Topk处理
 * - 验证优化策略：减少核间通信，提高小K值场景的处理效率
 * - 验证TilingKey为3003（多核基数排序优化模式标识，与中等模式共用Key但modeType不同）
 * - 验证核心参数：lastDimTileNum=3，tileNum=3，表示数据被切分为3个tile块
 * - 验证workspace大小为16777728，相对较小因为K值小
 * - 多核优化模式在小K值场景下性能优于普通多核模式，避免过多核间数据搬运
 */
TEST_F(TopKV2Tiling, test_tiling_top_k_radix_multi_core_optim_mode)
{
    optiling::TopKV2CompileInfo compileInfo = {64};
    vector<int64_t> k = {2};

    gert::TilingContextPara tilingContextPara(
        "TopKV2",
        {
            {{{10, 22340}, {10, 22340}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND, true, k.data()},
        },
        {
            {{{10, 2}, {10, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{10, 2}, {10, 2}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("sorted", Ops::Math::AnyValue::CreateFrom<bool>(true)),
            gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)),
            gert::TilingContextPara::OpAttr("largest", Ops::Math::AnyValue::CreateFrom<bool>(true)),
            gert::TilingContextPara::OpAttr("indices_dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(9)),
        },
        &compileInfo);
    uint64_t expectTilingKey = 3003;
    string expectTilingData = "4294967297 1 10 10 32985348833283 64 0 1 1 0 4294967300 22340 2 3 2 1 0 0 0 0 0 0 0 0 0 "
                              "0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777728};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

/**
 * @brief 测试TopK V2算子的基数排序多核中等规模模式
 *
 * 测试场景：
 * - 输入shape: [10, 22340]，最后轴长度为22340，非尾轴维度为10
 * - K值: 1258（取前1258个最大值）
 * - 数据类型: FLOAT32
 * - 排序要求: sorted=true
 * - 输出索引类型: INT64
 *
 * 测试目的：
 * - 验证中等规模数据场景，lastAxisNum <= sortedDimParallelData时选择多核中等模式
 * - sortedDimParallelData = (tileSize * maxCoreNum) / 2
 * - 验证多核切分策略：lastDimTileNum和unsortedDimParallel的计算
 * - 验证TilingKey为3003（多核基数排序中等规模模式标识）
 * - 验证workspace大小和tiling数据的正确性
 * - 验证核心参数：tileNum=3，表示需要3个tile块处理数据
 */
TEST_F(TopKV2Tiling, test_tiling_top_k_radix_multi_core_medium_mode)
{
    optiling::TopKV2CompileInfo compileInfo = {64};
    vector<int64_t> k = {1258};

    gert::TilingContextPara tilingContextPara(
        "TopKV2",
        {
            {{{10, 22340}, {10, 22340}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND, true, k.data()},
        },
        {
            {{{10, 1258}, {10, 1258}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{10, 1258}, {10, 1258}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("sorted", Ops::Math::AnyValue::CreateFrom<bool>(true)),
            gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)),
            gert::TilingContextPara::OpAttr("largest", Ops::Math::AnyValue::CreateFrom<bool>(true)),
            gert::TilingContextPara::OpAttr("indices_dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(9)),
        },
        &compileInfo);
    uint64_t expectTilingKey = 3003;
    string expectTilingData = "4294967297 1 10 10 32985348833283 64 0 1 1 0 4294967300 22340 1258 3 1258 1 0 0 0 0 0 0 "
                              "0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {17079168};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

/**
 * @brief 测试TopK V2算子的基数排序多核大规模模式
 *
 * 测试场景：
 * - 输入shape: [10, 3222340]，最后轴长度为3222340（超大规模数据）
 * - K值: 1258（取前1258个最大值）
 * - 数据类型: FLOAT32
 * - 排序要求: sorted=true
 * - 输出索引类型: INT64
 * - 非尾轴维度: 10（多batch场景）
 *
 * 测试目的：
 * - 验证超大规模数据场景下（lastAxisNum > sortedDimParallelData），选择多核大规模模式
 * - sortedDimParallelData = (tileSize * maxCoreNum) / 2，当lastAxisNum超过此阈值时进入大规模模式
 * - 验证大规模多核切分策略：lastAxisNum非常大，需要大量核并行处理
 * - 验证lastDimTileNum计算：ceil(3222340/tileSize) ≈ 420个tile块
 * - 验证核心参数：lastDimTileNum=420，unsortedDimParallel=7，表示7个核并行处理
 * - 验证TilingKey为3003（多核基数排序大规模模式标识）
 * - 验证workspace大小为16781632，包含大量数据的workspace空间
 * - 验证modeType为多核模式（MULT_CORE_MODE），unsortedDimParallel=1表示每个核独立处理一个batch
 * - 大规模模式下，每个核处理完整的lastAxisNum数据，充分利用核并行能力
 */
TEST_F(TopKV2Tiling, test_tiling_top_k_radix_multi_core_big_mode)
{
    optiling::TopKV2CompileInfo compileInfo = {64};
    vector<int64_t> k = {1258};

    gert::TilingContextPara tilingContextPara(
        "TopKV2",
        {
            {{{10, 3222340}, {10, 3222340}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND, true, k.data()},
        },
        {
            {{{10, 1258}, {10, 1258}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{10, 1258}, {10, 1258}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("sorted", Ops::Math::AnyValue::CreateFrom<bool>(true)),
            gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)),
            gert::TilingContextPara::OpAttr("largest", Ops::Math::AnyValue::CreateFrom<bool>(true)),
            gert::TilingContextPara::OpAttr("indices_dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(9)),
        },
        &compileInfo);
    uint64_t expectTilingKey = 3003;
    string expectTilingData = "4294967297 10 1 10 32985348833344 64 0 1 1 0 4294967298 3222340 1258 420 1258 7 0 0 0 0 "
                              "0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16781632};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

/**
 * @brief 测试TopK V2算子的基数排序与TopK组合模式（SortAndTopK）
 *
 * 测试场景：
 * - 输入shape: [10, 13222340]，最后轴长度为13222340（超大规模数据，超过1000万阈值）
 * - K值: 11258（取前11258个最大值）
 * - 数据类型: FLOAT32
 * - 排序要求: sorted=true，需要输出排序结果
 * - 输出索引类型: INT64
 * - 非尾轴维度: 10（多batch场景）
 *
 * 测试目的：
 * - 验证超大规模数据场景（lastAxisNum >= 10000000），选择SortAndTopK组合模式
 * - SortAndTopK模式的阈值：SORT_AND_TOP_K_THRESHOLD = 10000000（1000万）
 * - 该模式策略：先进行基数排序对全部数据排序，然后从排序结果中取前K个
 */
TEST_F(TopKV2Tiling, test_tiling_top_k_radix_sort_and_topk)
{
    optiling::TopKV2CompileInfo compileInfo = {64};
    vector<int64_t> k = {11258};

    gert::TilingContextPara tilingContextPara(
        "TopKV2",
        {
            {{{10, 13222340}, {10, 13222340}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND, true, k.data()},
        },
        {
            {{{10, 11258}, {10, 11258}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{10, 11258}, {10, 11258}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("sorted", Ops::Math::AnyValue::CreateFrom<bool>(true)),
            gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)),
            gert::TilingContextPara::OpAttr("largest", Ops::Math::AnyValue::CreateFrom<bool>(true)),
            gert::TilingContextPara::OpAttr("indices_dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(9)),
        },
        &compileInfo);
    uint64_t expectTilingKey = 3003;
    string expectTilingData = "2 10 1 10 63771674411072 0 0 0 0 0 4294967301 13222340 11258 891 0 0 0 0 0 0 0 0 0 0 0 "
                              "64 18657337933856 137438953476 74629351749552 28668 1 ";
    std::vector<size_t> expectWorkspaces = {299818368};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
