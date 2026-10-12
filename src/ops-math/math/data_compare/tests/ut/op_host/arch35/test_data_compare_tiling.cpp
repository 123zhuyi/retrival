/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * \file test_data_compare_tiling.cpp
 * \brief DataCompare Tiling UT
 *
 * 测试覆盖：
 *   1. Normal 模板路由（小 tensor，rOuter=1）
 *   2. EMPTY 模板路由（空 tensor，totalElements=0）
 *   3. Group 模板路由（大 tensor，rOuter≥2）
 *   4. 各 dtype Tiling 正确性
 *   5. UB 切分结果验证
 *   6. 不支持的 dtype 报错
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_kernel/arch35/data_compare_tiling_data.h"

namespace DataCompareUT {
using namespace std;
using namespace ge;
using namespace gert;

static const std::string OP_NAME = "DataCompare";

struct DataCompareCompileInfo {
} compileInfo;

struct DataCompareTilingParam {
    std::string caseName;
    std::initializer_list<int64_t> x1Shape;
    ge::DataType dtype;
    float atol;
    float rtol;
    ge::graphStatus expectStatus;
    uint64_t coreNum;
    uint64_t ubSize;
};

class DataCompareTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "DataCompareTilingTest SetUp." << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "DataCompareTilingTest TearDown." << std::endl;
    }
};

static void RunTilingTest(const DataCompareTilingParam& param)
{
    std::cout << "[TEST_CASE] " << param.caseName << std::endl;

    gert::StorageShape x1Shape = {param.x1Shape, param.x1Shape};
    gert::StorageShape x2Shape = {param.x1Shape, param.x1Shape};
    gert::StorageShape numShape = {{}, {}};  // 标量输出

    std::vector<gert::TilingContextPara::TensorDescription> inputTensors = {
        {x1Shape, param.dtype, ge::FORMAT_ND},
        {x2Shape, param.dtype, ge::FORMAT_ND}
    };
    std::vector<gert::TilingContextPara::TensorDescription> outputTensors = {
        {numShape, ge::DT_FLOAT, ge::FORMAT_ND}
    };

    // Attrs: atol(float, idx=0), rtol(float, idx=1) — 按 REG_OP 定义顺序
    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"atol", Ops::Math::AnyValue::CreateFrom<float>(param.atol)},
        {"rtol", Ops::Math::AnyValue::CreateFrom<float>(param.rtol)}
    };

    gert::TilingContextPara tilingContextPara(
        OP_NAME,
        inputTensors,
        outputTensors,
        attrs,
        &compileInfo,
        param.coreNum,
        param.ubSize,
        4096);

    // 使用 ExecuteTiling 验证成功/失败状态
    TilingInfo tilingInfo;
    bool result = ExecuteTiling(tilingContextPara, tilingInfo);

    if (param.expectStatus == ge::GRAPH_SUCCESS) {
        EXPECT_TRUE(result) << "Tiling should succeed for case: " << param.caseName;
        if (result) {
            // 验证 TilingData 基本字段
            const DataCompareTilingData* td =
                reinterpret_cast<const DataCompareTilingData*>(tilingInfo.tilingData.get());
            if (td != nullptr) {
                // 判断是否为空 tensor（EMPTY 模板）
                bool isEmpty = false;
                for (auto d : param.x1Shape) {
                    if (d == 0) { isEmpty = true; break; }
                }

                if (isEmpty) {
                    // EMPTY 模板：usedCoreNum=0, axisNum=0（短路路径）
                    EXPECT_EQ(td->usedCoreNum, 0) << "EMPTY: usedCoreNum should be 0";
                } else {
                    // Normal/Group 模板：axisNum=2, usedCoreNum>=1
                    EXPECT_EQ(td->axisNum, 2) << "axisNum should be 2 (AR pattern)";
                    EXPECT_GE(td->usedCoreNum, 1) << "usedCoreNum should be >= 1";
                }
                EXPECT_FLOAT_EQ(td->atol, param.atol);
                EXPECT_FLOAT_EQ(td->rtol, param.rtol);

                std::cout << "  TilingData: axisNum=" << td->axisNum
                          << ", axisShape=[" << td->axisShape[0] << "," << td->axisShape[1] << "]"
                          << ", rUbFactor=" << td->rUbFactor
                          << ", rLoopCntTotal=" << td->rLoopCntTotal
                          << ", usedCoreNum=" << td->usedCoreNum
                          << ", blockNum=" << tilingInfo.blockNum
                          << std::endl;
            }
        }
    } else {
        EXPECT_FALSE(result) << "Tiling should fail for case: " << param.caseName;
    }
}

// ─── Normal 模板（小 tensor，rOuter=1）───

TEST_F(DataCompareTilingTest, normal_fp32_small)
{
    RunTilingTest({"normal_fp32_small", {2, 3}, ge::DT_FLOAT, 1e-5f, 1e-3f,
                   ge::GRAPH_SUCCESS, 32, 248 * 1024});
}

TEST_F(DataCompareTilingTest, normal_fp16_small)
{
    RunTilingTest({"normal_fp16_small", {4, 5}, ge::DT_FLOAT16, 1e-5f, 1e-3f,
                   ge::GRAPH_SUCCESS, 32, 248 * 1024});
}

TEST_F(DataCompareTilingTest, normal_bf16_small)
{
    RunTilingTest({"normal_bf16_small", {10}, ge::DT_BF16, 1e-5f, 1e-3f,
                   ge::GRAPH_SUCCESS, 32, 248 * 1024});
}

TEST_F(DataCompareTilingTest, normal_int8_small)
{
    RunTilingTest({"normal_int8_small", {8, 8}, ge::DT_INT8, 0.0f, 0.0f,
                   ge::GRAPH_SUCCESS, 32, 248 * 1024});
}

TEST_F(DataCompareTilingTest, normal_uint8_small)
{
    RunTilingTest({"normal_uint8_small", {16}, ge::DT_UINT8, 1e-5f, 1e-3f,
                   ge::GRAPH_SUCCESS, 32, 248 * 1024});
}

TEST_F(DataCompareTilingTest, normal_int32_small)
{
    RunTilingTest({"normal_int32_small", {100}, ge::DT_INT32, 1e-5f, 1e-3f,
                   ge::GRAPH_SUCCESS, 32, 248 * 1024});
}

// ─── 标量输入（rank=0）───

TEST_F(DataCompareTilingTest, normal_scalar_rank0)
{
    RunTilingTest({"normal_scalar_rank0", {}, ge::DT_FLOAT, 1e-5f, 1e-3f,
                   ge::GRAPH_SUCCESS, 32, 248 * 1024});
}

// ─── 单元素 ───

TEST_F(DataCompareTilingTest, normal_single_element)
{
    RunTilingTest({"normal_single_element", {1}, ge::DT_FLOAT, 1e-5f, 1e-3f,
                   ge::GRAPH_SUCCESS, 32, 248 * 1024});
}

// ─── EMPTY 模板（空 tensor）───

TEST_F(DataCompareTilingTest, empty_tensor_0dim)
{
    RunTilingTest({"empty_tensor_0dim", {0, 4}, ge::DT_FLOAT, 1e-5f, 1e-3f,
                   ge::GRAPH_SUCCESS, 32, 248 * 1024});
}

TEST_F(DataCompareTilingTest, empty_tensor_single_0dim)
{
    RunTilingTest({"empty_tensor_single_0dim", {0}, ge::DT_FLOAT16, 1e-5f, 1e-3f,
                   ge::GRAPH_SUCCESS, 32, 248 * 1024});
}

// ─── Group 模板（大 tensor，rOuter≥2）───

TEST_F(DataCompareTilingTest, group_fp32_large)
{
    // 大 tensor 触发 Group 模板
    RunTilingTest({"group_fp32_large", {1024, 1024}, ge::DT_FLOAT, 1e-5f, 1e-3f,
                   ge::GRAPH_SUCCESS, 32, 248 * 1024});
}

TEST_F(DataCompareTilingTest, group_fp16_large)
{
    RunTilingTest({"group_fp16_large", {2048, 512}, ge::DT_FLOAT16, 1e-5f, 1e-3f,
                   ge::GRAPH_SUCCESS, 32, 248 * 1024});
}

TEST_F(DataCompareTilingTest, group_int8_large)
{
    RunTilingTest({"group_int8_large", {4096, 256}, ge::DT_INT8, 0.0f, 0.0f,
                   ge::GRAPH_SUCCESS, 32, 248 * 1024});
}

// ─── atol=0, rtol=0 精确比较 ───

TEST_F(DataCompareTilingTest, normal_exact_compare)
{
    RunTilingTest({"normal_exact_compare", {32, 32}, ge::DT_FLOAT, 0.0f, 0.0f,
                   ge::GRAPH_SUCCESS, 32, 248 * 1024});
}

// ─── 大 atol/rtol ───

TEST_F(DataCompareTilingTest, normal_large_tolerance)
{
    RunTilingTest({"normal_large_tolerance", {64}, ge::DT_FLOAT, 1.0f, 0.5f,
                   ge::GRAPH_SUCCESS, 32, 248 * 1024});
}

// ─── 不支持的 dtype ───

TEST_F(DataCompareTilingTest, failed_unsupported_dtype)
{
    RunTilingTest({"failed_unsupported_dtype", {4, 4}, ge::DT_DOUBLE, 1e-5f, 1e-3f,
                   ge::GRAPH_FAILED, 32, 248 * 1024});
}

// ─── 高 rank ───

TEST_F(DataCompareTilingTest, normal_high_rank)
{
    RunTilingTest({"normal_high_rank", {2, 2, 2, 2}, ge::DT_FLOAT, 1e-5f, 1e-3f,
                   ge::GRAPH_SUCCESS, 32, 248 * 1024});
}

// ─── UB 切分详细验证 ───

static void RunDetailedTilingTest(const DataCompareTilingParam& param,
                                    int64_t expectedTotalElements)
{
    std::cout << "[TEST_CASE_DETAILED] " << param.caseName << std::endl;

    gert::StorageShape x1Shape = {param.x1Shape, param.x1Shape};
    gert::StorageShape x2Shape = {param.x1Shape, param.x1Shape};
    gert::StorageShape numShape = {{}, {}};

    std::vector<gert::TilingContextPara::TensorDescription> inputTensors = {
        {x1Shape, param.dtype, ge::FORMAT_ND},
        {x2Shape, param.dtype, ge::FORMAT_ND}
    };
    std::vector<gert::TilingContextPara::TensorDescription> outputTensors = {
        {numShape, ge::DT_FLOAT, ge::FORMAT_ND}
    };

    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"atol", Ops::Math::AnyValue::CreateFrom<float>(param.atol)},
        {"rtol", Ops::Math::AnyValue::CreateFrom<float>(param.rtol)}
    };

    gert::TilingContextPara tilingContextPara(
        OP_NAME, inputTensors, outputTensors, attrs,
        &compileInfo, param.coreNum, param.ubSize, 4096);

    TilingInfo tilingInfo;
    bool result = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(result) << "Tiling should succeed for case: " << param.caseName;

    const DataCompareTilingData* td =
        reinterpret_cast<const DataCompareTilingData*>(tilingInfo.tilingData.get());
    ASSERT_NE(td, nullptr);

    // 空 tensor 特殊路径
    if (expectedTotalElements == 0) {
        EXPECT_EQ(td->usedCoreNum, 0);
        EXPECT_EQ(td->cacheBufUbSize, 16 * 1024);
        return;
    }

    // ─── AR pattern 基本字段 ───
    EXPECT_EQ(td->axisNum, 2);
    EXPECT_EQ(td->axisShape[0], 1) << "A axis should be 1 (All Reduce)";
    EXPECT_EQ(td->axisShape[1], expectedTotalElements) << "R axis should be totalElements";
    EXPECT_EQ(td->axisStride[0], expectedTotalElements);
    EXPECT_EQ(td->axisStride[1], 1);

    // ─── UB 切分字段 ───
    EXPECT_EQ(td->aSplitAxisIdx, 0);
    EXPECT_EQ(td->rSplitAxisIdx, 1);
    EXPECT_EQ(td->aUbFactor, 1) << "All Reduce: aUbFactor=1";
    EXPECT_EQ(td->aUbFactorAlign, 1);
    EXPECT_EQ(td->innerAProd, 1);
    EXPECT_EQ(td->innerAProdAlign, 1);
    EXPECT_EQ(td->innerRProd, 1);
    EXPECT_EQ(td->innerRProdAlign, 1);

    // ─── rUbFactor 约束 ───
    EXPECT_GT(td->rUbFactor, 0) << "rUbFactor should be > 0";
    EXPECT_LE(td->rUbFactor, expectedTotalElements) << "rUbFactor <= totalElements";
    EXPECT_GE(td->rUbFactorAlign, td->rUbFactor) << "rUbFactorAlign >= rUbFactor";

    // ─── rLoopCntTotal ───
    int64_t expectedRLC = (expectedTotalElements + td->rUbFactor - 1) / td->rUbFactor;
    EXPECT_EQ(td->rLoopCntTotal, expectedRLC) << "rLoopCntTotal = CeilDiv(total, rUbFactor)";

    // ─── 多核切分 ───
    EXPECT_EQ(td->aLoopCntTotal, 1) << "All Reduce: aLoopCntTotal=1";
    EXPECT_EQ(td->aSplitChunkCnt, 1);

    // ─── Buffer 大小 ───
    EXPECT_GT(td->preReduceUbSize, 0);
    EXPECT_GT(td->postReduceUbSize, 0);
    EXPECT_GT(td->tmpBufUbSize, 0);
    EXPECT_EQ(td->cacheBufUbSize, 16 * 1024);

    // ─── attrs 透传 ───
    EXPECT_FLOAT_EQ(td->atol, param.atol);
    EXPECT_FLOAT_EQ(td->rtol, param.rtol);

    // ─── Group 判定 ───
    int64_t aOuter = td->aLoopCntTotal;
    int64_t rOuter = td->rLoopCntTotal;
    bool expectGroup = (aOuter <= static_cast<int64_t>(param.coreNum) / 2) && (rOuter > 1);
    if (expectGroup) {
        EXPECT_GT(td->rGroupCnt, 0) << "Group: rGroupCnt should be > 0";
        EXPECT_GT(td->usedCoreNum, 1) << "Group: usedCoreNum should be > 1";
    } else {
        EXPECT_EQ(td->usedCoreNum, 1) << "Normal: usedCoreNum should be 1";
    }

    std::cout << "  Detailed: rUbFactor=" << td->rUbFactor
              << ", rUbFactorAlign=" << td->rUbFactorAlign
              << ", rLoopCntTotal=" << td->rLoopCntTotal
              << ", usedCoreNum=" << td->usedCoreNum
              << ", rGroupCnt=" << td->rGroupCnt
              << ", preReduceUbSize=" << td->preReduceUbSize
              << ", tmpBufUbSize=" << td->tmpBufUbSize
              << std::endl;
}

TEST_F(DataCompareTilingTest, detailed_fp32_normal_ub_split)
{
    // 6 elements, fp32: rUbFactor 应能全载（6 < r_i_max），rLoopCntTotal=1
    RunDetailedTilingTest({"detailed_fp32_normal", {2, 3}, ge::DT_FLOAT, 1e-5f, 1e-3f,
                           ge::GRAPH_SUCCESS, 32, 248 * 1024}, 6);
}

TEST_F(DataCompareTilingTest, detailed_fp32_group_trigger)
{
    // 1M elements, fp32: rLoopCntTotal > 1 → 触发 Group
    RunDetailedTilingTest({"detailed_fp32_group", {1024, 1024}, ge::DT_FLOAT, 1e-5f, 1e-3f,
                           ge::GRAPH_SUCCESS, 32, 248 * 1024}, 1048576);
}

TEST_F(DataCompareTilingTest, detailed_int8_large_group)
{
    // 1M elements, int8: typeSize=1, rUbFactor 更大
    RunDetailedTilingTest({"detailed_int8_group", {4096, 256}, ge::DT_INT8, 0.0f, 0.0f,
                           ge::GRAPH_SUCCESS, 32, 248 * 1024}, 1048576);
}

TEST_F(DataCompareTilingTest, detailed_empty_tensor)
{
    RunDetailedTilingTest({"detailed_empty", {0, 4}, ge::DT_FLOAT, 1e-5f, 1e-3f,
                           ge::GRAPH_SUCCESS, 32, 248 * 1024}, 0);
}

TEST_F(DataCompareTilingTest, detailed_scalar_rank0)
{
    RunDetailedTilingTest({"detailed_scalar", {}, ge::DT_FLOAT, 1e-5f, 1e-3f,
                           ge::GRAPH_SUCCESS, 32, 248 * 1024}, 1);
}

// ─── Group 触发边界条件 ───

TEST_F(DataCompareTilingTest, group_boundary_rOuter_eq_2)
{
    // 构造恰好使 rLoopCntTotal=2 的 shape
    // fp32, coreNum=32: r_i_max ≈ (248K-16K-4)/(3*4+8) ≈ 11872
    // 需要 totalElements > 11872 → rLoopCntTotal=2 → total=23744
    RunTilingTest({"group_boundary_r2", {23744}, ge::DT_FLOAT, 1e-5f, 1e-3f,
                   ge::GRAPH_SUCCESS, 32, 248 * 1024});
}

TEST_F(DataCompareTilingTest, normal_boundary_rOuter_eq_1)
{
    // 构造恰好使 rLoopCntTotal=1 的 shape（全载）
    // fp32: r_i_max ≈ 11872 → totalElements=11872 → rLoopCntTotal=1 → Normal
    RunTilingTest({"normal_boundary_r1", {11872}, ge::DT_FLOAT, 1e-5f, 1e-3f,
                   ge::GRAPH_SUCCESS, 32, 248 * 1024});
}

// ─── 不同 coreNum 下的 Group 行为 ───

TEST_F(DataCompareTilingTest, group_small_core_num)
{
    // coreNum=4, 大 tensor: aOuter=1 ≤ 4/2=2, rOuter>1 → Group
    RunTilingTest({"group_small_core", {100000}, ge::DT_FLOAT, 1e-5f, 1e-3f,
                   ge::GRAPH_SUCCESS, 4, 248 * 1024});
}

TEST_F(DataCompareTilingTest, normal_fp16_medium)
{
    // fp16, 中等 size: 可能全载走 Normal
    RunTilingTest({"normal_fp16_medium", {8192}, ge::DT_FLOAT16, 1e-5f, 1e-3f,
                   ge::GRAPH_SUCCESS, 32, 248 * 1024});
}

// ─── EMPTY 模板各 dtype ───

TEST_F(DataCompareTilingTest, empty_int8)
{
    RunTilingTest({"empty_int8", {0, 8}, ge::DT_INT8, 0.0f, 0.0f,
                   ge::GRAPH_SUCCESS, 32, 248 * 1024});
}

TEST_F(DataCompareTilingTest, empty_bf16)
{
    RunTilingTest({"empty_bf16", {0}, ge::DT_BF16, 1e-5f, 1e-3f,
                   ge::GRAPH_SUCCESS, 32, 248 * 1024});
}

TEST_F(DataCompareTilingTest, empty_int32)
{
    RunTilingTest({"empty_int32", {4, 0, 8}, ge::DT_INT32, 1e-5f, 1e-3f,
                   ge::GRAPH_SUCCESS, 32, 248 * 1024});
}

// ─── 最大 rank=8 ───

TEST_F(DataCompareTilingTest, normal_max_rank_8d)
{
    RunTilingTest({"normal_max_rank8", {2, 2, 2, 2, 2, 2, 2, 2}, ge::DT_FLOAT, 1e-5f, 1e-3f,
                   ge::GRAPH_SUCCESS, 32, 248 * 1024});
}

}  // namespace DataCompareUT
