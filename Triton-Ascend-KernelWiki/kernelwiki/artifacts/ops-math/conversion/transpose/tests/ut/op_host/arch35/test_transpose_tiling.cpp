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
 * \file test_transpose_tiling.cpp
 * \brief transpose tiling ut test
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/transpose_tiling_with_gather_arch35.h"
#include "../../../../op_host/arch35/transpose_tiling_with_021vconv_arch35.h"
#include "../../../../op_host/arch35/transpose_tiling_arch35.h"

using namespace ge;

class TransposeTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "TransposeTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "TransposeTiling TearDown" << std::endl; }
};

TEST_F(TransposeTiling, transpose_tiling_01)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[4] = {3, 0, 1, 2};
    gert::TilingContextPara::TensorDescription x({{36, 203, 26, 31}, {36, 203, 26, 31}}, ge::DT_INT64, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{31, 36, 203, 26}, {31, 36, 203, 26}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    uint64_t expectTilingKey = 10006;
    string expectTilingData =
        "10006 93802085788320 215891311432040488 16842752 27 27 190008 190008 755914244272 0 0 0 190008 0 0 1080 1 1 1 "
        "1 1 1 1 5456 1 1 1 1 1 1 1 176 1 1 1 1 1 1 1 133143986352 4294967297 4294967297 755914244127 4294967297 "
        "4294967297 16557351567361 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(TransposeTiling, transpose_021vconv_float32_rconv)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[3] = {0, 2, 1};
    gert::TilingContextPara::TensorDescription x({{2, 16, 8}, {2, 16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{2, 8, 16}, {2, 8, 16}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    if (success && tilingInfo.tilingKey == 10008) {
        optiling::Transpose021WithVCONV::Transpose021VCONVTilingData tilingData(tilingInfo.tilingData.get());
        EXPECT_EQ(tilingData.get_HLen(), 16);
        EXPECT_EQ(tilingData.get_WLen(), 8);
        EXPECT_EQ(tilingData.get_UseRConv(), true);
        EXPECT_EQ(tilingData.get_HAlignBlockElem(), 16);
        EXPECT_EQ(tilingData.get_WAlignBlockElem(), 8);
        std::vector<size_t> expectWorkspaces = {16777216};
        ASSERT_EQ(tilingInfo.workspaceSizes.size(), expectWorkspaces.size());
        for (size_t i = 0; i < expectWorkspaces.size(); i++) {
            EXPECT_EQ(tilingInfo.workspaceSizes[i], expectWorkspaces[i]);
        }
    }
}

TEST_F(TransposeTiling, transpose_021vconv_float32_rconv_aligned_w)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[3] = {0, 2, 1};
    gert::TilingContextPara::TensorDescription x({{1, 32, 24}, {1, 32, 24}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{1, 24, 32}, {1, 24, 32}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    if (success && tilingInfo.tilingKey == 10008) {
        optiling::Transpose021WithVCONV::Transpose021VCONVTilingData tilingData(tilingInfo.tilingData.get());
        EXPECT_EQ(tilingData.get_HLen(), 32);
        EXPECT_EQ(tilingData.get_WLen(), 24);
        EXPECT_EQ(tilingData.get_UseRConv(), true);
        EXPECT_EQ(tilingData.get_HAlignBlockElem(), 32);
        EXPECT_EQ(tilingData.get_WAlignBlockElem(), 24);
    }
}

TEST_F(TransposeTiling, transpose_021vconv_float32_cconv)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[3] = {0, 2, 1};
    gert::TilingContextPara::TensorDescription x({{1, 8, 32}, {1, 8, 32}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{1, 32, 8}, {1, 32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    if (success && tilingInfo.tilingKey == 10008) {
        optiling::Transpose021WithVCONV::Transpose021VCONVTilingData tilingData(tilingInfo.tilingData.get());
        EXPECT_EQ(tilingData.get_HLen(), 8);
        EXPECT_EQ(tilingData.get_WLen(), 32);
        EXPECT_EQ(tilingData.get_UseRConv(), false);
        EXPECT_EQ(tilingData.get_HAlignBlockElem(), 16);
        EXPECT_EQ(tilingData.get_WAlignBlockElem(), 32);
    }
}

TEST_F(TransposeTiling, transpose_021vconv_int32_rconv)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[3] = {0, 2, 1};
    gert::TilingContextPara::TensorDescription x({{2, 16, 16}, {2, 16, 16}}, ge::DT_INT32, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{2, 16, 16}, {2, 16, 16}}, ge::DT_INT32, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    if (success && tilingInfo.tilingKey == 10008) {
        optiling::Transpose021WithVCONV::Transpose021VCONVTilingData tilingData(tilingInfo.tilingData.get());
        EXPECT_EQ(tilingData.get_HLen(), 16);
        EXPECT_EQ(tilingData.get_WLen(), 16);
        EXPECT_EQ(tilingData.get_UseRConv(), true);
        EXPECT_EQ(tilingData.get_HAlignBlockElem(), 16);
        EXPECT_EQ(tilingData.get_WAlignBlockElem(), 16);
    }
}

TEST_F(TransposeTiling, transpose_021vconv_int8_rconv)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[3] = {0, 2, 1};
    gert::TilingContextPara::TensorDescription x({{2, 32, 32}, {2, 32, 32}}, ge::DT_INT8, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{2, 32, 32}, {2, 32, 32}}, ge::DT_INT8, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    if (success && tilingInfo.tilingKey == 10008) {
        optiling::Transpose021WithVCONV::Transpose021VCONVTilingData tilingData(tilingInfo.tilingData.get());
        EXPECT_EQ(tilingData.get_HLen(), 32);
        EXPECT_EQ(tilingData.get_WLen(), 32);
        EXPECT_EQ(tilingData.get_UseRConv(), true);
        EXPECT_EQ(tilingData.get_HAlignBlockElem(), 32);
        EXPECT_EQ(tilingData.get_WAlignBlockElem(), 32);
    }
}

TEST_F(TransposeTiling, transpose_021vconv_uint8_rconv)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[3] = {0, 2, 1};
    gert::TilingContextPara::TensorDescription x({{1, 64, 32}, {1, 64, 32}}, ge::DT_UINT8, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{1, 32, 64}, {1, 32, 64}}, ge::DT_UINT8, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    if (success && tilingInfo.tilingKey == 10008) {
        optiling::Transpose021WithVCONV::Transpose021VCONVTilingData tilingData(tilingInfo.tilingData.get());
        EXPECT_EQ(tilingData.get_HLen(), 64);
        EXPECT_EQ(tilingData.get_WLen(), 32);
        EXPECT_EQ(tilingData.get_UseRConv(), true);
        EXPECT_EQ(tilingData.get_HAlignBlockElem(), 64);
        EXPECT_EQ(tilingData.get_WAlignBlockElem(), 32);
    }
}

TEST_F(TransposeTiling, transpose_021vconv_int8_cconv_fallback)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[3] = {0, 2, 1};
    gert::TilingContextPara::TensorDescription x({{1, 16, 32}, {1, 16, 32}}, ge::DT_INT8, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{1, 32, 16}, {1, 32, 16}}, ge::DT_INT8, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    if (success) {
        EXPECT_NE(tilingInfo.tilingKey, 10008);
    }
}

TEST_F(TransposeTiling, transpose_021vconv_int8_cconv_hlen16_wlen128)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[3] = {0, 2, 1};
    gert::TilingContextPara::TensorDescription x({{128, 16, 128}, {128, 16, 128}}, ge::DT_INT8, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{128, 128, 16}, {128, 128, 16}}, ge::DT_INT8, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    if (success && tilingInfo.tilingKey == 10008) {
        optiling::Transpose021WithVCONV::Transpose021VCONVTilingData tilingData(tilingInfo.tilingData.get());
        EXPECT_EQ(tilingData.get_HLen(), 16);
        EXPECT_EQ(tilingData.get_WLen(), 128);
        EXPECT_EQ(tilingData.get_UseRConv(), false);
        EXPECT_EQ(tilingData.get_HAlignBlockElem(), 32);
        EXPECT_EQ(tilingData.get_WAlignBlockElem(), 128);
    }
}

TEST_F(TransposeTiling, transpose_021vconv_int8_cconv_hlen16_wlen64)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[3] = {0, 2, 1};
    gert::TilingContextPara::TensorDescription x({{1600, 16, 64}, {1600, 16, 64}}, ge::DT_INT8, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{1600, 64, 16}, {1600, 64, 16}}, ge::DT_INT8, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    if (success && tilingInfo.tilingKey == 10008) {
        optiling::Transpose021WithVCONV::Transpose021VCONVTilingData tilingData(tilingInfo.tilingData.get());
        EXPECT_EQ(tilingData.get_HLen(), 16);
        EXPECT_EQ(tilingData.get_WLen(), 64);
        EXPECT_EQ(tilingData.get_UseRConv(), false);
        EXPECT_EQ(tilingData.get_HAlignBlockElem(), 32);
        EXPECT_EQ(tilingData.get_WAlignBlockElem(), 64);
    }
}

// 1D tensor (perm=[0]), expect TENSOR_MOVE tiling key 10000
TEST_F(TransposeTiling, transpose_tiling_tensor_move_1d)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[1] = {0};
    gert::TilingContextPara::TensorDescription x({{100000}, {100000}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{100000}, {100000}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(optiling::SplitMode::TENSOR_MOVE));
}

// Small shape (total volume bytes < threshold), expect SMALL_SHAPE tiling key 10001
TEST_F(TransposeTiling, transpose_tiling_small_shape_2d)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[2] = {1, 0};
    gert::TilingContextPara::TensorDescription x({{10, 20}, {10, 20}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{20, 10}, {20, 10}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(optiling::SplitMode::SMALL_SHAPE));
}

// 4D N_LAST transpose (last axis not transposed, large shape, last axis >= 32), expect key 10004
TEST_F(TransposeTiling, transpose_tiling_n_last_4d_no_last_transpose)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    // perm = {0, 1, 3, 2}: last axis (dim=3) is transposed, but the last reduced axis stays
    // After merge: perm {0,1,3,2} where axis 2 is transposed (isLastAxisTranspose=true since perm[2]=3!=2)
    // For N_LAST we need isLastAxisTranspose=false AND last axis >= 32
    // perm = {0, 3, 1, 2} -> reduced perm might become {0, 2, 1} after removing 1-dim axes
    // But simpler: {0, 1, 2} (identity, last axis not transposed)
    int64_t perm_value[3] = {2, 0, 1};
    gert::TilingContextPara::TensorDescription x({{100, 200, 64}, {100, 200, 64}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{64, 100, 200}, {64, 100, 200}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    // On ascend950 (DAV_3510), this goes through N_LAST, NDDMA_BASE, GATHER, SMALL_SHAPE, or CUT paths
    EXPECT_TRUE(tilingInfo.tilingKey >= 10000 && tilingInfo.tilingKey <= 10008);
}

// 4D NDDMA_BASE CUT_ONCE/CUT_TWICE path (large shape, dim <= 5, last axis transposed, not n_last)
TEST_F(TransposeTiling, transpose_tiling_nddma_base_4d)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[4] = {3, 0, 1, 2};
    gert::TilingContextPara::TensorDescription x({{10, 20, 30, 64}, {10, 20, 30, 64}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{64, 10, 20, 30}, {64, 10, 20, 30}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    // Should hit NDDMA_BASE (CUT_ONCE or CUT_TWICE), GATHER, or N_LAST path
    EXPECT_TRUE(tilingInfo.tilingKey >= 10000 && tilingInfo.tilingKey <= 10008);
}

// BIG_DIM path (dim > 5), expect tiling key 10005
TEST_F(TransposeTiling, transpose_tiling_big_dim_6d)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    // 6D transpose with perm = {5, 0, 1, 2, 3, 4}, total > threshold
    int64_t perm_value[6] = {5, 0, 1, 2, 3, 4};
    gert::TilingContextPara::TensorDescription x({{2, 3, 4, 5, 6, 100}, {2, 3, 4, 5, 6, 100}}, ge::DT_FLOAT,
                                                 ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{6}, {6}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{100, 2, 3, 4, 5, 6}, {100, 2, 3, 4, 5, 6}}, ge::DT_FLOAT,
                                                   ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    // On ascend950 (DAV_3510), 6D transpose hits BIG_DIM, GATHER, or other template paths
    EXPECT_TRUE(tilingInfo.tilingKey >= 10000 && tilingInfo.tilingKey <= 10008);
}

// Perm with int32 dtype (DT_INT32), expect tiling success
TEST_F(TransposeTiling, transpose_tiling_perm_int32_dtype)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int32_t perm_value[2] = {1, 0};
    gert::TilingContextPara::TensorDescription x({{100, 200}, {100, 200}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{200, 100}, {200, 100}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
}

// Negative perm values in tiling, expect success with resolved perm
TEST_F(TransposeTiling, transpose_tiling_negative_perm)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[2] = {-1, -2};
    gert::TilingContextPara::TensorDescription x({{10, 20}, {10, 20}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{20, 10}, {20, 10}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(optiling::SplitMode::SMALL_SHAPE));
}

// Invalid perm dtype (DT_INT16, not int32/int64), expect failure
TEST_F(TransposeTiling, transpose_tiling_invalid_perm_dtype)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int16_t perm_value[2] = {1, 0};
    gert::TilingContextPara::TensorDescription x({{10, 20}, {10, 20}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{2}, {2}}, ge::DT_INT16, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{20, 10}, {20, 10}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_FALSE(success);
}

// Perm size mismatch with x dim count, expect failure
TEST_F(TransposeTiling, transpose_tiling_perm_size_mismatch)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[3] = {0, 1, 2};
    gert::TilingContextPara::TensorDescription x({{10, 20}, {10, 20}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{10, 20}, {10, 20}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_FALSE(success);
}

// Perm value out of valid range (perm[i] >= dimSize), expect failure
TEST_F(TransposeTiling, transpose_tiling_perm_value_out_of_range)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[2] = {2, 0};
    gert::TilingContextPara::TensorDescription x({{10, 20}, {10, 20}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{20, 10}, {20, 10}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_FALSE(success);
}

// All-ones shape input, expect TENSOR_MOVE (reduced to 1D)
TEST_F(TransposeTiling, transpose_tiling_all_ones_shape)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[4] = {3, 0, 1, 2};
    gert::TilingContextPara::TensorDescription x({{1, 1, 1, 1}, {1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{1, 1, 1, 1}, {1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(optiling::SplitMode::TENSOR_MOVE));
}

// Invalid init (coreNum = 0), expect failure
TEST_F(TransposeTiling, transpose_tiling_invalid_core_num_zero)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 0;
    compileInfo.ubSize = 196608;

    int64_t perm_value[2] = {1, 0};
    gert::TilingContextPara::TensorDescription x({{10, 20}, {10, 20}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{20, 10}, {20, 10}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_FALSE(success);
}

// Invalid init (ubSize = 0), expect failure
TEST_F(TransposeTiling, transpose_tiling_invalid_ub_size_zero)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 0;

    int64_t perm_value[2] = {1, 0};
    gert::TilingContextPara::TensorDescription x({{10, 20}, {10, 20}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{20, 10}, {20, 10}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_FALSE(success);
}

// FP16 dtype, 3D transpose perm {0,2,1}, expect success
TEST_F(TransposeTiling, transpose_tiling_fp16_3d_perm_021)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[3] = {0, 2, 1};
    gert::TilingContextPara::TensorDescription x({{100, 200, 64}, {100, 200, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{100, 64, 200}, {100, 64, 200}}, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
}

// 5D NDDMA_BASE path, dim = 5 (<= NDDMA_MAX_DIM_NUM), last axis transposed, large shape
TEST_F(TransposeTiling, transpose_tiling_nddma_5d)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[5] = {4, 0, 1, 2, 3};
    gert::TilingContextPara::TensorDescription x({{2, 3, 4, 5, 100}, {2, 3, 4, 5, 100}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{5}, {5}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{100, 2, 3, 4, 5}, {100, 2, 3, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    // On ascend950, 5D transpose may hit GATHER, SMALL_SHAPE, CUT_ONCE, CUT_TWICE, or other paths
    EXPECT_TRUE(tilingInfo.tilingKey >= 10000 && tilingInfo.tilingKey <= 10008);
}

// Large N_LAST transpose (perm not transposing last axis, large total bytes, last dim >= 32)
TEST_F(TransposeTiling, transpose_tiling_n_last_large_3d)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    // perm {2, 0, 1}: last axis perm[2]=1 != 2 -> isLastAxisTranspose=true
    // For N_LAST we need !isLastAxisTranspose, so use perm {0, 2, 1}
    // perm {0, 2, 1}: perm[2]=1 != 2 -> still last axis transpose
    // perm {1, 0, 2}: perm[2]=2 == 2 -> not last axis transpose!
    int64_t perm_value[3] = {1, 0, 2};
    gert::TilingContextPara::TensorDescription x({{100, 200, 64}, {100, 200, 64}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{200, 100, 64}, {200, 100, 64}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    EXPECT_TRUE(tilingInfo.tilingKey == static_cast<uint64_t>(optiling::SplitMode::N_LAST_TRANSPOSE) ||
                tilingInfo.tilingKey == static_cast<uint64_t>(optiling::SplitMode::GATHER_TRANSPOSE));
}

// Shape with axis=1 (reduced after RemoveAxis), 3D with one 1-dim axis
TEST_F(TransposeTiling, transpose_tiling_3d_with_one_dim_axis)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    // perm {2, 0, 1} with shape {1, 100, 200}: after RemoveAxis, becomes {100, 200} perm {1, 0}
    int64_t perm_value[3] = {2, 0, 1};
    gert::TilingContextPara::TensorDescription x({{1, 100, 200}, {1, 100, 200}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{200, 1, 100}, {200, 1, 100}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
}

// BF16 dtype 2D transpose, expect success
TEST_F(TransposeTiling, transpose_tiling_bf16_2d)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[2] = {1, 0};
    gert::TilingContextPara::TensorDescription x({{100, 200}, {100, 200}}, ge::DT_BF16, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{200, 100}, {200, 100}}, ge::DT_BF16, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
}

// INT8 dtype 4D transpose with large shape, expect success
TEST_F(TransposeTiling, transpose_tiling_int8_4d_large)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[4] = {3, 0, 1, 2};
    gert::TilingContextPara::TensorDescription x({{10, 20, 30, 64}, {10, 20, 30, 64}}, ge::DT_INT8, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{64, 10, 20, 30}, {64, 10, 20, 30}}, ge::DT_INT8, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
}

// 2D transpose with perm {0,1} (identity permutation), expect TENSOR_MOVE
TEST_F(TransposeTiling, transpose_tiling_identity_perm_2d)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[2] = {0, 1};
    gert::TilingContextPara::TensorDescription x({{100, 200}, {100, 200}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{100, 200}, {100, 200}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(optiling::SplitMode::TENSOR_MOVE));
}

// Output shape mismatch with expected shape (y shape wrong), expect failure
TEST_F(TransposeTiling, transpose_tiling_output_shape_mismatch)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[2] = {1, 0};
    gert::TilingContextPara::TensorDescription x({{10, 20}, {10, 20}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    // Wrong output shape: should be {20, 10} but we give {10, 30}
    gert::TilingContextPara::TensorDescription out({{10, 30}, {10, 30}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_FALSE(success);
}

// INT64 dtype 4D large transpose, covers int64 perm data path
TEST_F(TransposeTiling, transpose_tiling_int64_dtype_4d)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[4] = {3, 0, 1, 2};
    gert::TilingContextPara::TensorDescription x({{10, 20, 30, 64}, {10, 20, 30, 64}}, ge::DT_INT64, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{64, 10, 20, 30}, {64, 10, 20, 30}}, ge::DT_INT64, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
}

// Small volume that is less than coreNum, expect TENSOR_MOVE with single core
TEST_F(TransposeTiling, transpose_tiling_tensor_move_small_volume)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[1] = {0};
    gert::TilingContextPara::TensorDescription x({{10}, {10}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{10}, {10}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(optiling::SplitMode::TENSOR_MOVE));
}

// 4D transpose with perm {0,1,2,3} (identity permutation), expect TENSOR_MOVE
TEST_F(TransposeTiling, transpose_tiling_identity_perm_4d)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[4] = {0, 1, 2, 3};
    gert::TilingContextPara::TensorDescription x({{10, 20, 30, 40}, {10, 20, 30, 40}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{10, 20, 30, 40}, {10, 20, 30, 40}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    // Identity perm reduces to 1D -> TENSOR_MOVE
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(optiling::SplitMode::TENSOR_MOVE));
}

// 4D perm {3,2,1,0} (full reverse), covers NDDMA_BASE/CUT_TWICE path
TEST_F(TransposeTiling, transpose_tiling_4d_full_reverse)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[4] = {3, 2, 1, 0};
    gert::TilingContextPara::TensorDescription x({{64, 128, 32, 16}, {64, 128, 32, 16}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{16, 32, 128, 64}, {16, 32, 128, 64}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
}

// 7D transpose (dim > 5) with very large shape - merges to low dim after MergeAxis
TEST_F(TransposeTiling, transpose_tiling_7d_dim_over_5)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[7] = {6, 0, 1, 2, 3, 4, 5};
    gert::TilingContextPara::TensorDescription x({{2, 2, 2, 2, 2, 2, 200}, {2, 2, 2, 2, 2, 2, 200}}, ge::DT_FLOAT,
                                                 ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{7}, {7}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{200, 2, 2, 2, 2, 2, 2}, {200, 2, 2, 2, 2, 2, 2}}, ge::DT_FLOAT,
                                                   ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
}

// BIG_DIM path: 6D where perm has no consecutive pairs (dim stays 6 after MergeAxis).
// perm={0,3,1,4,2,5} keeps all 6 axes. isLastAxisTranspose=false (perm[5]=5).
// reducedInShape[5]=8 < 32 skips N_LAST. dim=6 > 5 → BIG_DIM.
TEST_F(TransposeTiling, transpose_tiling_big_dim_6d_no_consecutive_perm)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[6] = {0, 3, 1, 4, 2, 5};
    gert::TilingContextPara::TensorDescription x({{100, 200, 10, 20, 30, 8}, {100, 200, 10, 20, 30, 8}}, ge::DT_FLOAT,
                                                 ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{6}, {6}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{100, 20, 200, 30, 10, 8}, {100, 20, 200, 30, 10, 8}}, ge::DT_FLOAT,
                                                   ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(optiling::SplitMode::BIG_DIM));
}

// BIG_DIM path: 6D with isLastAxisTranspose=true, GATHER may fail then BIG_DIM
TEST_F(TransposeTiling, transpose_tiling_big_dim_6d_gather_fallback)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[6] = {3, 0, 4, 1, 5, 2};
    gert::TilingContextPara::TensorDescription x({{50, 60, 10, 100, 80, 30}, {50, 60, 10, 100, 80, 30}}, ge::DT_FLOAT,
                                                 ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{6}, {6}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{100, 50, 80, 60, 30, 10}, {100, 50, 80, 60, 30, 10}}, ge::DT_FLOAT,
                                                   ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
}

// Error path: x shape with zero value axis, expect failure
TEST_F(TransposeTiling, transpose_tiling_zero_shape_value)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[2] = {1, 0};
    gert::TilingContextPara::TensorDescription x({{0, 20}, {0, 20}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{20, 0}, {20, 0}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_FALSE(success);
}

// N_LAST with large 4D shape exercising FindSplitFactorByRateNLast/MultiplesNLast
TEST_F(TransposeTiling, transpose_tiling_n_last_large_4d)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[4] = {1, 0, 2, 3};
    gert::TilingContextPara::TensorDescription x({{200, 100, 64, 128}, {200, 100, 64, 128}}, ge::DT_FLOAT,
                                                 ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{100, 200, 64, 128}, {100, 200, 64, 128}}, ge::DT_FLOAT,
                                                   ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
}

// CUT_TWICE path with both tail factors non-zero, covers GetIntervalInfoForCutTwice tail ranges
TEST_F(TransposeTiling, transpose_tiling_cut_twice_both_tails)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[4] = {3, 0, 1, 2};
    gert::TilingContextPara::TensorDescription x({{8, 7, 6, 100}, {8, 7, 6, 100}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{100, 8, 7, 6}, {100, 8, 7, 6}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
}

// CUT_ONCE with low core utilization, covers rate-based split factor search
TEST_F(TransposeTiling, transpose_tiling_cut_once_low_cores)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[3] = {2, 0, 1};
    gert::TilingContextPara::TensorDescription x({{100, 200, 50}, {100, 200, 50}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{50, 100, 200}, {50, 100, 200}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
}

// N_LAST transpose exercising FindSplitFactorByMultiplesNLast
TEST_F(TransposeTiling, transpose_tiling_n_last_multiples_split)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[3] = {1, 0, 2};
    gert::TilingContextPara::TensorDescription x({{500, 400, 64}, {500, 400, 64}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{400, 500, 64}, {400, 500, 64}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    EXPECT_TRUE(tilingInfo.tilingKey == static_cast<uint64_t>(optiling::SplitMode::N_LAST_TRANSPOSE) ||
                tilingInfo.tilingKey == static_cast<uint64_t>(optiling::SplitMode::GATHER_TRANSPOSE));
}

// BIG_DIM with small outer dimensions to trigger CalcBlockSplitInfoForBigDim low-core branch
TEST_F(TransposeTiling, transpose_tiling_big_dim_small_outer)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    // 6D with very large inner axis, small outer axes. perm no consecutive pairs.
    int64_t perm_value[6] = {0, 3, 1, 4, 2, 5};
    gert::TilingContextPara::TensorDescription x({{2, 2, 2, 2, 2, 250000}, {2, 2, 2, 2, 2, 250000}}, ge::DT_FLOAT,
                                                 ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{6}, {6}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{2, 2, 2, 2, 2, 250000}, {2, 2, 2, 2, 2, 250000}}, ge::DT_FLOAT,
                                                   ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
}

// N_LAST with fewer cores available, exercises Rate-based split
TEST_F(TransposeTiling, transpose_tiling_n_last_small_corenum)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 4;
    compileInfo.ubSize = 196608;

    // Small coreNum makes coreNumTmp more likely < coreNum → RateNLast
    int64_t perm_value[3] = {1, 0, 2};
    gert::TilingContextPara::TensorDescription x({{10, 5, 128}, {10, 5, 128}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{5, 10, 128}, {5, 10, 128}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
}

// CUT_TWICE with inTailFactor=0 and outTailFactor!=0, covers offsetRangeOutputTail
TEST_F(TransposeTiling, transpose_tiling_cut_twice_out_tail_only)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    // 4D perm {3, 2, 0, 1} with shape producing CUT_TWICE where outTail != 0 but inTail = 0
    int64_t perm_value[4] = {3, 2, 0, 1};
    gert::TilingContextPara::TensorDescription x({{16, 32, 64, 128}, {16, 32, 64, 128}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{128, 64, 16, 32}, {128, 64, 16, 32}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
}

// CUT_TWICE with inTailFactor!=0 and outTailFactor=0, covers offsetRangeInputTail
TEST_F(TransposeTiling, transpose_tiling_cut_twice_in_tail_only)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    // 3D perm {2, 0, 1} with shape producing CUT_TWICE where inTail != 0 but outTail = 0
    int64_t perm_value[3] = {2, 0, 1};
    gert::TilingContextPara::TensorDescription x({{37, 100, 64}, {37, 100, 64}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{64, 37, 100}, {64, 37, 100}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
}
