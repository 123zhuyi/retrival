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
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "conversion/masked_select_v3/op_host/masked_select_v3_tiling.h"

#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;
using namespace gert;
using namespace optiling;

class l2_masked_select_regbase_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "masked_select_test Tiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "masked_select_test Tiling TearDown" << std::endl; }
};

TEST_F(l2_masked_select_regbase_test, regbase_fp16_scalar_input)
{
    optiling::MaskedSelectV3CompileInfo compileInfo = {48, 196608, 16777216, true};

    gert::TilingContextPara tilingContextPara(
        "MaskedSelectV3", {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo);
    TilingInfo tilingInfo;
    bool ok = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ok);
}

TEST_F(l2_masked_select_regbase_test, regbase_fp32_1d)
{
    optiling::MaskedSelectV3CompileInfo compileInfo = {48, 196608, 16777216, true};

    gert::TilingContextPara tilingContextPara(
        "MaskedSelectV3", {{{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{8}, {8}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);
    TilingInfo tilingInfo;
    bool ok = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ok);
}

TEST_F(l2_masked_select_regbase_test, regbase_int32_large)
{
    optiling::MaskedSelectV3CompileInfo compileInfo = {48, 196608, 16777216, true};

    gert::TilingContextPara tilingContextPara("MaskedSelectV3",
                                              {{{{1024, 1024}, {1024, 1024}}, ge::DT_INT32, ge::FORMAT_ND},
                                               {{{1024, 1024}, {1024, 1024}}, ge::DT_BOOL, ge::FORMAT_ND}},
                                              {{{{1024, 1024}, {1024, 1024}}, ge::DT_INT32, ge::FORMAT_ND}},
                                              &compileInfo);
    TilingInfo tilingInfo;
    bool ok = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ok);
}

TEST_F(l2_masked_select_regbase_test, regbase_int64_large)
{
    optiling::MaskedSelectV3CompileInfo compileInfo = {48, 196608, 16777216, true};

    gert::TilingContextPara tilingContextPara("MaskedSelectV3",
                                              {{{{512, 512}, {512, 512}}, ge::DT_INT64, ge::FORMAT_ND},
                                               {{{512, 512}, {512, 512}}, ge::DT_BOOL, ge::FORMAT_ND}},
                                              {{{{512, 512}, {512, 512}}, ge::DT_INT64, ge::FORMAT_ND}}, &compileInfo);
    TilingInfo tilingInfo;
    bool ok = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ok);
}

TEST_F(l2_masked_select_regbase_test, regbase_float16_3d)
{
    optiling::MaskedSelectV3CompileInfo compileInfo = {48, 196608, 16777216, true};

    gert::TilingContextPara tilingContextPara("MaskedSelectV3",
                                              {{{{2, 100, 200}, {2, 100, 200}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{2, 100, 200}, {2, 100, 200}}, ge::DT_BOOL, ge::FORMAT_ND}},
                                              {{{{2, 100, 200}, {2, 100, 200}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                              &compileInfo);
    TilingInfo tilingInfo;
    bool ok = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ok);
}

TEST_F(l2_masked_select_regbase_test, regbase_bf16_1d)
{
    optiling::MaskedSelectV3CompileInfo compileInfo = {48, 196608, 16777216, true};

    gert::TilingContextPara tilingContextPara(
        "MaskedSelectV3", {{{{256}, {256}}, ge::DT_BF16, ge::FORMAT_ND}, {{{256}, {256}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{{{256}, {256}}, ge::DT_BF16, ge::FORMAT_ND}}, &compileInfo);
    TilingInfo tilingInfo;
    bool ok = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ok);
}

TEST_F(l2_masked_select_regbase_test, regbase_int8_1d)
{
    optiling::MaskedSelectV3CompileInfo compileInfo = {48, 196608, 16777216, true};

    gert::TilingContextPara tilingContextPara(
        "MaskedSelectV3", {{{{512}, {512}}, ge::DT_INT8, ge::FORMAT_ND}, {{{512}, {512}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{{{512}, {512}}, ge::DT_INT8, ge::FORMAT_ND}}, &compileInfo);
    TilingInfo tilingInfo;
    bool ok = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ok);
}

TEST_F(l2_masked_select_regbase_test, regbase_uint16_2d)
{
    optiling::MaskedSelectV3CompileInfo compileInfo = {48, 196608, 16777216, true};

    gert::TilingContextPara tilingContextPara(
        "MaskedSelectV3",
        {{{{64, 128}, {64, 128}}, ge::DT_UINT16, ge::FORMAT_ND}, {{{64, 128}, {64, 128}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{{{64, 128}, {64, 128}}, ge::DT_UINT16, ge::FORMAT_ND}}, &compileInfo);
    TilingInfo tilingInfo;
    bool ok = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ok);
}

TEST_F(l2_masked_select_regbase_test, regbase_shape_mismatch_failed)
{
    optiling::MaskedSelectV3CompileInfo compileInfo = {48, 196608, 16777216, true};

    gert::TilingContextPara tilingContextPara(
        "MaskedSelectV3", {{{{8}, {8}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{16}, {16}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{{{8}, {8}}, ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo);
    TilingInfo tilingInfo;
    bool ok = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_FALSE(ok);
}

TEST_F(l2_masked_select_regbase_test, regbase_zero_shape_failed)
{
    optiling::MaskedSelectV3CompileInfo compileInfo = {48, 196608, 16777216, true};

    gert::TilingContextPara tilingContextPara(
        "MaskedSelectV3", {{{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{0}, {0}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);
    TilingInfo tilingInfo;
    bool ok = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_FALSE(ok);
}

TEST_F(l2_masked_select_regbase_test, regbase_uint32_1d)
{
    optiling::MaskedSelectV3CompileInfo compileInfo = {48, 196608, 16777216, true};

    gert::TilingContextPara tilingContextPara(
        "MaskedSelectV3",
        {{{{256}, {256}}, ge::DT_UINT32, ge::FORMAT_ND}, {{{256}, {256}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{{{256}, {256}}, ge::DT_UINT32, ge::FORMAT_ND}}, &compileInfo);
    TilingInfo tilingInfo;
    bool ok = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ok);
}

TEST_F(l2_masked_select_regbase_test, regbase_double_1d)
{
    optiling::MaskedSelectV3CompileInfo compileInfo = {48, 196608, 16777216, true};

    gert::TilingContextPara tilingContextPara(
        "MaskedSelectV3",
        {{{{128}, {128}}, ge::DT_DOUBLE, ge::FORMAT_ND}, {{{128}, {128}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{{{128}, {128}}, ge::DT_DOUBLE, ge::FORMAT_ND}}, &compileInfo);
    TilingInfo tilingInfo;
    bool ok = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ok);
}

TEST_F(l2_masked_select_regbase_test, regbase_bool_1d)
{
    optiling::MaskedSelectV3CompileInfo compileInfo = {48, 196608, 16777216, true};

    gert::TilingContextPara tilingContextPara(
        "MaskedSelectV3", {{{{512}, {512}}, ge::DT_BOOL, ge::FORMAT_ND}, {{{512}, {512}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{{{512}, {512}}, ge::DT_BOOL, ge::FORMAT_ND}}, &compileInfo);
    TilingInfo tilingInfo;
    bool ok = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ok);
}

TEST_F(l2_masked_select_regbase_test, regbase_int16_1d)
{
    optiling::MaskedSelectV3CompileInfo compileInfo = {48, 196608, 16777216, true};

    gert::TilingContextPara tilingContextPara(
        "MaskedSelectV3", {{{{256}, {256}}, ge::DT_INT16, ge::FORMAT_ND}, {{{256}, {256}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{{{256}, {256}}, ge::DT_INT16, ge::FORMAT_ND}}, &compileInfo);
    TilingInfo tilingInfo;
    bool ok = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ok);
}

TEST_F(l2_masked_select_regbase_test, regbase_uint8_1d)
{
    optiling::MaskedSelectV3CompileInfo compileInfo = {48, 196608, 16777216, true};

    gert::TilingContextPara tilingContextPara(
        "MaskedSelectV3", {{{{512}, {512}}, ge::DT_UINT8, ge::FORMAT_ND}, {{{512}, {512}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{{{512}, {512}}, ge::DT_UINT8, ge::FORMAT_ND}}, &compileInfo);
    TilingInfo tilingInfo;
    bool ok = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ok);
}

TEST_F(l2_masked_select_regbase_test, regbase_uint64_1d)
{
    optiling::MaskedSelectV3CompileInfo compileInfo = {48, 196608, 16777216, true};

    gert::TilingContextPara tilingContextPara(
        "MaskedSelectV3",
        {{{{128}, {128}}, ge::DT_UINT64, ge::FORMAT_ND}, {{{128}, {128}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{{{128}, {128}}, ge::DT_UINT64, ge::FORMAT_ND}}, &compileInfo);
    TilingInfo tilingInfo;
    bool ok = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ok);
}
