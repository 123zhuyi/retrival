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
#include <gtest/gtest.h>
#include "../../../../op_host/arch35/muls_tiling_arch35.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "atvoss/elewise/elewise_tiling.h"

using namespace std;

class MulsTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "MulsTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "MulsTilingTest TearDown" << std::endl;
    }
};

TEST_F(MulsTilingTest, muls_test_tiling_001)
{
    Ops::Base::ElewiseCompileInfo compileInfo = {64, 262144};

    gert::TilingContextPara tilingContextPara(
        "Muls",
        {
            {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("value", Ops::Math::AnyValue::CreateFrom<float>(1.0)),
        },
        &compileInfo);

    string expectTilingData = "4096 2 10880 2048 2 1 1 2048 2048 10880 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCaseForEle(tilingContextPara, ge::GRAPH_SUCCESS, false, 0, true, expectTilingData, expectWorkspaces);
}

TEST_F(MulsTilingTest, muls_test_tiling_002)
{
    Ops::Base::ElewiseCompileInfo compileInfo = {64, 262144};

    gert::TilingContextPara tilingContextPara(
        "Muls",
        {
            {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("value", Ops::Math::AnyValue::CreateFrom<float>(1.0)),
        },
        &compileInfo);

    string expectTilingData = "4096 4 16384 1024 4 1 1 1024 1024 16384 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCaseForEle(tilingContextPara, ge::GRAPH_SUCCESS, false, 0, true, expectTilingData, expectWorkspaces);
}

TEST_F(MulsTilingTest, muls_test_tiling_003)
{
    Ops::Base::ElewiseCompileInfo compileInfo = {64, 262144};

    gert::TilingContextPara tilingContextPara(
        "Muls",
        {
            {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("value", Ops::Math::AnyValue::CreateFrom<float>(1.0)),
        },
        &compileInfo);

    string expectTilingData = "4096 4 10880 1024 4 1 1 1024 1024 10880 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCaseForEle(tilingContextPara, ge::GRAPH_SUCCESS, false, 0, true, expectTilingData, expectWorkspaces);
}

TEST_F(MulsTilingTest, muls_test_tiling_004)
{
    Ops::Base::ElewiseCompileInfo compileInfo = {64, 262144};

    gert::TilingContextPara tilingContextPara(
        "Muls",
        {
            {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_INT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_INT16, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("value", Ops::Math::AnyValue::CreateFrom<float>(1.0)),
        },
        &compileInfo);

    string expectTilingData = "4096 2 10880 2048 2 1 1 2048 2048 10880 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCaseForEle(tilingContextPara, ge::GRAPH_SUCCESS, false, 0, true, expectTilingData, expectWorkspaces);
}

TEST_F(MulsTilingTest, muls_test_tiling_005)
{
    Ops::Base::ElewiseCompileInfo compileInfo = {64, 262144};

    gert::TilingContextPara tilingContextPara(
        "Muls",
        {
            {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("value", Ops::Math::AnyValue::CreateFrom<float>(1.0)),
        },
        &compileInfo);

    string expectTilingData = "4096 4 5440 1024 4 1 1 1024 1024 5440 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCaseForEle(tilingContextPara, ge::GRAPH_SUCCESS, false, 0, true, expectTilingData, expectWorkspaces);
}

TEST_F(MulsTilingTest, muls_test_tiling_006)
{
    Ops::Base::ElewiseCompileInfo compileInfo = {64, 262144};

    gert::TilingContextPara tilingContextPara(
        "Muls",
        {
            {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("value", Ops::Math::AnyValue::CreateFrom<float>(1.0)),
        },
        &compileInfo);

    string expectTilingData = "4096 2 10880 2048 2 1 1 2048 2048 10880 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCaseForEle(tilingContextPara, ge::GRAPH_SUCCESS, false, 0, true, expectTilingData, expectWorkspaces);
}

TEST_F(MulsTilingTest, muls_test_tiling_007)
{
    Ops::Base::ElewiseCompileInfo compileInfo = {64, 262144};

    gert::TilingContextPara tilingContextPara(
        "Muls",
        {
            {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_COMPLEX32, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_COMPLEX32, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("value", Ops::Math::AnyValue::CreateFrom<float>(1.0)),
        },
        &compileInfo);

    string expectTilingData = "4096 4 5440 1024 4 1 1 1024 1024 5440 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCaseForEle(tilingContextPara, ge::GRAPH_SUCCESS, false, 0, true, expectTilingData, expectWorkspaces);
}

TEST_F(MulsTilingTest, muls_test_tiling_008)
{
    Ops::Base::ElewiseCompileInfo compileInfo = {64, 262144};

    gert::TilingContextPara tilingContextPara(
        "Muls",
        {
            {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("value", Ops::Math::AnyValue::CreateFrom<float>(1.0)),
        },
        &compileInfo);

    string expectTilingData = "4096 8 8192 512 8 1 1 512 512 8192 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCaseForEle(tilingContextPara, ge::GRAPH_SUCCESS, false, 0, true, expectTilingData, expectWorkspaces);
}

TEST_F(MulsTilingTest, muls_test_tiling_009)
{
    Ops::Base::ElewiseCompileInfo compileInfo = {64, 262144};

    gert::TilingContextPara tilingContextPara(
        "Muls",
        {
            {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("value", Ops::Math::AnyValue::CreateFrom<float>(1.0)),
        },
        &compileInfo);

    string expectTilingData = "4096 2 10880 2048 2 1 1 2048 2048 10880 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCaseForEle(tilingContextPara, ge::GRAPH_FAILED, false, 0, true, expectTilingData, expectWorkspaces);
}
