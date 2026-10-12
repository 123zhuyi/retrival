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
 * \file test_confusion_transpose_d_tiling_arch35.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/confusion_transpose_d_tiling_arch35.h"

using namespace ge;

class ConfusionTransposeDTilingTest : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "ConfusionTransposeDTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ConfusionTransposeDTilingTest TearDown" << std::endl;
    }
};

TEST_F(ConfusionTransposeDTilingTest, ConfusionTransposeDTiling_Not_Transpose_First)
{
    // compile info
    optiling::ConfusionTransposeDCompileInfo compileInfo;
    compileInfo.transposeCompilerInfo.coreNum = 64;
    compileInfo.transposeCompilerInfo.ubSize = 253952;
    
    gert::TilingContextPara tilingContextPara(
        "ConfusionTransposeD",
        {
            {{{21, 27, 14, 6}, {21, 27, 14, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{162, 14, 21}, {162, 14, 21}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("perm", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 0})),
            gert::TilingContextPara::OpAttr("shape", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({21, 162, 14})),
            gert::TilingContextPara::OpAttr("transpose_first", Ops::Math::AnyValue::CreateFrom<bool>(false))
        },&compileInfo);

    uint64_t expectTilingKey = 10001;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(ConfusionTransposeDTilingTest, ConfusionTransposeDTiling_Transpose_First)
{
    // compile info
    optiling::ConfusionTransposeDCompileInfo compileInfo;
    
    compileInfo.transposeCompilerInfo.coreNum = 64;
    compileInfo.transposeCompilerInfo.ubSize = 253952;
    
    gert::TilingContextPara tilingContextPara(
        "ConfusionTransposeD",
        {
            {{{6, 31, 18, 4}, {6, 31, 18, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{6, 124, 18}, {6, 124, 18}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("perm", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({3, 2, 0, 1})),
            gert::TilingContextPara::OpAttr("shape", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({6, 124, 18})),
            gert::TilingContextPara::OpAttr("transpose_first", Ops::Math::AnyValue::CreateFrom<bool>(true))
        }, &compileInfo);

    uint64_t expectTilingKey = 10001;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(ConfusionTransposeDTilingTest, ConfusionTransposeDTiling_Fail_01)
{
    // compile info
    optiling::ConfusionTransposeDCompileInfo compileInfo;
    
    compileInfo.transposeCompilerInfo.coreNum = 64;
    compileInfo.transposeCompilerInfo.ubSize = 253952;
    
    gert::TilingContextPara tilingContextPara(
        "ConfusionTransposeD",
        {
            {{{21, 27, 16, 16}, {21, 27, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{3, 189, 16, 16}, {3, 189, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("perm", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})),
            gert::TilingContextPara::OpAttr("shape", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({3024, 48})),
            gert::TilingContextPara::OpAttr("transpose_first", Ops::Math::AnyValue::CreateFrom<bool>(true))
        }, &compileInfo);

    uint64_t expectTilingKey = 10001;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara,ge::GRAPH_FAILED, expectTilingKey, expectWorkspaces);
}

TEST_F(ConfusionTransposeDTilingTest, ConfusionTransposeDTiling_Fail_02)
{
    // compile info
    optiling::ConfusionTransposeDCompileInfo compileInfo;
    
    compileInfo.transposeCompilerInfo.coreNum = 64;
    compileInfo.transposeCompilerInfo.ubSize = 253952;
    
    gert::TilingContextPara tilingContextPara(
        "ConfusionTransposeD",
        {
            {{{6, 31, 18, 4}, {6, 31, 18, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{6, 124, 18}, {6, 124, 18}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("perm", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({5, 2, 0, 1})),
            gert::TilingContextPara::OpAttr("shape", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({6, 124, 18})),
            gert::TilingContextPara::OpAttr("transpose_first", Ops::Math::AnyValue::CreateFrom<bool>(true))
        }, &compileInfo);

    uint64_t expectTilingKey = 10001;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara,ge::GRAPH_FAILED, expectTilingKey, expectWorkspaces);
}

TEST_F(ConfusionTransposeDTilingTest, ConfusionTransposeDTiling_Fail_03)
{
    // compile info
    optiling::ConfusionTransposeDCompileInfo compileInfo;
    
    compileInfo.transposeCompilerInfo.coreNum = 64;
    compileInfo.transposeCompilerInfo.ubSize = 253952;
    
    gert::TilingContextPara tilingContextPara(
        "ConfusionTransposeD",
        {
            {{{21, 27, 14, 6}, {21, 27, 14, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{162, 14, 21}, {162, 14, 21}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("perm", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 0})),
            gert::TilingContextPara::OpAttr("shape", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({21, 162, 13})),
            gert::TilingContextPara::OpAttr("transpose_first", Ops::Math::AnyValue::CreateFrom<bool>(false))
        }, &compileInfo);

    uint64_t expectTilingKey = 10001;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara,ge::GRAPH_FAILED, expectTilingKey, expectWorkspaces);
}

TEST_F(ConfusionTransposeDTilingTest, ConfusionTransposeDTiling_Fail_04)
{
    // compile info
    optiling::ConfusionTransposeDCompileInfo compileInfo;
    
    compileInfo.transposeCompilerInfo.coreNum = 64;
    compileInfo.transposeCompilerInfo.ubSize = 253952;
    
    gert::TilingContextPara tilingContextPara(
        "ConfusionTransposeD",
        {
            {{{0, 27, 14, 6}, {0, 27, 14, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{162, 14, 0}, {162, 14, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("perm", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 0})),
            gert::TilingContextPara::OpAttr("shape", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0, 162, 14})),
            gert::TilingContextPara::OpAttr("transpose_first", Ops::Math::AnyValue::CreateFrom<bool>(false))
        }, &compileInfo);

    uint64_t expectTilingKey = 10001;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara,ge::GRAPH_FAILED, expectTilingKey, expectWorkspaces);
}
