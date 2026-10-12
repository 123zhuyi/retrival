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
 * \file test_axpy_tiling_arch35.cpp
 * \brief
 */

#include "../../../../op_host/arch35/axpy_tiling_arch35.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

class AxpyTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "AxpyTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AxpyTiling TearDown" << std::endl;
    }
};

TEST_F(AxpyTiling, axpy_test_float16)
{
    optiling::AxpyCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Axpy",
                                                      {
                                                        {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"alpha", Ops::Math::AnyValue::CreateFrom<float>(1.0f)}
                                                      },
                                                      &compileInfo);
    uint64_t expectTilingKey = 7;
    string expectTilingData = "1 137438953696 10 1 1 1065353216 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AxpyTiling, axpy_test_float32)
{
    optiling::AxpyCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Axpy",
                                                      {
                                                        {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"alpha", Ops::Math::AnyValue::CreateFrom<float>(1.0f)}
                                                      },
                                                      &compileInfo
                                                      );
    uint64_t expectTilingKey = 7;
    string expectTilingData = "1 137438953760 8 1 1 1065353216 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AxpyTiling, axpy_test_int32)
{
    optiling::AxpyCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Axpy",
                                                      {
                                                        {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"alpha", Ops::Math::AnyValue::CreateFrom<float>(1.0f)}
                                                      },
                                                      &compileInfo
                                                      );
    uint64_t expectTilingKey = 7;
    string expectTilingData = "1 137438953696 10 1 1 1065353216 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AxpyTiling, axpy_test_dtype_not_support)
{
    optiling::AxpyCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Axpy",
                                                      {
                                                        {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, ge::DT_INT8, ge::FORMAT_ND},
                                                        {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_INT8, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_INT8, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"alpha", Ops::Math::AnyValue::CreateFrom<float>(1.0f)}
                                                      },
                                                      &compileInfo
                                                      );
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(AxpyTiling, axpy_test_dtype_not_same)
{
    optiling::AxpyCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Axpy",
                                                      {
                                                        {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"alpha", Ops::Math::AnyValue::CreateFrom<float>(1.0f)}
                                                      },
                                                      &compileInfo
                                                      );
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}