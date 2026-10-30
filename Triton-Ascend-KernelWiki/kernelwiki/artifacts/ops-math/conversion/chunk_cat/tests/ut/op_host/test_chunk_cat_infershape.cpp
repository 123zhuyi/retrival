/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_chunk_cat_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class ChunkCatInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ChunkCatInfershapeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ChunkCatInfershapeTest TearDown" << std::endl;
    }
};

TEST_F(ChunkCatInfershapeTest, chunk_cat_infer_shape_fp16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ChunkCat",
        {
            {{{4, 16}, {4, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 8}, {4, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 12}, {4, 12}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dim", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"num_chunks", Ops::Math::AnyValue::CreateFrom<int64_t>(4)},
        },
        {3}, {1}
        );
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 36},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ChunkCatInfershapeTest, chunk_cat_infer_shape_fp16_1)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ChunkCat",
        {
            {{{5, 2}, {5, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{5, 3}, {5, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dim", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"num_chunks", Ops::Math::AnyValue::CreateFrom<int64_t>(5)},
        },
        {2}, {1}
        );
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {5, 5},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
