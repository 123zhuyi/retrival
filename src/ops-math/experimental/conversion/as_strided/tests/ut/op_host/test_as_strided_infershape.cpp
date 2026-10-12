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
 * \file test_as_strided_infershape.cpp
 * \brief AsStrided infershape UT.
 */

#include <gtest/gtest.h>
#include <vector>

#include "infershape_case_executor.h"
#include "infershape_context_faker.h"

class AsStridedInfershape : public testing::Test {};

TEST_F(AsStridedInfershape, infer_shape_from_int64_size)
{
    std::vector<int64_t> sizeValue = {2, 3, 4};
    std::vector<int64_t> strideValue = {12, 4, 1};
    std::vector<int64_t> storageOffsetValue = {0};

    gert::InfershapeContextPara infershapeContextPara(
        "AsStrided",
        {
            {{{24}, {24}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, sizeValue.data()},
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, strideValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, storageOffsetValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3, 4}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AsStridedInfershape, infer_shape_from_int32_size)
{
    std::vector<int32_t> sizeValue = {4, 2};
    std::vector<int32_t> strideValue = {2, 1};
    std::vector<int32_t> storageOffsetValue = {1};

    gert::InfershapeContextPara infershapeContextPara(
        "AsStrided",
        {
            {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeValue.data()},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, strideValue.data()},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, storageOffsetValue.data()},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AsStridedInfershape, infer_shape_rejects_unsupported_size_dtype)
{
    std::vector<uint32_t> sizeValue = {4};
    std::vector<int64_t> strideValue = {1};
    std::vector<int64_t> storageOffsetValue = {0};

    gert::InfershapeContextPara infershapeContextPara(
        "AsStrided",
        {
            {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_UINT32, ge::FORMAT_ND, true, sizeValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, strideValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, storageOffsetValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}
