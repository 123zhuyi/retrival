/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "../../../../op_host/arch35/stateless_random_choice_with_mask_simt_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class StatelessRandomChoiceWithMaskTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "StatelessRandomChoiceWithMask SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "StatelessRandomChoiceWithMask TearDown" << std::endl;
    }
};

TEST_F(StatelessRandomChoiceWithMaskTiling, stateless_random_choice_with_mask_tiling_001)
{
    optiling::StatelessRandomChoiceWithMaskCompileInfo compileInfo = {64, 196608};
    gert::StorageShape xShape = {{1}, {1}};
    gert::StorageShape countShape = {{1}, {1}};
    gert::StorageShape seedShape = {{1}, {1}};
    gert::StorageShape offsetShape = {{1}, {1}};
    gert::StorageShape yShape = {{1, 1}, {1, 1}};
    gert::StorageShape maskShape = {{1}, {1}};

    vector<int32_t> countValue = {20};
    vector<int64_t> seedValue = {12};
    vector<int64_t> offsetValue = {22};

    gert::TilingContextPara tilingContextPara(
        "StatelessRandomChoiceWithMask",
        {{xShape, ge::DT_BOOL, ge::FORMAT_ND},
         {countShape, ge::DT_INT32, ge::FORMAT_ND, true, countValue.data()},
         {seedShape, ge::DT_INT64, ge::FORMAT_ND, true, seedValue.data()},
         {offsetShape, ge::DT_INT64, ge::FORMAT_ND, true, offsetValue.data()}},
        {{yShape, ge::DT_INT32, ge::FORMAT_ND}, {maskShape, ge::DT_BOOL, ge::FORMAT_ND}},
        &compileInfo);

    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);
    EXPECT_EQ(tilingInfo.tilingKey, 0);
    EXPECT_EQ(tilingInfo.blockNum, 1);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], 16787456);
}

TEST_F(StatelessRandomChoiceWithMaskTiling, stateless_random_choice_with_mask_tiling_002)
{
    optiling::StatelessRandomChoiceWithMaskCompileInfo compileInfo = {64, 196608};
    gert::StorageShape xShape = {{66561}, {66561}};
    gert::StorageShape countShape = {{1}, {1}};
    gert::StorageShape seedShape = {{1}, {1}};
    gert::StorageShape offsetShape = {{1}, {1}};
    gert::StorageShape yShape = {{1, 1}, {1, 1}};
    gert::StorageShape maskShape = {{1}, {1}};

    vector<int32_t> countValue = {20};
    vector<int64_t> seedValue = {12};
    vector<int64_t> offsetValue = {22};

    gert::TilingContextPara tilingContextPara(
        "StatelessRandomChoiceWithMask",
        {{xShape, ge::DT_BOOL, ge::FORMAT_ND},
         {countShape, ge::DT_INT32, ge::FORMAT_ND, true, countValue.data()},
         {seedShape, ge::DT_INT64, ge::FORMAT_ND, true, seedValue.data()},
         {offsetShape, ge::DT_INT64, ge::FORMAT_ND, true, offsetValue.data()}},
        {{yShape, ge::DT_INT32, ge::FORMAT_ND}, {maskShape, ge::DT_BOOL, ge::FORMAT_ND}},
        &compileInfo);

    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);
    EXPECT_EQ(tilingInfo.tilingKey, 0);
    EXPECT_EQ(tilingInfo.blockNum, 64);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], 17582080);
}
