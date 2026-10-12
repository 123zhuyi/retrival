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
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "conversion/slice_with_axes/op_kernel/arch35/slice_with_axes_tiling_data.h"

class SliceWithAxesTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SliceWithAxesTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SliceWithAxesTiling TearDown" << std::endl;
    }
};

TEST_F(SliceWithAxesTilingTest, slice_with_axes_tiling_basic)
{
    SliceWithAxesCompileInfo compileInfo;
    int32_t offsets[] = {2};
    int32_t sizes[] = {5};
    gert::StorageShape inShape = {{10}, {10}};
    gert::StorageShape outShape = {{5}, {5}};
    gert::StorageShape offsetShape = {{1}, {1}};
    gert::StorageShape sizeShape = {{1}, {1}};
    gert::TilingContextPara tilingContextPara(
        "SliceWithAxes",
        {
            {inShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {offsetShape, ge::DT_INT32, ge::FORMAT_ND, true, offsets},
            {sizeShape, ge::DT_INT32, ge::FORMAT_ND, true, sizes},
        },
        {{outShape, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"axes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
        },
        &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    ASSERT_GT(tilingInfo.blockNum, 0);
}

TEST_F(SliceWithAxesTilingTest, slice_with_axes_tiling_multi_axes)
{
    SliceWithAxesCompileInfo compileInfo;
    int32_t offsets[] = {1, 5};
    int32_t sizes[] = {5, 10};
    gert::StorageShape inShape = {{10, 20, 30}, {10, 20, 30}};
    gert::StorageShape outShape = {{5, 20, 10}, {5, 20, 10}};
    gert::StorageShape offsetShape = {{2}, {2}};
    gert::StorageShape sizeShape = {{2}, {2}};
    gert::TilingContextPara tilingContextPara(
        "SliceWithAxes",
        {
            {inShape, ge::DT_INT32, ge::FORMAT_ND},
            {offsetShape, ge::DT_INT32, ge::FORMAT_ND, true, offsets},
            {sizeShape, ge::DT_INT32, ge::FORMAT_ND, true, sizes},
        },
        {{outShape, ge::DT_INT32, ge::FORMAT_ND}},
        {
            {"axes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0, 2})},
        },
        &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    ASSERT_GT(tilingInfo.blockNum, 0);
}
