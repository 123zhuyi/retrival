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
#include "conversion/slice_last_dim/op_kernel/arch35/slice_last_dim_tiling_data.h"

class SliceLastDimTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SliceLastDimTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SliceLastDimTiling TearDown" << std::endl;
    }
};

TEST_F(SliceLastDimTilingTest, slice_last_dim_tiling_basic)
{
    SliceLastDimCompileInfo compileInfo;
    gert::StorageShape inShape = {{4, 16}, {4, 16}};
    gert::StorageShape outShape = {{4, 8}, {4, 8}};
    gert::TilingContextPara tilingContextPara(
        "SliceLastDim", {{inShape, ge::DT_FLOAT16, ge::FORMAT_ND}}, {{outShape, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"start", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"end", Ops::Math::AnyValue::CreateFrom<int64_t>(8)},
            {"stride", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
        },
        &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    ASSERT_GT(tilingInfo.blockNum, 0);
}

TEST_F(SliceLastDimTilingTest, slice_last_dim_tiling_with_stride)
{
    SliceLastDimCompileInfo compileInfo;
    gert::StorageShape inShape = {{2, 3, 100}, {2, 3, 100}};
    gert::StorageShape outShape = {{2, 3, 50}, {2, 3, 50}};
    gert::TilingContextPara tilingContextPara(
        "SliceLastDim", {{inShape, ge::DT_FLOAT, ge::FORMAT_ND}}, {{outShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {"start", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"end", Ops::Math::AnyValue::CreateFrom<int64_t>(100)},
            {"stride", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
        },
        &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    ASSERT_GT(tilingInfo.blockNum, 0);
}
