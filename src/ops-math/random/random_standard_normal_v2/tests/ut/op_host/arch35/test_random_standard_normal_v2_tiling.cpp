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
#include "../../../../op_host/arch35/random_standard_normal_v2_tiling_arch35.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class RandomStandardNormalV2Tiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "RandomStandardNormalV2 SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "RandomStandardNormalV2 TearDown" << std::endl;
    }
};

TEST_F(RandomStandardNormalV2Tiling, random_standard_normal_v2_tiling_950_float_001)
{
    optiling::RandomStandardNormalV2CompileInfo compileInfo = {64, 196608};
    gert::StorageShape shapeShape = {{2}, {2}};
    gert::StorageShape offsetShape = {{1}, {1}};
    gert::StorageShape outShape = {{32, 512}, {32, 512}};
    auto seed = Ops::Math::AnyValue::CreateFrom<int64_t>(10);
    auto seed2 = Ops::Math::AnyValue::CreateFrom<int64_t>(5);
    auto dtype = Ops::Math::AnyValue::CreateFrom<int64_t>(0);

    vector<int32_t> shapeValue = {32, 512};
    vector<int64_t> offsetValue = {0};

    gert::TilingContextPara tilingContextPara(
        "RandomStandardNormalV2",
        {{shapeShape, ge::DT_INT32, ge::FORMAT_ND, true, shapeValue.data()},
         {offsetShape, ge::DT_INT64, ge::FORMAT_ND, true, offsetValue.data()}},
        {{outShape, ge::DT_FLOAT, ge::FORMAT_ND}, {offsetShape, ge::DT_INT64, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("dtype", dtype),
         gert::TilingContextPara::OpAttr("seed", seed),
         gert::TilingContextPara::OpAttr("seed2", seed2)},
        &compileInfo);
    uint64_t expectTilingKey = 100;
    string expectTilingData = "64 256 256 10912 10 0 5 16384 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(RandomStandardNormalV2Tiling, random_standard_normal_v2_tiling_950_float_002)
{
    optiling::RandomStandardNormalV2CompileInfo compileInfo = {64, 196608};
    gert::StorageShape shapeShape = {{2}, {2}};
    gert::StorageShape offsetShape = {{1}, {1}};
    gert::StorageShape outShape = {{32, 512}, {32, 512}};
    auto seed = Ops::Math::AnyValue::CreateFrom<int64_t>(0);
    auto seed2 = Ops::Math::AnyValue::CreateFrom<int64_t>(0);
    auto dtype = Ops::Math::AnyValue::CreateFrom<int64_t>(0);

    vector<int32_t> shapeValue = {32, 512};
    vector<int64_t> offsetValue = {0};

    gert::TilingContextPara tilingContextPara(
        "RandomStandardNormalV2",
        {{shapeShape, ge::DT_INT32, ge::FORMAT_ND, true, shapeValue.data()},
         {offsetShape, ge::DT_INT64, ge::FORMAT_ND, true, offsetValue.data()}},
        {{outShape, ge::DT_FLOAT, ge::FORMAT_ND}, {offsetShape, ge::DT_INT64, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("dtype", dtype),
         gert::TilingContextPara::OpAttr("seed", seed),
         gert::TilingContextPara::OpAttr("seed2", seed2)},
        &compileInfo);

    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);
}
