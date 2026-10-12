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
 * \file test_truncate_mod_tiling.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "truncate_mod_tiling.h"
#include "../../../op_kernel/truncate_mod_tiling_data.h"
#include "../../../op_kernel/truncate_mod_tiling_key.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace optiling;

class TruncateModTiling : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "TruncateModTiling SetUp" << endl; }

    static void TearDownTestCase() { cout << "TruncateModTiling TearDown " << endl; }
};

TEST_F(TruncateModTiling, ascend910b_test_tiling_FLOAT16_001)
{
    optiling::TruncateModCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("TruncateMod",
                                              {
                                                  {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    TilingInfo tilingInfo;
    bool ret = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ret);
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<int64_t>(TRUNCATEMOD_TPL_SCH_MODE_0));
}

TEST_F(TruncateModTiling, ascend910b_test_tiling_INT32_001)
{
    optiling::TruncateModCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("TruncateMod",
                                              {
                                                  {{{512, 128}, {512, 128}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{512, 128}, {512, 128}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{512, 128}, {512, 128}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    TilingInfo tilingInfo;
    bool ret = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ret);
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<int64_t>(TRUNCATEMOD_TPL_SCH_MODE_3));
}
