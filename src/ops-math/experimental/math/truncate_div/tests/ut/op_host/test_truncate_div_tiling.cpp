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
 * \file test_truncate_div_tiling.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "truncate_div_tiling.h"
#include "../../../op_kernel/truncate_div_tiling_data.h"
#include "../../../op_kernel/truncate_div_tiling_key.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace optiling;

class TruncateDivTiling : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "TruncateDivTiling SetUp" << endl; }

    static void TearDownTestCase() { cout << "TruncateDivTiling TearDown " << endl; }
};

TEST_F(TruncateDivTiling, ascend910b_test_tiling_FLOAT16_001)
{
    optiling::TruncateDivCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("TruncateDiv",
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
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<int64_t>(TRUNCATEDIV_TPL_SCH_MODE_1));
}

TEST_F(TruncateDivTiling, ascend910b_test_tiling_INT32_001)
{
    optiling::TruncateDivCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("TruncateDiv",
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
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<int64_t>(TRUNCATEDIV_TPL_SCH_MODE_6));
}
