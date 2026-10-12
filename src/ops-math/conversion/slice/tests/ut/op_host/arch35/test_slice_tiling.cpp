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
 * \file test_slice_tiling.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/slice_tiling_arch35.h"

using namespace std;
using namespace ge;

class SliceTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SliceTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SliceTiling TearDown" << std::endl;
  }
};

TEST_F(SliceTiling, slice_test_tiling_001)
{
    vector<int64_t> offsetsValue = {7};
    vector<int64_t> sizeValue = {1};
    optiling::SliceCompileParam compileInfo = {32, 253952, 10, false};
    gert::TilingContextPara tilingContextPara(
        "Slice",
        {
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, offsetsValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, sizeValue.data()},
        },
        {
            {{{1}, {1}}, ge::DT_INT8, ge::FORMAT_ND},
        },
         &compileInfo);
    uint64_t expectTilingKey = 103;
    string expectTilingData = "1 1 0 0 4295221248 0 65537 1 0 0 0 0 0 0 0 7 "
                      "0 0 0 0 0 0 0 8 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}