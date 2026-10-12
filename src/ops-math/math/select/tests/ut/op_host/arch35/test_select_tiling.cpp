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
 * \file test_select_tiling.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/select_tiling.h"

using namespace std;
using namespace ge;

class SelectTiling : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "SelectTiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "SelectTiling TearDown" << std::endl;
    }
};

TEST_F(SelectTiling, select_test_0) {
    optiling::SelectCompileInfo compileInfo = {64, 245760};
    gert::StorageShape conditionShape = {{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}};
    gert::StorageShape x1Shape = {{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}};
    gert::StorageShape x2Shape = {{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}};
    gert::StorageShape yShape = {{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}};
    gert::TilingContextPara::TensorDescription condition(conditionShape, ge::DT_BOOL, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription x1(x1Shape, ge::DT_BF16, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription x2(x2Shape, ge::DT_BF16, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara(
        "Select",
        {condition, x1, x2},
        {{ yShape, ge::DT_BF16, ge::FORMAT_ND }},
        &compileInfo);
    uint64_t expectedTilingKey = 8;
    std::vector<size_t> expectedWorkspaces = { 16777216 };
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}


TEST_F(SelectTiling, select_test_1) {
   optiling::SelectCompileInfo compileInfo = {64, 245760};
   gert::StorageShape conditionShape = {{8, 11, 12, 14, 6}, {8, 11, 12, 14, 6}};
   gert::StorageShape x1Shape = {{8, 11, 12, 14, 6}, {8, 11, 12, 14, 6}};
   gert::StorageShape x2Shape = {{8, 11, 12, 14, 6}, {8, 11, 12, 14, 6}};
   gert::StorageShape yShape = {{8, 11, 12, 14, 6}, {8, 11, 12, 14, 6}};
   gert::TilingContextPara::TensorDescription condition(conditionShape, ge::DT_BOOL, ge::FORMAT_ND);
   gert::TilingContextPara::TensorDescription x1(x1Shape, ge::DT_UINT8, ge::FORMAT_ND);
   gert::TilingContextPara::TensorDescription x2(x2Shape, ge::DT_UINT8, ge::FORMAT_ND);
   gert::TilingContextPara tilingContextPara(
       "Select",
       {condition, x1, x2},
       {{ yShape, ge::DT_UINT8, ge::FORMAT_ND }},
       &compileInfo);
   uint64_t expectedTilingKey = 8;
   std::vector<size_t> expectedWorkspaces = { 16777216 };
   ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}
 
TEST_F(SelectTiling, select_test_2) {
   optiling::SelectCompileInfo compileInfo = {64, 245760};
   gert::StorageShape conditionShape = {{20, 5, 1, 9, 17, 10, 3}, {20, 5, 1, 9, 17, 10, 3}};
   gert::StorageShape x1Shape = {{20, 5, 1, 9, 17, 10, 3}, {20, 5, 1, 9, 17, 10, 3}};
   gert::StorageShape x2Shape = {{20, 5, 1, 9, 17, 10, 3}, {20, 5, 1, 9, 17, 10, 3}};
   gert::StorageShape yShape = {{20, 5, 1, 9, 17, 10, 3}, {20, 5, 1, 9, 17, 10, 3}};
   gert::TilingContextPara::TensorDescription condition(conditionShape, ge::DT_BOOL, ge::FORMAT_ND);
   gert::TilingContextPara::TensorDescription x1(x1Shape, ge::DT_FLOAT16, ge::FORMAT_ND);
   gert::TilingContextPara::TensorDescription x2(x2Shape, ge::DT_FLOAT16, ge::FORMAT_ND);
   gert::TilingContextPara tilingContextPara(
       "Select",
       {condition, x1, x2},
       {{ yShape, ge::DT_FLOAT16, ge::FORMAT_ND }},
       &compileInfo);
   uint64_t expectedTilingKey = 8;
   std::vector<size_t> expectedWorkspaces = { 16777216 };
   ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}