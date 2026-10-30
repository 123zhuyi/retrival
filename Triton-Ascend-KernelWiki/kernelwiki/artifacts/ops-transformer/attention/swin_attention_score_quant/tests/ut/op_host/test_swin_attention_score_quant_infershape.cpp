/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "array_ops.h"
#include "fusion_ops.h"
#include "elewise_calculation_ops.h"
#include "graph/debug/ge_attr_define.h"
#include "utils/attr_utils.h"
#include "common/utils/ut_op_common.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "util/util.h"
// #include "graph_dsl/graph_dsl.h"
#include "op_proto_test_util.h"

class SwinAttentionScoreQuantTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SwinAttentionScoreQuantTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SwinAttentionScoreQuantTest TearDown" << std::endl;
  }
};

TEST_F(SwinAttentionScoreQuantTest, swin_attention_score_quant_infershape_test_1){
	ge::op::SwinAttentionScoreQuant op;
    uint32_t b = 10;
    uint32_t n = 3;
    uint32_t s = 49;
    uint32_t h = 32;
    op.UpdateInputDesc("query", create_desc_shape_range(
        {-1, -1, -1, -1},
        ge::DT_INT8, ge::FORMAT_ND,
        {b, n, s, h}, ge::FORMAT_ND,
        {{b, b}, {n, n},{s, s}, {h, h}}));
    op.UpdateInputDesc("key", create_desc_shape_range(
        {-1, -1, -1, -1},
        ge::DT_INT8, ge::FORMAT_ND,
        {b, n, s, h}, ge::FORMAT_ND,
        {{b, b}, {n, n},{s, s}, {h, h}}));
    op.UpdateInputDesc("value", create_desc_shape_range(
        {-1, -1, -1, -1},
        ge::DT_INT8, ge::FORMAT_ND,
        {b, n, s, h}, ge::FORMAT_ND,
        {{b, b}, {n, n},{s, s}, {h, h}}));
    op.UpdateInputDesc("scale_quant", create_desc_shape_range(
        {-1, -1},
        ge::DT_FLOAT16, ge::FORMAT_ND,
        {1, s}, ge::FORMAT_ND,
        {{1, 1}, {s, s}}));
    op.UpdateInputDesc("scale_dequant1", create_desc_shape_range(
        {-1, -1},
        ge::DT_UINT64, ge::FORMAT_ND,
        {1, s}, ge::FORMAT_ND,
        {{1, 1}, {s, s}}));
    op.UpdateInputDesc("scale_dequant2", create_desc_shape_range(
        {-1, -1},
        ge::DT_UINT64, ge::FORMAT_ND,
        {1, h}, ge::FORMAT_ND,
        {{1, 1}, {h, h}}));
    op.UpdateInputDesc("bias_quant", create_desc_shape_range(
        {-1, -1},
        ge::DT_FLOAT16, ge::FORMAT_ND,
        {1, s}, ge::FORMAT_ND,
        {{1, 1}, {s, s}}));
    op.UpdateInputDesc("bias_dequant1", create_desc_shape_range(
        {-1, -1},
        ge::DT_INT32, ge::FORMAT_ND,
        {1, s}, ge::FORMAT_ND,
        {{1, 1}, {s, s}}));
    op.UpdateInputDesc("bias_dequant2", create_desc_shape_range(
        {-1, -1},
        ge::DT_INT32, ge::FORMAT_ND,
        {1, h}, ge::FORMAT_ND,
        {{1, 1}, {h, h}}));
    op.UpdateInputDesc("padding_mask1", create_desc_shape_range(
        {-1, -1, -1, -1},
        ge::DT_FLOAT16, ge::FORMAT_ND,
        {1, n, s, s}, ge::FORMAT_ND,
        {{1, 1}, {n, n}, {s, s}, {s, s}}));
    op.UpdateInputDesc("padding_mask2", create_desc_shape_range(
        {-1, -1, -1, -1},
        ge::DT_FLOAT16, ge::FORMAT_ND,
        {1, n, s, s}, ge::FORMAT_ND,
        {{1, 1}, {n, n}, {s, s}, {s, s}}));
	op.SetAttr("query_transpose", false);
	op.SetAttr("key_transpose", false);
	op.SetAttr("value_transpose", false);
	op.SetAttr("softmax_axes", -1);
    ge::graphStatus ret = op.InferShapeAndType();
}