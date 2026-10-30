/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "experiment_ops.h"
#include "op_proto_test_util.h"
#include "graph/operator_factory_impl.h"
#include "common/utils/ut_op_common.h"
#include "graph/utils/op_desc_utils.h"

class QuantGroupedMatmulDequantTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "QuantGroupedMatmulDequantTest Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "QuantGroupedMatmulDequantTest Proto Test TearDown" << std::endl;
  }
};

TEST_F(QuantGroupedMatmulDequantTest, test_float16_1){
  ge::op::QuantMatmulDequant op;
  int64_t G=16;
  int64_t M=256;
  int64_t K=256;
  int64_t N=256;

  op.UpdateInputDesc("x", create_desc({M,K}, ge::DT_FLOAT16));
  op.UpdateInputDesc("quantized_weight", create_desc({G,K/32,N/16,16,32}, ge::DT_INT8));
  op.UpdateInputDesc("weight_scale", create_desc({G,N}, ge::DT_FLOAT));
  op.UpdateInputDesc("group_list", create_desc({G}, ge::DT_INT64));
  op.UpdateInputDesc("bias", create_desc({N}, ge::DT_INT32));
  op.UpdateInputDesc("x_scale", create_desc({M}, ge::DT_FLOAT));
  op.UpdateInputDesc("x_offset", create_desc({M}, ge::DT_FLOAT));
  op.UpdateInputDesc("smooth_scale", create_desc({K}, ge::DT_FLOAT16));
  op.SetAttr("x_quant_mode", "pertoken");
  op.SetAttr("transpose_weight", true);

  Runtime2TestParam param{{"x_quant_mode","transpose_weight"}, {}, {}};
  auto ret = InferShapeTest(op, param);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  std::vector<int64_t> expected_output_shape = {M,N};
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);

  ret = InferDataTypeTest(op, param);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
}
