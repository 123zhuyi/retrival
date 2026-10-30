/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/add_mat_mat_elements_tiling_arch35.h"

using namespace std;
using namespace ge;
using namespace optiling;

namespace optiling {
struct AddMatMatElementsCompileInfo {};
}

// 输入顺序与 op_def / proto.h 一致：c, a, b, beta, alpha
// beta/alpha 为 1-element 标量 tensor
class AddMatMatElementsTilingTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "AddMatMatElementsTilingTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "AddMatMatElementsTilingTest TearDown" << std::endl;
  }
};

TEST_F(AddMatMatElementsTilingTest, add_mat_mat_elements_tiling_fp32_basic) {
  gert::StorageShape shape = {{32, 32}, {32, 32}};
  gert::StorageShape scalarShape = {{1}, {1}};

  AddMatMatElementsCompileInfo compileInfo;

  gert::TilingContextPara tilingContextPara(
      "AddMatMatElements",
      {{shape, ge::DT_FLOAT, ge::FORMAT_ND},        // c
       {shape, ge::DT_FLOAT, ge::FORMAT_ND},        // a
       {shape, ge::DT_FLOAT, ge::FORMAT_ND},        // b
       {scalarShape, ge::DT_FLOAT, ge::FORMAT_ND},  // beta
       {scalarShape, ge::DT_FLOAT, ge::FORMAT_ND}}, // alpha
      {{shape, ge::DT_FLOAT, ge::FORMAT_ND}},
      {},
      &compileInfo);

  TilingInfo tilingInfo;
  EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

TEST_F(AddMatMatElementsTilingTest, add_mat_mat_elements_tiling_fp16_basic) {
  gert::StorageShape shape = {{16, 64}, {16, 64}};
  gert::StorageShape scalarShape = {{1}, {1}};

  AddMatMatElementsCompileInfo compileInfo;

  gert::TilingContextPara tilingContextPara(
      "AddMatMatElements",
      {{shape, ge::DT_FLOAT16, ge::FORMAT_ND},
       {shape, ge::DT_FLOAT16, ge::FORMAT_ND},
       {shape, ge::DT_FLOAT16, ge::FORMAT_ND},
       {scalarShape, ge::DT_FLOAT16, ge::FORMAT_ND},
       {scalarShape, ge::DT_FLOAT16, ge::FORMAT_ND}},
      {{shape, ge::DT_FLOAT16, ge::FORMAT_ND}},
      {},
      &compileInfo);

  TilingInfo tilingInfo;
  EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

TEST_F(AddMatMatElementsTilingTest, add_mat_mat_elements_tiling_bf16_basic) {
  gert::StorageShape shape = {{8, 128}, {8, 128}};
  gert::StorageShape scalarShape = {{1}, {1}};

  AddMatMatElementsCompileInfo compileInfo;

  gert::TilingContextPara tilingContextPara(
      "AddMatMatElements",
      {{shape, ge::DT_BF16, ge::FORMAT_ND},
       {shape, ge::DT_BF16, ge::FORMAT_ND},
       {shape, ge::DT_BF16, ge::FORMAT_ND},
       {scalarShape, ge::DT_BF16, ge::FORMAT_ND},
       {scalarShape, ge::DT_BF16, ge::FORMAT_ND}},
      {{shape, ge::DT_BF16, ge::FORMAT_ND}},
      {},
      &compileInfo);

  TilingInfo tilingInfo;
  EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

TEST_F(AddMatMatElementsTilingTest, add_mat_mat_elements_tiling_fp32_large) {
  gert::StorageShape shape = {{1024, 1024}, {1024, 1024}};
  gert::StorageShape scalarShape = {{1}, {1}};

  AddMatMatElementsCompileInfo compileInfo;

  gert::TilingContextPara tilingContextPara(
      "AddMatMatElements",
      {{shape, ge::DT_FLOAT, ge::FORMAT_ND},
       {shape, ge::DT_FLOAT, ge::FORMAT_ND},
       {shape, ge::DT_FLOAT, ge::FORMAT_ND},
       {scalarShape, ge::DT_FLOAT, ge::FORMAT_ND},
       {scalarShape, ge::DT_FLOAT, ge::FORMAT_ND}},
      {{shape, ge::DT_FLOAT, ge::FORMAT_ND}},
      {},
      &compileInfo);

  TilingInfo tilingInfo;
  EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}
