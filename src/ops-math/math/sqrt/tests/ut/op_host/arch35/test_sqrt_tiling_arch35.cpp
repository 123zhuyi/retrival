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
 * \file test_lerp_tiling_arch35.cpp
 * \brief
 */

#include "../../../../op_host/arch35/sqrt_tiling_arch35.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

class SqrtTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SqrtTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SqrtTiling TearDown" << std::endl;
  }
};

TEST_F(SqrtTiling, test_tiling_failed_bf16_fp32_004) {
    optiling::SqrtCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("Sqrt",
                                              {
                                                {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(SqrtTiling, test_tiling_failed_empty_tensor_005) {
    optiling::SqrtCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("Sqrt",
                                              {
                                                {{{1, 0, 2, 64}, {1, 0, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 0, 2, 64}, {1, 0, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(SqrtTiling, test_tiling_failed_unsupport_type_006) {
    optiling::SqrtCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("Sqrt",
                                              {
                                                {{{1, 1, 2, 64}, {1, 1, 2, 64}}, ge::DT_DOUBLE, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 1, 2, 64}, {1, 1, 2, 64}}, ge::DT_DOUBLE, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}