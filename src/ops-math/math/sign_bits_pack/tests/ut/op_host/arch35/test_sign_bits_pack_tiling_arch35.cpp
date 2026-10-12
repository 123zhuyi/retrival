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

using namespace std;
using namespace ge;

namespace optiling {
struct SignBitsPackCompileInfo {
    uint64_t coreNum;
    uint64_t ubSize;
};
} // namespace optiling

class SignBitsPackTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SignBitsPackTilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "SignBitsPackTilingTest TearDown" << std::endl; }
};

// 正常1D输入 N=256 fp32 size=1
TEST_F(SignBitsPackTilingTest, test_tiling_normal_256_fp32_size1)
{
    optiling::SignBitsPackCompileInfo compileInfo;
    compileInfo.coreNum = 8;
    compileInfo.ubSize = 256 * 1024;
    gert::TilingContextPara tilingContextPara("SignBitsPack",
                                              {
                                                  {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 32}, {1, 32}}, ge::DT_UINT8, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"size", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 1;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, EMPTY_EXPECT_TILING_DATA);
}

// 正常1D输入 N=512 fp16 size=8
TEST_F(SignBitsPackTilingTest, test_tiling_normal_512_fp16_size8)
{
    optiling::SignBitsPackCompileInfo compileInfo;
    compileInfo.coreNum = 8;
    compileInfo.ubSize = 256 * 1024;
    gert::TilingContextPara tilingContextPara("SignBitsPack",
                                              {
                                                  {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 8}, {8, 8}}, ge::DT_UINT8, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"size", Ops::Math::AnyValue::CreateFrom<int64_t>(8)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 1;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, EMPTY_EXPECT_TILING_DATA);
}

// 空Tensor N=0
TEST_F(SignBitsPackTilingTest, test_tiling_empty_tensor)
{
    optiling::SignBitsPackCompileInfo compileInfo;
    compileInfo.coreNum = 8;
    compileInfo.ubSize = 256 * 1024;
    gert::TilingContextPara tilingContextPara("SignBitsPack",
                                              {
                                                  {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 0}, {1, 0}}, ge::DT_UINT8, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"size", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, EMPTY_EXPECT_TILING_DATA);
}

// 非8倍数 N=17 fp32 size=1 (填充场景)
TEST_F(SignBitsPackTilingTest, test_tiling_non8_pad)
{
    optiling::SignBitsPackCompileInfo compileInfo;
    compileInfo.coreNum = 8;
    compileInfo.ubSize = 256 * 1024;
    gert::TilingContextPara tilingContextPara("SignBitsPack",
                                              {
                                                  {{{17}, {17}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 3}, {1, 3}}, ge::DT_UINT8, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"size", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 1;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, EMPTY_EXPECT_TILING_DATA);
}
