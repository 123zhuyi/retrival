/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conversion/im2col/op_kernel/arch35/im2col_tilingdata.h"
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

class Im2colTilingNHWCTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "Im2colTilingNHWCTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "Im2colTilingNHWCTest TearDown" << std::endl;
    }
};

namespace {
Im2ColCompileInfo compileInfo;
}

// // 全载
// TEST_F(Im2colTilingNHWCTest, Im2colTilingNHWCTest1)
// {
//     gert::TilingContextPara tilingContextPara(
//         "Im2col",
//         {
//             {{{1, 2, 3, 128}, {1, 2, 3, 128}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
//         },
//         {
//             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
//         },
//         {
//             {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2})},
//             {"strides", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
//             {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2})},
//             {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>("VALID")},
//         },
//         &compileInfo);
//     uint64_t expectTilingKey = 0b00000000'0'0'0'00000001;
//     std::vector<size_t> expectWorkspaces = {0};
//     ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
// }

// // 切W轴
// TEST_F(Im2colTilingNHWCTest, Im2colTilingNHWCTest2)
// {
//     gert::TilingContextPara tilingContextPara(
//         "Im2col",
//         {
//             {{{1, 2, 3, 8192}, {1, 2, 3, 8192}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
//         },
//         {
//             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
//         },
//         {
//             {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2})},
//             {"strides", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
//             {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2})},
//             {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>("SAME")},
//             {"pads", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
//         },
//         &compileInfo);
//     uint64_t expectTilingKey = 0b00000010'0'1'0'00000001;
//     std::vector<size_t> expectWorkspaces = {0};
//     ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
// }

// // 切w轴 pad
// TEST_F(Im2colTilingNHWCTest, Im2colTilingNHWCTest3)
// {
//     gert::TilingContextPara tilingContextPara(
//         "Im2col",
//         {
//             {{{1, 2, 1, 8192}, {1, 2, 1, 8192}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
//         },
//         {
//             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
//         },
//         {
//             {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2})},
//             {"strides", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
//             {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2})},
//             {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>("SAME")},
//             {"pads", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
//         },
//         &compileInfo);
//     uint64_t expectTilingKey = 0b00000010'0'1'0'00000001;
//     std::vector<size_t> expectWorkspaces = {0};
//     ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
// }

// // 切w轴 pad 尾轴对齐
// TEST_F(Im2colTilingNHWCTest, Im2colTilingNHWCTest4)
// {
//     gert::TilingContextPara tilingContextPara(
//         "Im2col",
//         {
//             {{{1, 2, 1, 8191}, {1, 2, 1, 8191}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
//         },
//         {
//             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
//         },
//         {
//             {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2})},
//             {"strides", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
//             {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2})},
//             {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>("SAME")},
//             {"pads", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
//         },
//         &compileInfo);

//     TilingInfo tilingInfo;
//     auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
//     ASSERT_TRUE(tilingRet);
//     EXPECT_EQ(tilingInfo.tilingKey, 0b00000010'0'1'0'00000001);
//     ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(Im2ColNHWCTilingData));
//     Im2ColNHWCTilingData* data = reinterpret_cast<Im2ColNHWCTilingData*>(tilingInfo.tilingData.get());
//     EXPECT_EQ(data->ubFactorN, 1);
//     EXPECT_EQ(data->ubFactorH, 1);
//     EXPECT_EQ(data->ubFactorW, 2);
//     EXPECT_EQ(data->ubFactorC, 8192);
//     EXPECT_EQ(data->convKernelNumInHeight, 2);
//     EXPECT_EQ(data->convKernelNumInWidth, 1);
//     EXPECT_EQ(data->totalLines, 4);
// }

// // 切w轴 pad 尾轴对齐
// TEST_F(Im2colTilingNHWCTest, Im2colTilingNHWCTest5)
// {
//     gert::TilingContextPara tilingContextPara(
//         "Im2col",
//         {
//             {{{1, 2, 1, 8193}, {1, 2, 1, 8193}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
//         },
//         {
//             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
//         },
//         {
//             {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2})},
//             {"strides", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
//             {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2})},
//             {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>("SAME")},
//             {"pads", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
//         },
//         &compileInfo);

//     TilingInfo tilingInfo;
//     auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
//     ASSERT_TRUE(tilingRet);
//     EXPECT_EQ(tilingInfo.tilingKey, 0b00000010'0'1'0'00000001);
//     ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(Im2ColNHWCTilingData));
//     Im2ColNHWCTilingData* data = reinterpret_cast<Im2ColNHWCTilingData*>(tilingInfo.tilingData.get());
//     EXPECT_EQ(data->ubFactorN, 1);
//     EXPECT_EQ(data->ubFactorH, 1);
//     EXPECT_EQ(data->ubFactorW, 1);
//     EXPECT_EQ(data->ubFactorC, 8200);
//     EXPECT_EQ(data->convKernelNumInHeight, 2);
//     EXPECT_EQ(data->convKernelNumInWidth, 1);
//     EXPECT_EQ(data->totalLines, 8);
// }

// // 切c轴 pad 尾轴对齐
// TEST_F(Im2colTilingNHWCTest, Im2colTilingNHWCTest6)
// {
//     gert::TilingContextPara tilingContextPara(
//         "Im2col",
//         {
//             {{{1, 2, 1, 21203}, {1, 2, 1, 21203}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
//         },
//         {
//             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
//         },
//         {
//             {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2})},
//             {"strides", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
//             {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2})},
//             {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>("SAME")},
//             {"pads", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
//         },
//         &compileInfo);

//     TilingInfo tilingInfo;
//     auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
//     ASSERT_TRUE(tilingRet);
//     EXPECT_EQ(tilingInfo.tilingKey, 0b00000011'0'1'0'00000001);
//     ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(Im2ColNHWCTilingData));
//     Im2ColNHWCTilingData* data = reinterpret_cast<Im2ColNHWCTilingData*>(tilingInfo.tilingData.get());
//     EXPECT_EQ(data->ubFactorN, 1);
//     EXPECT_EQ(data->ubFactorH, 1);
//     EXPECT_EQ(data->ubFactorW, 1);
//     EXPECT_EQ(data->ubFactorC, 16384);
//     EXPECT_EQ(data->convKernelNumInHeight, 2);
//     EXPECT_EQ(data->convKernelNumInWidth, 1);
//     EXPECT_EQ(data->totalLines, 16);
// }

// // 切h轴 pad 尾轴对齐
// TEST_F(Im2colTilingNHWCTest, Im2colTilingNHWCTest7)
// {
//     gert::TilingContextPara tilingContextPara(
//         "Im2col",
//         {
//             {{{1, 3, 6, 1025}, {1, 3, 6, 1025}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
//         },
//         {
//             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
//         },
//         {
//             {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2})},
//             {"strides", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
//             {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2})},
//             {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>("SAME")},
//             {"pads", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
//         },
//         &compileInfo);

//     TilingInfo tilingInfo;
//     auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
//     ASSERT_TRUE(tilingRet);
//     EXPECT_EQ(tilingInfo.tilingKey, 0b00000001'0'1'0'00000001);
//     ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(Im2ColNHWCTilingData));
//     Im2ColNHWCTilingData* data = reinterpret_cast<Im2ColNHWCTilingData*>(tilingInfo.tilingData.get());
//     EXPECT_EQ(data->ubFactorN, 1);
//     EXPECT_EQ(data->ubFactorH, 3);
//     EXPECT_EQ(data->ubFactorW, 4);
//     EXPECT_EQ(data->ubFactorC, 1032);
//     EXPECT_EQ(data->convKernelNumInHeight, 3);
//     EXPECT_EQ(data->convKernelNumInWidth, 6);
//     EXPECT_EQ(data->totalLines, 6);
// }

// // 切w轴 pad 尾轴对齐
// TEST_F(Im2colTilingNHWCTest, Im2colTilingNHWCTest8)
// {
//     gert::TilingContextPara tilingContextPara(
//         "Im2col",
//         {
//             {{{1, 3, 6, 3200}, {1, 3, 6, 3200}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
//         },
//         {
//             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
//         },
//         {
//             {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({3, 2})},
//             {"strides", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
//             {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2})},
//             {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>("SAME")},
//             {"pads", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
//         },
//         &compileInfo);

//     TilingInfo tilingInfo;
//     auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
//     ASSERT_TRUE(tilingRet);
//     EXPECT_EQ(tilingInfo.tilingKey, 0b00000010'0'1'0'00000001);
//     ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(Im2ColNHWCTilingData));
//     Im2ColNHWCTilingData* data = reinterpret_cast<Im2ColNHWCTilingData*>(tilingInfo.tilingData.get());
//     EXPECT_EQ(data->ubFactorN, 1);
//     EXPECT_EQ(data->ubFactorH, 1);
//     EXPECT_EQ(data->ubFactorW, 4);
//     EXPECT_EQ(data->ubFactorC, 3200);
//     EXPECT_EQ(data->convKernelNumInHeight, 3);
//     EXPECT_EQ(data->convKernelNumInWidth, 6);
//     EXPECT_EQ(data->totalLines, 36);
// }

// // 切w轴 pad 尾轴对齐
// TEST_F(Im2colTilingNHWCTest, Im2colTilingNHWCTest9)
// {
//     gert::TilingContextPara tilingContextPara(
//         "Im2col",
//         {
//             {{{1, 3, 6, 3200}, {1, 3, 6, 3200}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
//         },
//         {
//             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
//         },
//         {
//             {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({3, 5})},
//             {"strides", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
//             {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
//             {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>("SAME")},
//             {"pads", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
//         },
//         &compileInfo);

//     TilingInfo tilingInfo;
//     auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
//     ASSERT_TRUE(tilingRet);
//     EXPECT_EQ(tilingInfo.tilingKey, 0b00000010'0'1'0'00000001);
//     ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(Im2ColNHWCTilingData));
//     Im2ColNHWCTilingData* data = reinterpret_cast<Im2ColNHWCTilingData*>(tilingInfo.tilingData.get());
//     EXPECT_EQ(data->ubFactorN, 1);
//     EXPECT_EQ(data->ubFactorH, 1);
//     EXPECT_EQ(data->ubFactorW, 5);
//     EXPECT_EQ(data->ubFactorC, 3200);
//     EXPECT_EQ(data->convKernelNumInHeight, 3);
//     EXPECT_EQ(data->convKernelNumInWidth, 6);
//     EXPECT_EQ(data->totalLines, 54);
// }

// // 切w轴 kh*(w) 尾轴不对齐
// TEST_F(Im2colTilingNHWCTest, Im2colTilingNHWCTest10)
// {
//     gert::TilingContextPara tilingContextPara(
//         "Im2col",
//         {
//             {{{1, 3, 6, 3284}, {1, 3, 6, 3284}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
//         },
//         {
//             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
//         },
//         {
//             {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({3, 5})},
//             {"strides", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
//             {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
//             {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>("SAME")},
//             {"pads", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
//         },
//         &compileInfo);

//     TilingInfo tilingInfo;
//     auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
//     ASSERT_TRUE(tilingRet);
//     EXPECT_EQ(tilingInfo.tilingKey, 0b00000010'0'1'0'00000001);
//     ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(Im2ColNHWCTilingData));
//     Im2ColNHWCTilingData* data = reinterpret_cast<Im2ColNHWCTilingData*>(tilingInfo.tilingData.get());
//     EXPECT_EQ(data->ubFactorN, 1);
//     EXPECT_EQ(data->ubFactorH, 1);
//     EXPECT_EQ(data->ubFactorW, 4);
//     EXPECT_EQ(data->ubFactorC, 3288);
//     EXPECT_EQ(data->convKernelNumInHeight, 3);
//     EXPECT_EQ(data->convKernelNumInWidth, 6);
//     EXPECT_EQ(data->totalLines, 108);
// }
