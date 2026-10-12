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

namespace {
Im2ColCompileInfo compileInfo;
}

class Im2colTilingNCHWTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "Im2colTilingNCHWTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "Im2colTilingNCHWTest TearDown" << std::endl;
    }

    template <typename T>
    inline T AlignBlockSize(T size)
    {
        return (size + 31) / 32 * 32;
    }

    void CheckFullLoadBuf(Im2ColNCHWTilingData* data, ge::DataType dataType)
    {
        int dSize = ge::GetSizeByDataType(dataType);
        // 检查输入buf是否够用
        int64_t needInputBufSize = AlignBlockSize(data->w4ubFactorW * dSize) * data->ubFactorNC;
        EXPECT_GE(data->inputBufferSize, needInputBufSize);
        // 检查输出buf是否够用
        int64_t needOutputBufSize = data->ubFactorH * data->ubFactorW * data->ubFactorNC * dSize;
        EXPECT_GE(data->outputBufferSize, needOutputBufSize);
        // 检查输入元素数量是否不超过64K
        int64_t inputElementCount = (data->input.H + data->input.hPaddingBefore + data->input.hPaddingAfter) *
                                    (data->input.W + data->input.wPaddingBefore + data->input.wPaddingAfter) *
                                    data->ubFactorNC;
        EXPECT_LE(inputElementCount, 64 * 1024);
        // 检查输出元素数量是否不超过64K
        int64_t outputElementCount = data->ubFactorH * data->ubFactorW * data->ubFactorNC;
        EXPECT_LE(outputElementCount, 64 * 1024);
    }

    void CheckUnFullLoadBuf(Im2ColNCHWTilingData* data, ge::DataType dataType)
    {
        int dSize = ge::GetSizeByDataType(dataType);
        // 检查输入buf是否够用
        // 没有左pad时的最大输入buf
        int64_t maxInputBufSize =
            AlignBlockSize(data->w4ubFactorW * dSize) * data->lines4ubFactorW * data->lines4ubFactorH;
        std::cout << "CheckUnFullLoadBuf, nopad needInputBufSize:" << maxInputBufSize << std::endl;
        // 计算有左pad场景的最大输入buff
        int64_t inputWRectOffset = data->ubFactorW * data->input.wStride;
        for (int64_t start = 0; start < data->input.wPaddingBefore; start += inputWRectOffset) {
            if (start + data->w4ubFactorW <= data->input.wPaddingBefore) {
                continue;
            }
            int64_t padPart = data->input.wPaddingBefore - start;
            int64_t restPart = data->w4ubFactorW - padPart;
            int64_t needInputBufSize = (AlignBlockSize(padPart * dSize) + AlignBlockSize(restPart * dSize)) *
                                       data->lines4ubFactorW * data->lines4ubFactorH;
            std::cout << "CheckUnFullLoadBuf, input w interval=[" << start << ", " << start + data->w4ubFactorW << "]"
                      << ", padPart=" << padPart << ", restPart=" << restPart
                      << ", needInputBufSize:" << needInputBufSize << std::endl;
            maxInputBufSize = std::max(maxInputBufSize, needInputBufSize);
        }
        std::cout << "CheckUnFullLoadBuf, max needInputBufSize:" << maxInputBufSize << std::endl;
        EXPECT_GE(data->inputBufferSize, maxInputBufSize);
        // 检查输出buf是否够用
        int64_t needOutputBufSize = data->ubFactorH * data->ubFactorW * data->ubFactorNC * dSize;
        std::cout << "CheckUnFullLoadBuf, needOutputBufSize:" << needOutputBufSize << std::endl;
        EXPECT_GE(data->outputBufferSize, needOutputBufSize);
        // 检查输入元素数量是否不超过64K
        int64_t inputElementCount = data->w4ubFactorW * data->lines4ubFactorW * data->lines4ubFactorH;
        EXPECT_LE(inputElementCount, 64 * 1024);
        // 检查输出元素数量是否不超过64K
        int64_t outputElementCount = data->ubFactorH * data->ubFactorW * data->ubFactorNC;
        EXPECT_LE(outputElementCount, 64 * 1024);
    }

    void TestUnFullLoadNopadBufSize(
        const std::initializer_list<int64_t>& x, ge::DataType dataType, const std::initializer_list<int64_t>& ksize,
        const std::initializer_list<int64_t>& strides, const std::initializer_list<int64_t>& dilations)
    {
        gert::TilingContextPara tilingContextPara(
            "Im2col",
            {
                {{x, x}, dataType, ge::FORMAT_NCHW},
            },
            {
                {{{}, {}}, dataType, ge::FORMAT_NCHW},
            },
            {
                {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>(ksize)},
                {"strides", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
                {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>(dilations)},
                {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>("VALID")},
                {"pads", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
            },
            &compileInfo);
        TilingInfo tilingInfo;
        auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
        ASSERT_TRUE(tilingRet);
        EXPECT_EQ(tilingInfo.tilingKey & 0b0'11111111'11111111, 0b00000001'00000000);
        ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(Im2ColNCHWTilingData));
        Im2ColNCHWTilingData* data = reinterpret_cast<Im2ColNCHWTilingData*>(tilingInfo.tilingData.get());
        EXPECT_EQ(data->ubFactorNC, 1);
        CheckUnFullLoadBuf(data, dataType);
    }
};

TEST_F(Im2colTilingNCHWTest, Im2colTilingNCHWTest_nopad_small_cutnc)
{
    gert::TilingContextPara tilingContextPara(
        "Im2col",
        {
            {{{100, 2, 3, 5}, {100, 2, 3, 5}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2})},
            {"strides", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2})},
            {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>("CALCULATED")},
            {"pads", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
        },
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
    std::vector<int64_t> expectWorkspaces = {0};
    EXPECT_EQ(tilingInfo.workspaceSizes, expectWorkspaces);
    EXPECT_EQ(tilingInfo.tilingKey, 0b0'0'0'00000000'00000000);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(Im2ColNCHWTilingData));
    Im2ColNCHWTilingData* data = reinterpret_cast<Im2ColNCHWTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(data->ubFactorH, 4);
    EXPECT_EQ(data->ubFactorW, 6);
    EXPECT_EQ(data->ubFactorNC, 200);
    EXPECT_EQ(data->w4ubFactorW, 15);
    EXPECT_EQ(data->convKernelNumInWidth, 3);
    EXPECT_EQ(data->convKernelNumInHeight, 2);
    EXPECT_EQ(data->totalRectAngles, 1);
    EXPECT_EQ(data->rectAnglesPerCore, 1);
    EXPECT_EQ(tilingInfo.blockNum, 1);
    CheckFullLoadBuf(data, ge::DT_FLOAT);
}

TEST_F(Im2colTilingNCHWTest, Im2colTilingNCHWTest_nopad_bigH_cutnc)
{
    gert::TilingContextPara tilingContextPara(
        "Im2col",
        {
            {{{10, 1, 620, 1}, {10, 1, 620, 1}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({6, 1})},
            {"strides", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})},
            {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({8})},
            {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>("CALCULATED")},
            {"pads", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
        },
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
    EXPECT_EQ(tilingInfo.tilingKey, 0b0'0'0'00000000'00000000);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(Im2ColNCHWTilingData));
    Im2ColNCHWTilingData* data = reinterpret_cast<Im2ColNCHWTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(data->ubFactorH, 6);
    EXPECT_EQ(data->ubFactorW, 580);
    EXPECT_EQ(data->ubFactorNC, 7);
    EXPECT_EQ(data->w4ubFactorW, 620);
    EXPECT_EQ(data->convKernelNumInWidth, 1);
    EXPECT_EQ(data->convKernelNumInHeight, 580);
    EXPECT_EQ(data->totalRectAngles, 2);
    EXPECT_EQ(data->rectAnglesPerCore, 1);
    EXPECT_EQ(tilingInfo.blockNum, 2);
    CheckFullLoadBuf(data, ge::DT_FLOAT);
}

TEST_F(Im2colTilingNCHWTest, Im2colTilingNCHWTest_nopad_bigw_cuthw)
{
    gert::TilingContextPara tilingContextPara(
        "Im2col",
        {
            {{{56, 1, 66, 1901}, {56, 1, 66, 1901}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
        },
        {
            {{{56, 20, 32, 16}, {56, 20, 32, 16}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
        },
        {
            {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({4, 5})},
            {"strides", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2, 100})},
            {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 100})},
            {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>("CALCULATED")},
            {"pads", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
        },
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
    EXPECT_EQ(tilingInfo.tilingKey, 0b0'0'0'00000001'00000000);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(Im2ColNCHWTilingData));
    Im2ColNCHWTilingData* data = reinterpret_cast<Im2ColNCHWTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(data->ubFactorH, 20);
    EXPECT_EQ(data->ubFactorW, 128);
    EXPECT_EQ(data->ubFactorNC, 1);
    EXPECT_EQ(data->w4ubFactorW, 1901);
    EXPECT_EQ(data->convKernelNumInWidth, 16);
    EXPECT_EQ(data->convKernelNumInHeight, 32);
    EXPECT_EQ(data->totalRectAngles, 224);
    EXPECT_EQ(data->rectAnglesPerCore, 4);
    EXPECT_EQ(tilingInfo.blockNum, 56);
    CheckUnFullLoadBuf(data, ge::DT_FLOAT16);
}

TEST_F(Im2colTilingNCHWTest, Im2colTilingNCHWTest_nopad_bigw_int8_cuthw)
{
    gert::TilingContextPara tilingContextPara(
        "Im2col",
        {
            {{{1, 48, 258, 1102}, {1, 48, 258, 1102}}, ge::DT_INT8, ge::FORMAT_NCHW},
        },
        {
            {{{1, 768, 128, 550}, {1, 768, 128, 550}}, ge::DT_INT8, ge::FORMAT_NCHW},
        },
        {
            {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({4, 4})},
            {"strides", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2})},
            {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>("CALCULATED")},
            {"pads", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
        },
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
    EXPECT_EQ(tilingInfo.tilingKey, 0b0'0'0'00000001'00000000);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(Im2ColNCHWTilingData));
    Im2ColNCHWTilingData* data = reinterpret_cast<Im2ColNCHWTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(data->ubFactorH, 16);
    EXPECT_EQ(data->ubFactorW, 128);
    EXPECT_EQ(data->ubFactorNC, 1);
    EXPECT_EQ(data->w4ubFactorW, 258);
    EXPECT_EQ(data->convKernelNumInWidth, 550);
    EXPECT_EQ(data->convKernelNumInHeight, 128);
    EXPECT_EQ(data->totalRectAngles, 30720);
    EXPECT_EQ(data->rectAnglesPerCore, 480);
    EXPECT_EQ(tilingInfo.blockNum, 64);
    CheckUnFullLoadBuf(data, ge::DT_INT8);
}

TEST_F(Im2colTilingNCHWTest, Im2colTilingNCHWTest_nopad_W1_bigH_cuthw)
{
    gert::TilingContextPara tilingContextPara(
        "Im2col",
        {
            {{{1, 69, 1453, 1}, {1, 69, 1453, 1}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {{{1, 8556, 1330, 1}, {1, 8556, 1330, 1}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({124, 1})},
            {"strides", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})},
            {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>("VALID")},
            {"pads", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
        },
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
    EXPECT_EQ(tilingInfo.tilingKey, 0b0'0'0'00000001'00000000);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(Im2ColNCHWTilingData));
    Im2ColNCHWTilingData* data = reinterpret_cast<Im2ColNCHWTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(data->ubFactorH, 56);
    EXPECT_EQ(data->ubFactorW, 64);
    EXPECT_EQ(data->ubFactorNC, 1);
    EXPECT_EQ(data->w4ubFactorW, 1);
    EXPECT_EQ(data->convKernelNumInWidth, 1);
    EXPECT_EQ(data->convKernelNumInHeight, 1330);
    EXPECT_EQ(data->totalRectAngles, 4347);
    EXPECT_EQ(data->rectAnglesPerCore, 68);
    EXPECT_EQ(tilingInfo.blockNum, 64);
    CheckUnFullLoadBuf(data, ge::DT_FLOAT);
}

TEST_F(Im2colTilingNCHWTest, Im2colTilingNCHWTest_nopad_cuthw_bufSize_notExceed)
{
    TestUnFullLoadNopadBufSize({11, 48, 9431, 19}, ge::DT_FLOAT, {132, 1}, {20, 34}, {1, 1});
    TestUnFullLoadNopadBufSize({1, 1, 2017, 3896}, ge::DT_FLOAT, {118, 118}, {34, 34}, {1, 1});
    TestUnFullLoadNopadBufSize({188, 5, 1184, 216}, ge::DT_FLOAT, {3, 3}, {37, 37}, {18, 18});
    TestUnFullLoadNopadBufSize({1, 1, 2472, 1}, ge::DT_FLOAT, {242, 1}, {1, 1}, {1, 1});
    TestUnFullLoadNopadBufSize({1, 12, 443, 127}, ge::DT_FLOAT16, {4, 4}, {1, 1}, {1, 1});
    TestUnFullLoadNopadBufSize({1, 2, 167, 1103}, ge::DT_FLOAT16, {26, 26}, {1, 1}, {1, 1});
    TestUnFullLoadNopadBufSize({1, 292, 5856, 192}, ge::DT_FLOAT16, {3, 3}, {23, 57}, {1, 1});
    TestUnFullLoadNopadBufSize({1, 75, 896, 547}, ge::DT_FLOAT16, {3, 3}, {22, 43}, {13, 13});
    TestUnFullLoadNopadBufSize({1, 1, 4409, 1}, ge::DT_FLOAT, {51, 1}, {1, 1}, {1, 1});
    TestUnFullLoadNopadBufSize({1, 80, 773, 67}, ge::DT_FLOAT, {6, 6}, {2, 2}, {1, 1});
    TestUnFullLoadNopadBufSize({1, 69, 1453, 1}, ge::DT_FLOAT, {124, 1}, {1, 1}, {1, 1});
}

TEST_F(Im2colTilingNCHWTest, Im2colTilingNCHWTest_pad_bigLeftPad_int8_cuthw)
{
    gert::TilingContextPara tilingContextPara(
        "Im2col",
        {
            {{{1, 1, 256, 720}, {1, 1, 256, 720}}, ge::DT_INT16, ge::FORMAT_NCHW},
        },
        {
            {{{1, 512, 129, 984}, {1, 512, 129, 984}}, ge::DT_INT16, ge::FORMAT_NCHW},
        },
        {
            {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({128, 4})},
            {"strides", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>("CALCULATED")},
            {"pads", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 257, 10})},
        },
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
    EXPECT_EQ(tilingInfo.tilingKey, 0b0'0'1'00000001'00000000);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(Im2ColNCHWTilingData));
    Im2ColNCHWTilingData* data = reinterpret_cast<Im2ColNCHWTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(data->ubFactorH, 388);
    EXPECT_EQ(data->ubFactorW, 128);
    EXPECT_EQ(data->ubFactorNC, 1);
    EXPECT_EQ(data->w4ubFactorW, 131);
    EXPECT_EQ(data->convKernelNumInWidth, 984);
    EXPECT_EQ(data->convKernelNumInHeight, 129);
    EXPECT_EQ(data->totalRectAngles, 2064);
    EXPECT_EQ(data->rectAnglesPerCore, 33);
    EXPECT_EQ(tilingInfo.blockNum, 63);
    CheckUnFullLoadBuf(data, ge::DT_INT16);
}
