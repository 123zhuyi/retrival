/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"
using OpAttr = gert::InfershapeContextPara::OpAttr;

class Im2ColInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "Im2ColInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "Im2ColInfershape TearDown" << std::endl; }
};

// 辅助函数：计算Im2Col输出形状
static std::vector<int64_t> CalcIm2ColOutputShape(
    const std::vector<int64_t>& input_shape, int64_t kernel_h, int64_t kernel_w, int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w, int64_t dilation_h, int64_t dilation_w)
{
    // 输入形状解析
    int64_t H, W, C, N;
    if (input_shape.size() == 4) {
        // [N, C, H, W]
        N = input_shape[0];
        C = input_shape[1];
        H = input_shape[2];
        W = input_shape[3];
    } else if (input_shape.size() == 3) {
        // [C, H, W] 或 [N, H, W]
        // 假设是 [N, H, W] 格式，C=1
        N = input_shape[0];
        C = 1;
        H = input_shape[1];
        W = input_shape[2];
    } else if (input_shape.size() == 2) {
        // [H, W]
        N = 1;
        C = 1;
        H = input_shape[0];
        W = input_shape[1];
    } else {
        // 不支持其他维度
        return {};
    }

    // 计算输出高度和宽度
    int64_t out_H = (H + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int64_t out_W = (W + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    int64_t L = out_H * out_W;
    int64_t output_channels = C * kernel_h * kernel_w;

    // 输出形状: [N, output_channels, L]
    return {N, output_channels, L};
}

// 基础测试：4D输入，默认参数
TEST_F(Im2ColInfershape, im2col_4d_default_params)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Im2Col",
        {
            {{{2, 3, 4, 4}, {2, 3, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        // 属性参数：kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w
        {
            OpAttr("kernel_h", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),   // kernel_h
            OpAttr("kernel_w", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),   // kernel_w
            OpAttr("stride_h", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),   // stride_h
            OpAttr("stride_w", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),   // stride_w
            OpAttr("pad_h", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),      // pad_h
            OpAttr("pad_w", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),      // pad_w
            OpAttr("dilation_h", Ops::Math::AnyValue::CreateFrom<int64_t>(1)), // dilation_h
            OpAttr("dilation_w", Ops::Math::AnyValue::CreateFrom<int64_t>(1)), // dilation_w
        });

    std::vector<int64_t> input_shape = {2, 3, 4, 4};
    auto expected_shape = CalcIm2ColOutputShape(input_shape, 2, 2, 1, 1, 0, 0, 1, 1);

    std::vector<std::vector<int64_t>> expectOutputShape = {expected_shape};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 测试：带有padding的4D输入
TEST_F(Im2ColInfershape, im2col_4d_with_padding)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Im2Col",
        {
            {{{1, 16, 28, 28}, {1, 16, 28, 28}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            OpAttr("kernel_h", Ops::Math::AnyValue::CreateFrom<int64_t>(3)),   // kernel_h
            OpAttr("kernel_w", Ops::Math::AnyValue::CreateFrom<int64_t>(3)),   // kernel_w
            OpAttr("stride_h", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),   // stride_h
            OpAttr("stride_w", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),   // stride_w
            OpAttr("pad_h", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),      // pad_h
            OpAttr("pad_w", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),      // pad_w
            OpAttr("dilation_h", Ops::Math::AnyValue::CreateFrom<int64_t>(1)), // dilation_h
            OpAttr("dilation_w", Ops::Math::AnyValue::CreateFrom<int64_t>(1)), // dilation_w
        });

    std::vector<int64_t> input_shape = {1, 16, 28, 28};
    auto expected_shape = CalcIm2ColOutputShape(input_shape, 3, 3, 2, 2, 1, 1, 1, 1);

    std::vector<std::vector<int64_t>> expectOutputShape = {expected_shape};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 测试：带有dilation的3D输入
TEST_F(Im2ColInfershape, im2col_3d_with_dilation)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Im2Col",
        {
            {{{8, 32, 32}, {8, 32, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            OpAttr("kernel_h", Ops::Math::AnyValue::CreateFrom<int64_t>(3)),   // kernel_h
            OpAttr("kernel_w", Ops::Math::AnyValue::CreateFrom<int64_t>(3)),   // kernel_w
            OpAttr("stride_h", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),   // stride_h
            OpAttr("stride_w", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),   // stride_w
            OpAttr("pad_h", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),      // pad_h
            OpAttr("pad_w", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),      // pad_w
            OpAttr("dilation_h", Ops::Math::AnyValue::CreateFrom<int64_t>(2)), // dilation_h
            OpAttr("dilation_w", Ops::Math::AnyValue::CreateFrom<int64_t>(2)), // dilation_w
        });

    std::vector<int64_t> input_shape = {8, 32, 32};
    auto expected_shape = CalcIm2ColOutputShape(input_shape, 3, 3, 1, 1, 2, 2, 2, 2);

    std::vector<std::vector<int64_t>> expectOutputShape = {expected_shape};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 测试：2D输入（单张图像）
TEST_F(Im2ColInfershape, im2col_2d_single_image)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Im2Col",
        {
            {{{64, 64}, {64, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            OpAttr("kernel_h", Ops::Math::AnyValue::CreateFrom<int64_t>(5)),   // kernel_h
            OpAttr("kernel_w", Ops::Math::AnyValue::CreateFrom<int64_t>(5)),   // kernel_w
            OpAttr("stride_h", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),   // stride_h
            OpAttr("stride_w", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),   // stride_w
            OpAttr("pad_h", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),      // pad_h
            OpAttr("pad_w", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),      // pad_w
            OpAttr("dilation_h", Ops::Math::AnyValue::CreateFrom<int64_t>(1)), // dilation_h
            OpAttr("dilation_w", Ops::Math::AnyValue::CreateFrom<int64_t>(1)), // dilation_w
        });

    std::vector<int64_t> input_shape = {64, 64};
    auto expected_shape = CalcIm2ColOutputShape(input_shape, 5, 5, 1, 1, 0, 0, 1, 1);

    std::vector<std::vector<int64_t>> expectOutputShape = {expected_shape};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 测试：大尺寸输入
TEST_F(Im2ColInfershape, im2col_large_batch)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Im2Col",
        {
            {{{32, 256, 56, 56}, {32, 256, 56, 56}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            OpAttr("kernel_h", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),   // kernel_h
            OpAttr("kernel_w", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),   // kernel_w
            OpAttr("stride_h", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),   // stride_h
            OpAttr("stride_w", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),   // stride_w
            OpAttr("pad_h", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),      // pad_h
            OpAttr("pad_w", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),      // pad_w
            OpAttr("dilation_h", Ops::Math::AnyValue::CreateFrom<int64_t>(1)), // dilation_h
            OpAttr("dilation_w", Ops::Math::AnyValue::CreateFrom<int64_t>(1)), // dilation_w
        });

    std::vector<int64_t> input_shape = {32, 256, 56, 56};
    auto expected_shape = CalcIm2ColOutputShape(input_shape, 1, 1, 1, 1, 0, 0, 1, 1);

    std::vector<std::vector<int64_t>> expectOutputShape = {expected_shape};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 测试：动态形状（带-1维度）
// TEST_F(Im2ColInfershape, im2col_dynamic_shape)
// {
//     gert::InfershapeContextPara infershapeContextPara(
//         "Im2Col",
//         {
//             {{{-1, 3, -1, -1}, {-1, 3, -1, -1}}, ge::DT_FLOAT, ge::FORMAT_ND},
//         },
//         {
//             {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
//         },
//         {
//             OpAttr("kernel_h", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),  // kernel_h
//             OpAttr("kernel_w", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),  // kernel_w
//             OpAttr("stride_h", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),  // stride_h
//             OpAttr("stride_w", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),  // stride_w
//             OpAttr("pad_h", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),  // pad_h
//             OpAttr("pad_w", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),  // pad_w
//             OpAttr("dilation_h", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),  // dilation_h
//             OpAttr("dilation_w", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),  // dilation_w
//         });

//     // 对于动态形状，期望输出也是动态的
//     std::vector<std::vector<int64_t>> expectOutputShape = {
//         {-1, 12, -1}  // N=动态, output_channels=3*2*2=12, L=动态
//     };
//     ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
// }

// 测试：错误情况 - 输入维度不足
TEST_F(Im2ColInfershape, im2col_invalid_dimension)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Im2Col",
        {
            {{{10}, {10}}, ge::DT_FLOAT, ge::FORMAT_ND}, // 只有1D，应该失败
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            OpAttr("kernel_h", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),   // kernel_h
            OpAttr("kernel_w", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),   // kernel_w
            OpAttr("stride_h", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),   // stride_h
            OpAttr("stride_w", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),   // stride_w
            OpAttr("pad_h", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),      // pad_h
            OpAttr("pad_w", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),      // pad_w
            OpAttr("dilation_h", Ops::Math::AnyValue::CreateFrom<int64_t>(1)), // dilation_h
            OpAttr("dilation_w", Ops::Math::AnyValue::CreateFrom<int64_t>(1)), // dilation_w
        });

    // 期望推断失败
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {} // 空形状表示失败
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// 测试：不同类型的数据类型
TEST_F(Im2ColInfershape, im2col_different_dtypes)
{
    // 测试float16
    {
        gert::InfershapeContextPara infershapeContextPara(
            "Im2Col",
            {
                {{{4, 8, 16, 16}, {4, 8, 16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            },
            {
                {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            },
            {
                OpAttr("kernel_h", Ops::Math::AnyValue::CreateFrom<int64_t>(3)),   // kernel_h
                OpAttr("kernel_w", Ops::Math::AnyValue::CreateFrom<int64_t>(3)),   // kernel_w
                OpAttr("stride_h", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),   // stride_h
                OpAttr("stride_w", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),   // stride_w
                OpAttr("pad_h", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),      // pad_h
                OpAttr("pad_w", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),      // pad_w
                OpAttr("dilation_h", Ops::Math::AnyValue::CreateFrom<int64_t>(1)), // dilation_h
                OpAttr("dilation_w", Ops::Math::AnyValue::CreateFrom<int64_t>(1)), // dilation_w
            });

        std::vector<int64_t> input_shape = {4, 8, 16, 16};
        auto expected_shape = CalcIm2ColOutputShape(input_shape, 3, 3, 1, 1, 1, 1, 1, 1);

        std::vector<std::vector<int64_t>> expectOutputShape = {expected_shape};
        ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
    }

    // 测试float32
    {
        gert::InfershapeContextPara infershapeContextPara(
            "Im2Col",
            {
                {{{4, 8, 16, 16}, {4, 8, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            },
            {
                {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            },
            {
                OpAttr("kernel_h", Ops::Math::AnyValue::CreateFrom<int64_t>(3)),   // kernel_h
                OpAttr("kernel_w", Ops::Math::AnyValue::CreateFrom<int64_t>(3)),   // kernel_w
                OpAttr("stride_h", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),   // stride_h
                OpAttr("stride_w", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),   // stride_w
                OpAttr("pad_h", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),      // pad_h
                OpAttr("pad_w", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),      // pad_w
                OpAttr("dilation_h", Ops::Math::AnyValue::CreateFrom<int64_t>(1)), // dilation_h
                OpAttr("dilation_w", Ops::Math::AnyValue::CreateFrom<int64_t>(1)), // dilation_w
            });

        std::vector<int64_t> input_shape = {4, 8, 16, 16};
        auto expected_shape = CalcIm2ColOutputShape(input_shape, 3, 3, 1, 1, 1, 1, 1, 1);

        std::vector<std::vector<int64_t>> expectOutputShape = {expected_shape};
        ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
    }
}