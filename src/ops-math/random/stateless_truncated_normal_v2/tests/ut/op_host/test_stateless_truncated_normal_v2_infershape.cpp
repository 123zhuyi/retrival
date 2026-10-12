/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_stateless_truncated_normal_v2_infershape.cpp
 * \brief StatelessTruncatedNormalV2 InferShape UT
 *
 * 被测实现：op_host/stateless_truncated_normal_v2_infershape.cpp
 *   → ops::randomCommon::CommonInferShape(context, {{"shape",0}}, {{"y",0}}, MODE_DEPENDENCY)
 *
 * 关键语义（random_infershape_base.cpp）：
 *   xShapeSize = inShape->GetShapeSize()   ← 输入 shape tensor 的元素个数 = 输出 rank
 *   DependencyMode() 按 shape tensor 的 dtype 分派：
 *     DT_INT64 → HandleShapeTensor<int64_t>，逐维写入 const 值
 *     DT_INT32 → HandleShapeTensor<int32_t>
 *     其它     → return false → GRAPH_FAILED
 *   const data 为 nullptr 时 → SetUnknownShape(xShapeSize) → 各维 -1
 *
 * 输入布局（与 op_def / tiling UT 一致）：
 *   [0] shape:   DT_INT32/DT_INT64, 1D, const（值依赖，InputsDataDependency({0})）
 *   [1] key:     DT_UINT64, shape {1}
 *   [2] counter: DT_UINT64, shape {2}
 *   [3] alg:     DT_INT32 scalar（ALG_PHILOX = 1）
 * 属性：dtype（OPTIONAL Int，0=float32 / 1=float16 / 27=bfloat16，与本算子 tiling UT 一致的取值约定）
 *
 * infershape 逻辑仅依赖 shape（index 0），key/counter/alg 的取值不影响推导结果，
 * 这里填入与 tiling UT 一致的合法哑值，仅用于保持输入布局完整。
 *
 * 覆盖矩阵：
 *   1D 输出                     : case_1d
 *   2D 输出                     : case_2d
 *   4D 输出                     : case_4d
 *   5D 输出                     : case_5d
 *   含 1 的维度                  : case_dim_with_one
 *   float16 / bfloat16 输出 dtype: case_fp16 / case_bf16
 *   INT32 / INT64 shape tensor  : case_int32_shape / case_int64_shape
 *   const data 缺失 → unknown    : case_null_const_1d / case_null_const_2d
 *   0 元素 shape tensor（标量输出）: case_0dim_scalar
 *   非法 shape dtype → FAILED    : case_invalid_shape_dtype
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

using namespace std;

class StatelessTruncatedNormalV2InferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "StatelessTruncatedNormalV2InferShapeTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "StatelessTruncatedNormalV2InferShapeTest TearDown" << std::endl; }
};

// case 1: 1D 输出。shape tensor = [16384]，1 个元素 → 输出 rank 1
TEST_F(StatelessTruncatedNormalV2InferShapeTest, stateless_truncated_normal_v2_infershape_case_1d)
{
    vector<int64_t> shapeValue = {16384};
    uint64_t keyValue = 42;
    uint64_t counterValue[2] = {0, 0};
    int32_t algValue = 1;
    gert::InfershapeContextPara infershapeContextPara(
        "StatelessTruncatedNormalV2",
        {
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
            {{{1}, {1}}, ge::DT_UINT64, ge::FORMAT_ND, true, &keyValue},
            {{{2}, {2}}, ge::DT_UINT64, ge::FORMAT_ND, true, counterValue},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND, true, &algValue},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{16384}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// case 2: 2D 输出。shape tensor 含 2 个元素 → 输出 rank 2
TEST_F(StatelessTruncatedNormalV2InferShapeTest, stateless_truncated_normal_v2_infershape_case_2d)
{
    vector<int64_t> shapeValue = {32, 512};
    uint64_t keyValue = 42;
    uint64_t counterValue[2] = {0, 0};
    int32_t algValue = 1;
    gert::InfershapeContextPara infershapeContextPara(
        "StatelessTruncatedNormalV2",
        {
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
            {{{1}, {1}}, ge::DT_UINT64, ge::FORMAT_ND, true, &keyValue},
            {{{2}, {2}}, ge::DT_UINT64, ge::FORMAT_ND, true, counterValue},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND, true, &algValue},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{32, 512}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// case 3: 4D 输出，验证高维逐维写入正确
TEST_F(StatelessTruncatedNormalV2InferShapeTest, stateless_truncated_normal_v2_infershape_case_4d)
{
    vector<int64_t> shapeValue = {2, 8, 32, 32};
    uint64_t keyValue = 42;
    uint64_t counterValue[2] = {0, 0};
    int32_t algValue = 1;
    gert::InfershapeContextPara infershapeContextPara(
        "StatelessTruncatedNormalV2",
        {
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
            {{{1}, {1}}, ge::DT_UINT64, ge::FORMAT_ND, true, &keyValue},
            {{{2}, {2}}, ge::DT_UINT64, ge::FORMAT_ND, true, counterValue},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND, true, &algValue},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 8, 32, 32}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// case 4: 5D 输出，覆盖 tiling UT 中出现的最大维度场景
TEST_F(StatelessTruncatedNormalV2InferShapeTest, stateless_truncated_normal_v2_infershape_case_5d)
{
    vector<int64_t> shapeValue = {2, 3, 4, 5, 6};
    uint64_t keyValue = 42;
    uint64_t counterValue[2] = {0, 0};
    int32_t algValue = 1;
    gert::InfershapeContextPara infershapeContextPara(
        "StatelessTruncatedNormalV2",
        {
            {{{5}, {5}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
            {{{1}, {1}}, ge::DT_UINT64, ge::FORMAT_ND, true, &keyValue},
            {{{2}, {2}}, ge::DT_UINT64, ge::FORMAT_ND, true, counterValue},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND, true, &algValue},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3, 4, 5, 6}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// case 5: 含 1 的维度，确认不会被误当作标量或被压缩掉
TEST_F(StatelessTruncatedNormalV2InferShapeTest, stateless_truncated_normal_v2_infershape_case_dim_with_one)
{
    vector<int64_t> shapeValue = {1, 4097, 1};
    uint64_t keyValue = 42;
    uint64_t counterValue[2] = {0, 0};
    int32_t algValue = 1;
    gert::InfershapeContextPara infershapeContextPara(
        "StatelessTruncatedNormalV2",
        {
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
            {{{1}, {1}}, ge::DT_UINT64, ge::FORMAT_ND, true, &keyValue},
            {{{2}, {2}}, ge::DT_UINT64, ge::FORMAT_ND, true, counterValue},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND, true, &algValue},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 4097, 1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// case 6: float16 输出（dtype attr = 1，与本算子约定一致）
TEST_F(StatelessTruncatedNormalV2InferShapeTest, stateless_truncated_normal_v2_infershape_case_fp16)
{
    vector<int64_t> shapeValue = {64, 256};
    uint64_t keyValue = 42;
    uint64_t counterValue[2] = {0, 0};
    int32_t algValue = 1;
    gert::InfershapeContextPara infershapeContextPara(
        "StatelessTruncatedNormalV2",
        {
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
            {{{1}, {1}}, ge::DT_UINT64, ge::FORMAT_ND, true, &keyValue},
            {{{2}, {2}}, ge::DT_UINT64, ge::FORMAT_ND, true, counterValue},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND, true, &algValue},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{64, 256}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// case 7: bfloat16 输出（dtype attr = 27，与本算子约定一致）
TEST_F(StatelessTruncatedNormalV2InferShapeTest, stateless_truncated_normal_v2_infershape_case_bf16)
{
    vector<int64_t> shapeValue = {32, 512};
    uint64_t keyValue = 42;
    uint64_t counterValue[2] = {0, 0};
    int32_t algValue = 1;
    gert::InfershapeContextPara infershapeContextPara(
        "StatelessTruncatedNormalV2",
        {
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
            {{{1}, {1}}, ge::DT_UINT64, ge::FORMAT_ND, true, &keyValue},
            {{{2}, {2}}, ge::DT_UINT64, ge::FORMAT_ND, true, counterValue},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND, true, &algValue},
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(27)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{32, 512}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// case 8: INT32 shape tensor → DependencyMode 的 HandleShapeTensor<int32_t> 分支
TEST_F(StatelessTruncatedNormalV2InferShapeTest, stateless_truncated_normal_v2_infershape_case_int32_shape)
{
    vector<int32_t> shapeValue = {16, 64};
    uint64_t keyValue = 42;
    uint64_t counterValue[2] = {0, 0};
    int32_t algValue = 1;
    gert::InfershapeContextPara infershapeContextPara(
        "StatelessTruncatedNormalV2",
        {
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, shapeValue.data()},
            {{{1}, {1}}, ge::DT_UINT64, ge::FORMAT_ND, true, &keyValue},
            {{{2}, {2}}, ge::DT_UINT64, ge::FORMAT_ND, true, counterValue},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND, true, &algValue},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{16, 64}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// case 9: INT64 shape tensor（显式覆盖，与 case_1d 等共用分支但语义上独立列出）
TEST_F(StatelessTruncatedNormalV2InferShapeTest, stateless_truncated_normal_v2_infershape_case_int64_shape)
{
    vector<int64_t> shapeValue = {8, 128};
    uint64_t keyValue = 42;
    uint64_t counterValue[2] = {0, 0};
    int32_t algValue = 1;
    gert::InfershapeContextPara infershapeContextPara(
        "StatelessTruncatedNormalV2",
        {
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
            {{{1}, {1}}, ge::DT_UINT64, ge::FORMAT_ND, true, &keyValue},
            {{{2}, {2}}, ge::DT_UINT64, ge::FORMAT_ND, true, counterValue},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND, true, &algValue},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{8, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// case 10: const data 缺失（1 元素）→ GetData 返回 nullptr → SetUnknownShape(1) → {-1}
TEST_F(StatelessTruncatedNormalV2InferShapeTest, stateless_truncated_normal_v2_infershape_case_null_const_1d)
{
    vector<int64_t> shapeValue = {};
    uint64_t keyValue = 42;
    uint64_t counterValue[2] = {0, 0};
    int32_t algValue = 1;
    gert::InfershapeContextPara infershapeContextPara(
        "StatelessTruncatedNormalV2",
        {
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
            {{{1}, {1}}, ge::DT_UINT64, ge::FORMAT_ND, true, &keyValue},
            {{{2}, {2}}, ge::DT_UINT64, ge::FORMAT_ND, true, counterValue},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND, true, &algValue},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// case 11: const data 缺失，shape tensor 有 2 个元素 → SetUnknownShape(2) → {-1,-1}
TEST_F(StatelessTruncatedNormalV2InferShapeTest, stateless_truncated_normal_v2_infershape_case_null_const_2d)
{
    vector<int64_t> shapeValue = {};
    uint64_t keyValue = 42;
    uint64_t counterValue[2] = {0, 0};
    int32_t algValue = 1;
    gert::InfershapeContextPara infershapeContextPara(
        "StatelessTruncatedNormalV2",
        {
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
            {{{1}, {1}}, ge::DT_UINT64, ge::FORMAT_ND, true, &keyValue},
            {{{2}, {2}}, ge::DT_UINT64, ge::FORMAT_ND, true, counterValue},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND, true, &algValue},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// case 12: shape tensor 0 元素（标量输出场景）→ 输出 rank 0
TEST_F(StatelessTruncatedNormalV2InferShapeTest, stateless_truncated_normal_v2_infershape_case_0dim_scalar)
{
    int32_t shapeDummy = 0;
    uint64_t keyValue = 42;
    uint64_t counterValue[2] = {0, 0};
    int32_t algValue = 1;
    gert::InfershapeContextPara infershapeContextPara(
        "StatelessTruncatedNormalV2",
        {
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true, &shapeDummy},
            {{{1}, {1}}, ge::DT_UINT64, ge::FORMAT_ND, true, &keyValue},
            {{{2}, {2}}, ge::DT_UINT64, ge::FORMAT_ND, true, counterValue},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND, true, &algValue},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// case 13: 非法 shape dtype（DT_FLOAT）→ DependencyMode 两个分支都不命中 → GRAPH_FAILED
TEST_F(StatelessTruncatedNormalV2InferShapeTest, stateless_truncated_normal_v2_infershape_case_invalid_shape_dtype)
{
    vector<float> shapeValue = {32.0f, 512.0f};
    uint64_t keyValue = 42;
    uint64_t counterValue[2] = {0, 0};
    int32_t algValue = 1;
    gert::InfershapeContextPara infershapeContextPara(
        "StatelessTruncatedNormalV2",
        {
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND, true, shapeValue.data()},
            {{{1}, {1}}, ge::DT_UINT64, ge::FORMAT_ND, true, &keyValue},
            {{{2}, {2}}, ge::DT_UINT64, ge::FORMAT_ND, true, counterValue},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND, true, &algValue},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}
