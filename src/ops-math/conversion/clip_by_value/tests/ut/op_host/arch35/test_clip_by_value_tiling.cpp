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
 * \file test_clip_by_value_tiling.cpp
 * \brief
 */

#include "conversion/clip_by_value/op_host/arch35/clip_by_value_tiling.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class ClipByValueTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ClipByValueTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ClipByValueTiling TearDown" << std::endl;
    }
};

static gert::StorageShape MakeStorageShape(const std::vector<int64_t>& shape)
{
    gert::StorageShape storageShape;
    for (const auto dim : shape) {
        storageShape.MutableOriginShape().AppendDim(dim);
        storageShape.MutableStorageShape().AppendDim(dim);
    }
    return storageShape;
}

static void RunClipByValueSuccessCase(ge::DataType dtype, const std::vector<int64_t>& xShape,
    const std::vector<int64_t>& minShape, const std::vector<int64_t>& maxShape,
    const std::vector<int64_t>& yShape)
{
    optiling::ClipByValueCompileInfo compileInfo = {64, 245760};
    gert::StorageShape xStorageShape = MakeStorageShape(xShape);
    gert::StorageShape minStorageShape = MakeStorageShape(minShape);
    gert::StorageShape maxStorageShape = MakeStorageShape(maxShape);
    gert::StorageShape yStorageShape = MakeStorageShape(yShape);
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc = {
        {xStorageShape, dtype, ge::FORMAT_ND},
        {minStorageShape, dtype, ge::FORMAT_ND},
        {maxStorageShape, dtype, ge::FORMAT_ND},
    };
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc = {
        {yStorageShape, dtype, ge::FORMAT_ND},
    };
    gert::TilingContextPara tilingContextPara(
        "ClipByValue", inputTensorDesc, outputTensorDesc, &compileInfo);
    ExecuteTestCaseForEle(tilingContextPara, ge::GRAPH_SUCCESS, false, 0, false, "", {16777216});
}

TEST_F(ClipByValueTiling, same_shape_fp16)
{
    RunClipByValueSuccessCase(ge::DT_FLOAT16, {48, 16, 16, 16}, {1}, {1}, {48, 16, 16, 16});
}

TEST_F(ClipByValueTiling, last_dim_broadcast_fp16)
{
    RunClipByValueSuccessCase(
        ge::DT_FLOAT16, {48, 16, 16, 16}, {48, 16, 16, 1}, {48, 16, 16, 1}, {48, 16, 16, 16});
}

TEST_F(ClipByValueTiling, fp32_broadcast)
{
    RunClipByValueSuccessCase(
        ge::DT_FLOAT, {48, 16, 16, 16}, {48, 16, 16, 1}, {48, 16, 16, 1}, {48, 16, 16, 16});
}

TEST_F(ClipByValueTiling, bf16_scalar_bounds)
{
    RunClipByValueSuccessCase(ge::DT_BF16, {32, 64}, {1}, {1}, {32, 64});
}

TEST_F(ClipByValueTiling, int32_broadcast)
{
    RunClipByValueSuccessCase(
        ge::DT_INT32, {48, 16, 16, 16}, {48, 16, 16, 1}, {48, 16, 16, 1}, {48, 16, 16, 16});
}

TEST_F(ClipByValueTiling, int64_broadcast)
{
    RunClipByValueSuccessCase(
        ge::DT_INT64, {48, 16, 16, 16}, {48, 16, 16, 1}, {48, 16, 16, 1}, {48, 16, 16, 16});
}

TEST_F(ClipByValueTiling, mismatched_dtype_failed)
{
    optiling::ClipByValueCompileInfo compileInfo = {64, 245760};
    gert::StorageShape xStorageShape({16, 16}, {16, 16});
    gert::StorageShape boundStorageShape({1}, {1});
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc = {
        {xStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND},
        {boundStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND},
        {boundStorageShape, ge::DT_FLOAT, ge::FORMAT_ND},
    };
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc = {
        {xStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND},
    };
    gert::TilingContextPara tilingContextPara(
        "ClipByValue", inputTensorDesc, outputTensorDesc, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
