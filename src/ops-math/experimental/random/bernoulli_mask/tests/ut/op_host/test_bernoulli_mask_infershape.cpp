/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "infershape_case_executor.h"
#include "infershape_context_faker.h"

namespace {
void CheckInferShape(const std::vector<int64_t>& requestedShape)
{
    gert::StorageShape maskShape = {{128}, {128}};
    gert::StorageShape outputShape = {};
    gert::InfershapeContextPara context(
        "BernoulliMask",
        {
            {maskShape, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {outputShape, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("output_shape",
                                                Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>(requestedShape)),
        });
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {requestedShape});
}

void CheckInferShapeRejected(const std::vector<int64_t>& requestedShape)
{
    gert::StorageShape maskShape = {{128}, {128}};
    gert::StorageShape outputShape = {};
    gert::InfershapeContextPara context(
        "BernoulliMask",
        {
            {maskShape, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {outputShape, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("output_shape",
                                                Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>(requestedShape)),
        });
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST(BernoulliMaskInferShape, preserves_rank_eight_shape) { CheckInferShape({1, 2, 1, 2, 1, 2, 1, 2}); }

TEST(BernoulliMaskInferShape, preserves_scalar_shape) { CheckInferShape({}); }

TEST(BernoulliMaskInferShape, preserves_empty_dimension) { CheckInferShape({2, 0, 3}); }

TEST(BernoulliMaskInferShape, rejects_rank_greater_than_eight) { CheckInferShapeRejected({1, 1, 1, 1, 1, 1, 1, 1, 1}); }

TEST(BernoulliMaskInferShape, rejects_negative_dimension) { CheckInferShapeRejected({2, -1, 3}); }
} // namespace
