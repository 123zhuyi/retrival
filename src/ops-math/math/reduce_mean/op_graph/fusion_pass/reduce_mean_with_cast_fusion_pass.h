/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATH_REDUCE_MEAN_WITH_CAST_FUSION_PASS_H
#define MATH_REDUCE_MEAN_WITH_CAST_FUSION_PASS_H

#include "ge/fusion/pass/pattern_fusion_pass.h"

namespace ops {
using namespace ge;
using namespace fusion;

class __attribute__((visibility("default"))) ReduceMeanWithCastFusionPass : public PatternFusionPass {
protected:
    std::vector<PatternUniqPtr> Patterns() override;

    bool MeetRequirements(const std::unique_ptr<MatchResult>& matchResult) override;

    std::unique_ptr<Graph> Replacement(const std::unique_ptr<MatchResult>& matchResult) override;
};

} // namespace ops
#endif // MATH_REDUCE_MEAN_WITH_CAST_FUSION_PASS_H
