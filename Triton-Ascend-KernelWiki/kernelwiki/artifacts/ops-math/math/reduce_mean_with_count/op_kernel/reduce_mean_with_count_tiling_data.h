/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file reduce_mean_with_count_tiling_data.h
 * \brief Tiling data for ReduceMeanWithCount operator.
 *
 * Embeds the standard ReduceOpTilingData computed by Tiling4ReduceOp.
 * The DAG handles x * count / count_sum pre-processing internally,
 * so no additional custom fields are needed beyond the reduce tiling.
 */

#ifndef REDUCE_MEAN_WITH_COUNT_TILING_DATA_H
#define REDUCE_MEAN_WITH_COUNT_TILING_DATA_H

#include "atvoss/reduce/reduce_tiling_data.h"

struct ReduceMeanWithCountTilingData {
    Ops::Base::ReduceOpTilingData reduceTiling;
};

#endif  // REDUCE_MEAN_WITH_COUNT_TILING_DATA_H
