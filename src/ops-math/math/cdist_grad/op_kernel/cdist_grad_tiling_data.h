/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cdist_grad_tiling_data.h
 * \brief Tiling data for CdistGrad operator.
 */

#ifndef CDIST_GRAD_TILING_DATA_H
#define CDIST_GRAD_TILING_DATA_H

#include "atvoss/reduce/reduce_tiling_data.h"

struct CdistGradTilingData {
    Ops::Base::ReduceOpTilingData reduceTiling;
    float powCdist;     // p - 1
    float powDiff;      // p - 2 (used by CdistGradLargePDag)
};

#endif  // CDIST_GRAD_TILING_DATA_H
