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
 * \file asin_tilingdata.h
 * \brief Asin tiling data
 */

#ifndef ASIN_TILINGDATA_H
#define ASIN_TILINGDATA_H

#include "atvoss/elewise/elewise_base_struct.h"

using namespace Ops::Base;

struct AsinTilingData {
    EleBaseTilingData baseTiling;
    // Asin 算子无额外参数
};

#endif  // ASIN_TILINGDATA_H
