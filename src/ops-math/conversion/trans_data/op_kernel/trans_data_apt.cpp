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
 * \file trans_data_apt.cpp
 * \brief kernel of op trans_data
 */
#include "arch35/trans_data.h"

extern "C" __global__ __aicore__ void trans_data(GM_ADDR src, GM_ADDR dst, GM_ADDR workspace, GM_ADDR tiling)
{
    trans_data_impl(src, dst, workspace, tiling);
}
