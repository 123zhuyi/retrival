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
 * \file trilu_apt.cpp
 * \brief Kernel entry for trilu operator.
 *
 * GM_ADDR parameters: x (REQUIRED INPUT), k (OPTIONAL INPUT, ignored in kernel),
 * y (REQUIRED OUTPUT), workspace, tiling.
 * k's value was read during tiling (value dependency) and encoded into tiling
 * data; the kernel ignores the k parameter.
 */

#include "arch35/trilu_entry.h"

extern "C" __global__ __aicore__ void trilu(GM_ADDR x, GM_ADDR k, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)k; // k is ignored; its value was consumed during tiling
    NsTrilu::triluEntry(x, y, workspace, tiling);
}
