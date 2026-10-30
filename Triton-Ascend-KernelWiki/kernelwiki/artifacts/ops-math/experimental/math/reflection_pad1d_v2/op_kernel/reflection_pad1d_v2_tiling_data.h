/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Zhou Jiamin <@zhou-jiamin-666>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file reflection_pad1d_v2_tiling_data.h
 * \brief
 */
#ifndef __REFLECTION_PAD1D_V2_TILING_DATA_H__
#define __REFLECTION_PAD1D_V2_TILING_DATA_H__

struct ReflectionPad1dV2TilingData {
    uint32_t wSize;
    uint32_t alignWSize;
    uint32_t padLeft;
    uint32_t padRight;
    uint32_t blockNum;
    uint32_t ncPerCore;
    uint32_t tailNC;
    uint32_t ubElementNum;
    uint64_t workspacePerCore;
};
#endif // __REFLECTION_PAD1D_V2_TILING_DATA_H__