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
 * \file diag_v2_tiling.h
 * \brief DiagV2 arch35 tiling declarations
 *
 * Design: DESIGN.md v2.5
 * One-way dependency: diag_v2 → diag_flat (includes diag_flat_tiling.h for 1D→2D delegation).
 */

#ifndef __DIAG_V2_ARCH35_TILING_H__
#define __DIAG_V2_ARCH35_TILING_H__

#include "register/op_impl_registry.h"

namespace optiling {

struct DiagV2CompileInfo {};

} // namespace optiling

#endif // __DIAG_V2_ARCH35_TILING_H__
