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
 * \file pad_v2_tiling_arch35.h
 * \brief PadV2 tiling header - 复用 PadV3 实现
 */
#ifndef OP_HOST_ARCH35_PAD_V2_TILING_ARCH35_H_
#define OP_HOST_ARCH35_PAD_V2_TILING_ARCH35_H_

#include "conversion/pad_v3/op_host/arch35/pad_v3_tiling_arch35.h"

// PadV2 直接复用 PadV3 的 Tiling 实现，无需额外类型定义
// Tiling 实现使用：PadACTiling
// 编译信息使用：PadV3CompileInfo
// 参考：op_host/arch35/pad_v2_tiling_arch35.cpp

#endif // OP_HOST_ARCH35_PAD_V2_TILING_ARCH35_H_
