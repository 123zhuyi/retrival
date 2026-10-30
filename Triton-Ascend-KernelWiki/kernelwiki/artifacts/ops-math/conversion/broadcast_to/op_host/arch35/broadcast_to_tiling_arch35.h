/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file broadcast_to_tiling_arch35.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_BROADCAST_TO_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_BROADCAST_TO_H_
#include <cstdint>
#include <vector>
#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"

namespace optiling {
struct BroadcastToCompileInfo {
    int64_t coreNum;
    int64_t ubSize;
    uint32_t clSize;
    uint32_t vRegSize;
    int64_t blockSize;
};
} // namespace optiling
#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_BROADCAST_TO_H_
