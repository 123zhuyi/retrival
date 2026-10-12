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
 * \file drop_out_v3_grad_tiling.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_DROPOUTV3GRAD_RUNTIME2_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_DROPOUTV3GRAD_RUNTIME2_H_

#include <graph/utils/type_utils.h>

#include "atvoss/broadcast/broadcast_tiling.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base_class.h"
#include "op_host/math_tiling_templates_registry.h"

namespace optiling {
struct DropOutV3GradCompileInfo {
    int64_t coreNum;
    int64_t ubEle;
    int64_t ubSize{0};
};
} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_DROPOUTV3GRAD_RUNTIME2_H_
