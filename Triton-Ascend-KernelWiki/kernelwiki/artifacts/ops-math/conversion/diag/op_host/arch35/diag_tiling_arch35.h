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
 * \file diag_tiling_arch35.h
 * \brief Tiling header for Diag operator (arch35)
 */

#ifndef OPS_MATH_CONVERSION_DIAG_OP_HOST_ARCH35_DIAG_TILING_ARCH35_H_
#define OPS_MATH_CONVERSION_DIAG_OP_HOST_ARCH35_DIAG_TILING_ARCH35_H_

#include "conversion/diag/op_kernel/arch35/diag_struct.h"
#include "register/op_impl_registry.h"

namespace optiling {

struct DiagCompileInfo {
    uint32_t coreNum{0};
    uint32_t ubSize{0};
};

} // namespace optiling
#endif // OPS_MATH_CONVERSION_DIAG_OP_HOST_ARCH35_DIAG_TILING_ARCH35_H_
