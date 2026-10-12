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
 * \file signbit_tiling_arch35.h
 * \brief 为arch35架构提供signbit算子的tiling编译信息结构体定义
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_SIGNBIT_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_SIGNBIT_H

namespace optiling {

struct SignbitCompileInfo {
    uint64_t coreNum;
    uint64_t ubSize;
};

} // namespace optiling

#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_SIGNBIT_H