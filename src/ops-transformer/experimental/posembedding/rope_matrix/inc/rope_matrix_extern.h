/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ROPE_MATRIX_EXTERN_H
#define ROPE_MATRIX_EXTERN_H

#include <cstdint>

namespace npu_ops_transformer_ext {
namespace RopeMatrix {
// define a struct as TCubeTiling, which can be call by both op_device and op_host
struct RopeMatrixTiling
{
    uint32_t blockDim;
    uint32_t b;
    uint32_t n;
    uint32_t s;
    uint32_t d;
};
} // namespace RopeMatrix
} //namespace npu_ops_transformer_ext

#endif