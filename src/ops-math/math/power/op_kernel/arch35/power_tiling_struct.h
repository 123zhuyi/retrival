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
 * \file power_tiling_struct.h
 * \brief Power 算子 tilingData 结构，承接 host 端 ElewiseBaseTiling::DoTiling 写入的
 *        EleBaseTilingData（dim0 / coreNum / ubFormer / blockNum / ubLoopOf* / 等），
 *        以及 host 端预计算好的 4 个 float 标量（scale / shift / power / negScalar），
 *        kernel 端通过 ElementwiseSch::SetVar 把标量绑定到 DAG 的 Placeholder::Var<>。
 */

#ifndef POWER_TILING_STRUCT_H
#define POWER_TILING_STRUCT_H

#include "atvoss/elewise/elewise_base_struct.h"

namespace PowerOp {
// 不同 culType 下各 scalar 字段的语义（与 power_tiling_arch35.h 的 scalar0..3 一致）：
//   ALL_ZEROS         : 不读取任何 scalar
//   BROADCAST_SCALAR  : scale 存放 host 预计算的 bcastVal
//   LINEAR/SQUARE/CUBE: scale / shift
//   GENERIC_POW_*     : scale / shift / power / negScalar
struct PowerTilingData {
    Ops::Base::EleBaseTilingData baseTiling;
    float scale;
    float shift;
    float power;
    float negScalar;
};
} // namespace PowerOp

#endif // POWER_TILING_STRUCT_H
