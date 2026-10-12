/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file masked_scale_dag.h
 * \brief
 */

#ifndef CANN_CUSTOM_OPS_MASKED_SCALE_DAG_H
#define CANN_CUSTOM_OPS_MASKED_SCALE_DAG_H
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace MaskedScaleOp {
using namespace Ops::Base;

template <typename Tx, typename Tmask>
struct MaskedScaleMaskCastOne {
  using OpCopyIn0 = Bind<Vec::CopyIn<Tx>, Placeholder::In0<Tx>>;
  using OpCopyIn1 = Bind<Vec::CopyIn<Tmask>, Placeholder::In1<Tmask>>;
  using Cast0 = Bind<Vec::Cast<float, Tx, 0>, OpCopyIn0>;
  using Cast1 = Bind<Vec::Cast<float, Tmask, 0>, OpCopyIn1>;
  using OpMul = Bind<Vec::Mul<float>, Cast0, Cast1>;
  using OpMuls = Bind<Vec::Muls<float>, OpMul, Placeholder::Var<float, 0>>;
  using Cast3 = Bind<Vec::Cast<Tx, float, 1>, OpMuls>;
  using OpCopyOut = Bind<Vec::CopyOut<Tx>, Placeholder::Out0<Tx>, Cast3>;

  using Outputs = Elems<OpCopyOut>;
  using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
  using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename Tx, typename Tmask>
struct MaskedScaleMaskCastTwo {
  using OpCopyIn0 = Bind<Vec::CopyIn<Tx>, Placeholder::In0<Tx>>;
  using OpCopyIn1 = Bind<Vec::CopyIn<Tmask>, Placeholder::In1<Tmask>>;
  using Cast0 = Bind<Vec::Cast<float, Tx, 0>, OpCopyIn0>;
  using Cast1 = Bind<Vec::Cast<half, Tmask, 0>, OpCopyIn1>;
  using Cast2 = Bind<Vec::Cast<float, half, 0>, Cast1>;
  using OpMul = Bind<Vec::Mul<float>, Cast0, Cast2>;
  using OpMuls = Bind<Vec::Muls<float>, OpMul, Placeholder::Var<float, 0>>;
  using Cast3 = Bind<Vec::Cast<Tx, float, 1>, OpMuls>;
  using OpCopyOut = Bind<Vec::CopyOut<Tx>, Placeholder::Out0<Tx>, Cast3>;

  using Outputs = Elems<OpCopyOut>;
  using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
  using OpDag = DAGSch<Outputs, void, MemCfg>;
};

}  // namespace MaskedScaleOp
#endif  // CANN_CUSTOM_OPS_MASKED_SCALE_DAG_H
