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
 * \file masked_scale_apt.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "arch35/masked_scale_dag.h"
#include "arch35/masked_scale_tiling_struct.h"
#include "atvoss/elewise/elewise_sch.h"
#include "atvoss/util/dfx.h"

using namespace AscendC;
using namespace MaskedScaleOp;
extern "C" __global__ __aicore__ void masked_scale(GM_ADDR x, GM_ADDR mask, GM_ADDR y, GM_ADDR workspace,
                                                   GM_ADDR tiling) {
  if (workspace == nullptr) {
      return;
  }

  SetSysWorkspace(workspace);
  GM_ADDR userWS = GetUserWorkspace(workspace);
  if (userWS == nullptr) {
      return;
  }
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  REGISTER_TILING_DEFAULT(MaskedScaleNs::MaskedScaleTilingData);
  GET_TILING_DATA_WITH_STRUCT(MaskedScaleNs::MaskedScaleTilingData, tilingData, tiling);
  TPipe pipe;
  if (TILING_KEY_IS(11UL)) {
      ElementwiseSch<0UL, MaskedScaleMaskCastTwo<half, uint8_t>::OpDag> sch(&(tilingData.baseTiling), &pipe);
      sch.SetVar<float, 0>(tilingData.scale);
      sch.Init(x, mask, y);
      sch.Process();
      return;
  } else if (TILING_KEY_IS(12UL)) {
      ElementwiseSch<0UL, MaskedScaleMaskCastTwo<bfloat16_t, uint8_t>::OpDag> sch(&(tilingData.baseTiling), &pipe);
      sch.SetVar<float, 0>(tilingData.scale);
      sch.Init(x, mask, y);
      sch.Process();
      return;
  } else if (TILING_KEY_IS(13UL)) {
      ElementwiseSch<0UL, MaskedScaleMaskCastTwo<float, uint8_t>::OpDag> sch(&(tilingData.baseTiling), &pipe);
      sch.SetVar<float, 0>(tilingData.scale);
      sch.Init(x, mask, y);
      sch.Process();
      return;
  } else if (TILING_KEY_IS(21UL)) {
      ElementwiseSch<0UL, MaskedScaleMaskCastTwo<half, int8_t>::OpDag> sch(&(tilingData.baseTiling), &pipe);
      sch.SetVar<float, 0>(tilingData.scale);
      sch.Init(x, mask, y);
      sch.Process();
      return;
  } else if (TILING_KEY_IS(22UL)) {
      ElementwiseSch<0UL, MaskedScaleMaskCastTwo<bfloat16_t, int8_t>::OpDag> sch(&(tilingData.baseTiling), &pipe);
      sch.SetVar<float, 0>(tilingData.scale);
      sch.Init(x, mask, y);
      sch.Process();
      return;
  } else if (TILING_KEY_IS(23UL)) {
      ElementwiseSch<0UL, MaskedScaleMaskCastTwo<float, int8_t>::OpDag> sch(&(tilingData.baseTiling), &pipe);
      sch.SetVar<float, 0>(tilingData.scale);
      sch.Init(x, mask, y);
      sch.Process();
      return;
  } else if (TILING_KEY_IS(31UL)) {
      ElementwiseSch<0UL, MaskedScaleMaskCastOne<half, half>::OpDag> sch(&(tilingData.baseTiling), &pipe);
      sch.SetVar<float, 0>(tilingData.scale);
      sch.Init(x, mask, y);
      sch.Process();
      return;
  } else if (TILING_KEY_IS(32UL)) {
      ElementwiseSch<0UL, MaskedScaleMaskCastOne<bfloat16_t, half>::OpDag> sch(&(tilingData.baseTiling), &pipe);
      sch.SetVar<float, 0>(tilingData.scale);
      sch.Init(x, mask, y);
      sch.Process();
      return;
  } else if (TILING_KEY_IS(33UL)) {
      ElementwiseSch<0UL, MaskedScaleMaskCastOne<float, half>::OpDag> sch(&(tilingData.baseTiling), &pipe);
      sch.SetVar<float, 0>(tilingData.scale);
      sch.Init(x, mask, y);
      sch.Process();
      return;
  } else if (TILING_KEY_IS(41UL)) {
      ElementwiseSch<0UL, MaskedScaleMaskCastOne<half, float>::OpDag> sch(&(tilingData.baseTiling), &pipe);
      sch.SetVar<float, 0>(tilingData.scale);
      sch.Init(x, mask, y);
      sch.Process();
      return;
  } else if (TILING_KEY_IS(42UL)) {
      ElementwiseSch<0UL, MaskedScaleMaskCastOne<bfloat16_t, float>::OpDag> sch(&(tilingData.baseTiling), &pipe);
      sch.SetVar<float, 0>(tilingData.scale);
      sch.Init(x, mask, y);
      sch.Process();
      return;
  } else if (TILING_KEY_IS(43UL)) {
      ElementwiseSch<0UL, MaskedScaleMaskCastOne<float, float>::OpDag> sch(&(tilingData.baseTiling), &pipe);
      sch.SetVar<float, 0>(tilingData.scale);
      sch.Init(x, mask, y);
      sch.Process();
      return;
  }
  return;
}
