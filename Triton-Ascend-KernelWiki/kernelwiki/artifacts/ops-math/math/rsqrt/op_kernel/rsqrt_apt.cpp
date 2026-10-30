/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 /*!
 * \file rsqrt.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "atvoss/elewise/elewise_sch.h"
#include "atvoss/util/dfx.h"
#include "arch35/rsqrt.h"

using namespace AscendC;
using namespace Ops::Base;

extern "C" __global__ __aicore__ void rsqrt(GM_ADDR x, GM_ADDR y,
                                          GM_ADDR workspace, GM_ADDR tiling) 
{
  if (workspace == nullptr) {
    return;
  }
  SetSysWorkspace(workspace);
  GM_ADDR userWS = GetUserWorkspace(workspace);
  if (userWS == nullptr) {
    return;
  }
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  REGISTER_TILING_DEFAULT(EleBaseTilingDataV2);
  GET_TILING_DATA_WITH_STRUCT(EleBaseTilingDataV2, tilingData, tiling);
  TPipe pipe;
  if (TILING_KEY_IS(101UL)) {
    ElementwiseSch<0UL, RsqrtDag::RsqrtOp<half>::OpDag> sch(&tilingData, &pipe); 
    sch.Init(x, y);
    sch.Process();
    return;
  } else if (TILING_KEY_IS(102UL)) {
    ElementwiseSch<0UL, RsqrtDag::RsqrtOp<bfloat16_t>::OpDag> sch(&tilingData, &pipe);
    sch.Init(x, y);
    sch.Process();
    return;
  } else if (TILING_KEY_IS(103UL)) {
    ElementwiseSch<0UL, RsqrtDag::RsqrtOp<float>::OpDag> sch(&tilingData, &pipe); 
    sch.Init(x, y);
    sch.Process();
    return;
  }  
  return;
}