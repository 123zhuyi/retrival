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
 * \file cumsum_tiling_ascendc.h
 * \brief cumsum tiling for ascendc float
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_CUMSUM_TILING_ASCENDC_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_CUMSUM_TILING_ASCENDC_H

#include "register/tilingdata_base.h"
#include "op_host/tiling_base_class.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(CumsumFoldPara)
TILING_DATA_FIELD_DEF(int32_t, foldCount);
TILING_DATA_FIELD_DEF(int64_t, foldLen);
TILING_DATA_FIELD_DEF(int32_t, sklanskyIter);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(CumsumFoldParaOp, CumsumFoldPara);

BEGIN_TILING_DATA_DEF(CumsumBlockPara)
TILING_DATA_FIELD_DEF(int32_t, blockCount);
TILING_DATA_FIELD_DEF(int64_t, blockFactor);
TILING_DATA_FIELD_DEF(int64_t, blockTailFactor);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(CumsumBlockParaOp, CumsumBlockPara);

BEGIN_TILING_DATA_DEF(UbParaUnit)
TILING_DATA_FIELD_DEF(int32_t, ubCount);
TILING_DATA_FIELD_DEF(int32_t, ubFactor);
TILING_DATA_FIELD_DEF(int32_t, ubTailFactor);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(UbParaUnitOp, UbParaUnit);

BEGIN_TILING_DATA_DEF(CumsumUbPara)
TILING_DATA_FIELD_DEF_STRUCT(UbParaUnit, mainCoreUbPara);
TILING_DATA_FIELD_DEF_STRUCT(UbParaUnit, tailCoreUbPara);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(CumsumUbParaOp, CumsumUbPara);

BEGIN_TILING_DATA_DEF(CumsumIterGroupTiling)
TILING_DATA_FIELD_DEF(int64_t, iterGroupBlockFactor);
TILING_DATA_FIELD_DEF(int32_t, iterGroupMainBlockCount);
TILING_DATA_FIELD_DEF(int32_t, iterGroupTailBlockCount);
TILING_DATA_FIELD_DEF(int32_t, iterGroupMainCoreUbFactorCount);
TILING_DATA_FIELD_DEF(int32_t, iterGroupMainCoreUbFactor);
TILING_DATA_FIELD_DEF(int32_t, iterGroupMainCoreUbTailFactor);
TILING_DATA_FIELD_DEF(int32_t, iterGroupTailCoreUbFactorCount);
TILING_DATA_FIELD_DEF(int32_t, iterGroupTailCoreUbFactor);
TILING_DATA_FIELD_DEF(int32_t, iterGroupTailCoreUbTailFactor);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(CumsumIterGroupTilingOp, CumsumIterGroupTiling);

BEGIN_TILING_DATA_DEF(CumsumSklanskyItersTiling)
TILING_DATA_FIELD_DEF(int32_t, iterGroupCount);
TILING_DATA_FIELD_DEF(int32_t, iterGroupAddCount);
TILING_DATA_FIELD_DEF(int32_t, iterGroupAddOffset);
TILING_DATA_FIELD_DEF(int32_t, iterGroupStartOffset);
TILING_DATA_FIELD_DEF(int32_t, iterGroupCoreNum);
TILING_DATA_FIELD_DEF_STRUCT(CumsumIterGroupTiling, mainIterGroupMainNTiling);
TILING_DATA_FIELD_DEF_STRUCT(CumsumIterGroupTiling, tailIterGroupMainNTiling);
TILING_DATA_FIELD_DEF_STRUCT(CumsumIterGroupTiling, mainIterGroupTailNTiling);
TILING_DATA_FIELD_DEF_STRUCT(CumsumIterGroupTiling, tailIterGroupTailNTiling);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(CumsumSklanskyItersTilingOp, CumsumSklanskyItersTiling);

BEGIN_TILING_DATA_DEF(CumsumSklanskyTilingData)
TILING_DATA_FIELD_DEF(int64_t, lenM);
TILING_DATA_FIELD_DEF(int64_t, lenR);
TILING_DATA_FIELD_DEF(int64_t, lenN);
TILING_DATA_FIELD_DEF(int32_t, exclusive);
TILING_DATA_FIELD_DEF(int32_t, reverse);
TILING_DATA_FIELD_DEF(int32_t, realCoreNum);
TILING_DATA_FIELD_DEF(int32_t, xBufSize);
TILING_DATA_FIELD_DEF(int32_t, xUnfoldBufSize);
TILING_DATA_FIELD_DEF(int32_t, ubSklanskyBufSize);
TILING_DATA_FIELD_DEF(int32_t, addFactorRepeatTimesMain);
TILING_DATA_FIELD_DEF(int32_t, addFactorRepeatTimesTail);
TILING_DATA_FIELD_DEF(int32_t, coreGroupCount);
TILING_DATA_FIELD_DEF(int32_t, coreGroupCoreNum);
TILING_DATA_FIELD_DEF(int32_t, sklanskyItersCount);
TILING_DATA_FIELD_DEF(int32_t, addFactorBufferSize);
TILING_DATA_FIELD_DEF(int32_t, addDataBufferSize);
TILING_DATA_FIELD_DEF(int32_t, addFactorOffsetBufferSize);
TILING_DATA_FIELD_DEF(int32_t, nBorrow);
TILING_DATA_FIELD_DEF(int32_t, rBorrow);
TILING_DATA_FIELD_DEF(int32_t, mBorrow);
TILING_DATA_FIELD_DEF_STRUCT(CumsumFoldPara, mainCoreMainUbFoldPara);
TILING_DATA_FIELD_DEF_STRUCT(CumsumFoldPara, mainCoreTailUbFoldPara);
TILING_DATA_FIELD_DEF_STRUCT(CumsumFoldPara, tailCoreMainUbFoldPara);
TILING_DATA_FIELD_DEF_STRUCT(CumsumFoldPara, tailCoreTailUbFoldPara);
TILING_DATA_FIELD_DEF_STRUCT(CumsumBlockPara, mBlockPara);
TILING_DATA_FIELD_DEF_STRUCT(CumsumBlockPara, rBlockPara);
TILING_DATA_FIELD_DEF_STRUCT(CumsumBlockPara, nBlockPara);
TILING_DATA_FIELD_DEF_STRUCT(CumsumUbPara, mUbPara);
TILING_DATA_FIELD_DEF_STRUCT(CumsumUbPara, rUbPara);
TILING_DATA_FIELD_DEF_STRUCT(CumsumUbPara, nUbPara);
TILING_DATA_FIELD_DEF_STRUCT(CumsumSklanskyItersTiling, sklanskyItersTiling0);
TILING_DATA_FIELD_DEF_STRUCT(CumsumSklanskyItersTiling, sklanskyItersTiling1);
TILING_DATA_FIELD_DEF_STRUCT(CumsumSklanskyItersTiling, sklanskyItersTiling2);
TILING_DATA_FIELD_DEF_STRUCT(CumsumSklanskyItersTiling, sklanskyItersTiling3);
TILING_DATA_FIELD_DEF_STRUCT(CumsumSklanskyItersTiling, sklanskyItersTiling4);
TILING_DATA_FIELD_DEF_STRUCT(CumsumSklanskyItersTiling, sklanskyItersTiling5);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Cumsum_1001, CumsumSklanskyTilingData);
REGISTER_TILING_DATA_CLASS(Cumsum_1002, CumsumSklanskyTilingData);
REGISTER_TILING_DATA_CLASS(Cumsum_1011, CumsumSklanskyTilingData);
REGISTER_TILING_DATA_CLASS(Cumsum_1012, CumsumSklanskyTilingData);
REGISTER_TILING_DATA_CLASS(Cumsum_1101, CumsumSklanskyTilingData);
REGISTER_TILING_DATA_CLASS(Cumsum_1102, CumsumSklanskyTilingData);
REGISTER_TILING_DATA_CLASS(Cumsum_1111, CumsumSklanskyTilingData);
REGISTER_TILING_DATA_CLASS(Cumsum_1112, CumsumSklanskyTilingData);

ge::graphStatus TilingCumsumAscendc(gert::TilingContext* context);

} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_CUMSUM_TILING_ASCENDC_H