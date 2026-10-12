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
 * \file quant_grouped_matmul_dequant_tiling.cpp
 * \brief
 */

#include <cmath>
#include <vector>
#include "register/op_impl_registry.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling_base/tiling_base.h"
#include "log/log.h"
#include "err/ops_err.h"
#include "platform/platform_infos_def.h"
#include "quant_grouped_matmul_dequant_tiling.h"

bool AddWorkspace(gert::TilingContext* context, const size_t workspace) {
  size_t* workspace_size = context->GetWorkspaceSizes(1);
  OP_CHECK_NULL_WITH_CONTEXT(context, workspace_size);
  *workspace_size = workspace;
  return true;
}

namespace optiling {

bool QuantMatmulDequantTiling::CheckInOutShapes(gert::TilingContext* context)
{
    uint32_t idx = 0;
    auto x = context->GetInputShape(idx++);
    OP_CHECK_NULL_WITH_CONTEXT(context, x);
    auto shape = x->GetStorageShape();
    OP_CHECK_IF(shape.GetDimNum() != NUMBER_2,
                    OP_LOGE(context->GetNodeName(),
                    "QuantMatmulDequant get x shape ndim is not 2, please check."), return false);
    _Params.originM = (uint64_t)shape.GetDim(0);
    _Params.originK = (uint64_t)shape.GetDim(1);
    _Params.fracK = (_Params.originK + K_FRACTAL_INT8 -1) / K_FRACTAL_INT8;

    if (isGrouped) {
        auto quantized_weight = context->GetInputShape(idx++);
        OP_CHECK_NULL_WITH_CONTEXT(context, quantized_weight);
        shape = quantized_weight->GetStorageShape();
        OP_CHECK_IF(shape.GetDimNum() != NUMBER_5,
                        OP_LOGE(context->GetNodeName(),
                        "QuantMatmulDequant get quantized_weight shape ndim is not 5, please check."), return false);
        _Params.originE = (uint64_t)shape.GetDim(0);
        _Params.fracN = (uint64_t)shape.GetDim(NUMBER_2);
        OP_CHECK_IF((uint64_t)shape.GetDim(1) != _Params.fracK || shape.GetDim(NUMBER_3) != NM_FRACTAL_INT8 ||
                        shape.GetDim(NUMBER_4) != K_FRACTAL_INT8,
                        OP_LOGE(context->GetNodeName(),
                        "QuantMatmulDequant get quantized_weight shape not (G, K//32, N//16, 16, 32), please check."),
                        return false);

        auto weight_scale = context->GetInputShape(idx++);
        OP_CHECK_NULL_WITH_CONTEXT(context, weight_scale);
        shape = weight_scale->GetStorageShape();
        OP_CHECK_IF(shape.GetDimNum() != NUMBER_2,
                        OP_LOGE(context->GetNodeName(),
                        "QuantMatmulDequant get weight_scale shape ndim is not 1, please check."), return false);
        OP_CHECK_IF(_Params.originE != (uint64_t)shape.GetDim(0),
                      OP_LOGE(context->GetNodeName(),
                      "QuantMatmulDequant weight_scale shape[0] and quantized_weight shape[0] not equal please check."),
                      return false);
        _Params.originN = shape.GetDim(1);
        OP_CHECK_IF((_Params.originN + NM_FRACTAL_INT8 -1) / NM_FRACTAL_INT8 != _Params.fracN,
                  OP_LOGE(context->GetNodeName(),
                  "QuantMatmulDequant get n-dim of weight_scale and quantized_weight not match, please check."),
                  return false);

        auto group_list = context->GetInputShape(idx++);
        OP_CHECK_NULL_WITH_CONTEXT(context, group_list);
        shape = group_list->GetStorageShape();
        OP_CHECK_IF(shape.GetDimNum() != 1 || (uint64_t)shape.GetDim(0) != _Params.originE,
                        OP_LOGE(context->GetNodeName(),
                        "QuantMatmulDequant group_list shape[0] and quantized_weight shape[0] not equal please check."),
                        return false);
    } else {
        auto quantized_weight = context->GetInputShape(idx++);
        OP_CHECK_NULL_WITH_CONTEXT(context, quantized_weight);
        shape = quantized_weight->GetStorageShape();
        OP_CHECK_IF(shape.GetDimNum() != NUMBER_4,
                        OP_LOGE(context->GetNodeName(),
                        "QuantMatmulDequant get quantized_weight shape ndim is not 4, please check."), return false);
        _Params.fracN = (uint64_t)shape.GetDim(1);
        OP_CHECK_IF((uint64_t)shape.GetDim(0) != _Params.fracK || shape.GetDim(NUMBER_2) != NM_FRACTAL_INT8 ||
                        shape.GetDim(NUMBER_3) != K_FRACTAL_INT8,
                        OP_LOGE(context->GetNodeName(),
                        "QuantMatmulDequant get quantized_weight shape not (K//32, N//16, 16, 32), please check."),
                        return false);

        auto weight_scale = context->GetInputShape(idx++);
        OP_CHECK_NULL_WITH_CONTEXT(context, weight_scale);
        shape = weight_scale->GetStorageShape();
        OP_CHECK_IF(shape.GetDimNum() != 1,
                        OP_LOGE(context->GetNodeName(),
                        "QuantMatmulDequant get weight_scale shape ndim is not 1, please check."), return false);
        _Params.originN = (uint64_t)shape.GetDim(0);
        OP_CHECK_IF((_Params.originN + NM_FRACTAL_INT8 -1) / NM_FRACTAL_INT8 != _Params.fracN,
                  OP_LOGE(context->GetNodeName(),
                  "QuantMatmulDequant get n-dim of weight_scale and quantized_weight not match, please check."),
                  return false);
    }

    auto bias = context->GetOptionalInputShape(idx++);
    OP_CHECK_IF(bias != nullptr,
                OP_LOGE(context->GetNodeName(),
                "QuantMatmulDequant not support bias for now, please check."), return false);

    auto x_scale = context->GetOptionalInputShape(idx++);
    _Params.isXScaleHalf = 0;
    if (x_scale == nullptr) {
        OP_CHECK_IF(_Params.perToken != true,
                    OP_LOGE(context->GetNodeName(),
                    "QuantMatmulDequant only support pertoken in dynamic quant mode, please check."), return false);
        _Params.dynamicQuant = true;
    } else {
        shape = x_scale->GetStorageShape();
        if (_Params.perToken) {
            OP_CHECK_IF(shape.GetDimNum() != 1,
                            OP_LOGE(context->GetNodeName(),
                            "QuantMatmulDequant get x_scale shape ndim is not 1, please check."), return false);
            OP_CHECK_IF(_Params.originM != (uint64_t)shape.GetDim(0),
                            OP_LOGE(context->GetNodeName(),
                            "QuantMatmulDequant get m-dim of x_scale and x not match, please check."), return false);
        } else {
            OP_CHECK_IF(shape.GetDimNum() != 1 && shape.GetDimNum() != 0,
                            OP_LOGE(context->GetNodeName(),
                            "QuantMatmulDequant get x_scale shape ndim is not 1 or 0, please check."), return false);
            if (shape.GetDimNum() == 1) {
                OP_CHECK_IF(1 != (uint64_t)shape.GetDim(0),
                                OP_LOGE(context->GetNodeName(),
                                "QuantMatmulDequant get x_scale shape not [1], please check."), return false);
            }
        }
        _Params.dynamicQuant = false;

        auto x_scale_desc = context->GetOptionalInputDesc(idx-1);
        if (x_scale_desc != nullptr) {
            auto x_scale_dtype = x_scale_desc->GetDataType();
            if (x_scale_dtype == ge::DT_FLOAT16) {
                _Params.isXScaleHalf = 1;
            }
            OP_CHECK_IF(x_scale_dtype == ge::DT_FLOAT16 && _Params.perToken,
                  OP_LOGE(context->GetNodeName(),
                  "QuantMatmulDequant x_scale_dtype == ge::DT_FLOAT16 only support pertensor mode, please check."),
                  return false);
        }
    }

    auto x_offset = context->GetOptionalInputShape(idx++);
    OP_CHECK_IF(x_offset != nullptr,
                    OP_LOGE(context->GetNodeName(),
                    "QuantMatmulDequant not support x_offset for now, please check."), return false);

    auto smooth_scale = context->GetOptionalInputShape(idx++);
    if (smooth_scale == nullptr) {
        _Params.smoothScale = false;
    } else {
        shape = smooth_scale->GetStorageShape();
        OP_CHECK_IF(shape.GetDimNum() != 1,
                        OP_LOGE(context->GetNodeName(),
                        "QuantMatmulDequant get smooth_scale shape ndim is not 1, please check."), return false);
        OP_CHECK_IF(_Params.originK != (uint64_t)shape.GetDim(0),
                        OP_LOGE(context->GetNodeName(),
                        "QuantMatmulDequant get k-dim of smooth_scale and x not match, please check."), return false);
        _Params.smoothScale = true;
    }

    auto y = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, y);
    shape = y->GetStorageShape();
    OP_CHECK_IF(shape.GetDimNum() != NUMBER_2,
                    OP_LOGE(context->GetNodeName(),
                    "QuantMatmulDequant get y shape ndim is not 2, please check."), return false);
    OP_CHECK_IF(_Params.originM != (uint64_t)shape.GetDim(0),
                    OP_LOGE(context->GetNodeName(),
                    "QuantMatmulDequant get m-dim of y and x not match, please check."), return false);
    OP_CHECK_IF(_Params.originN != (uint64_t)shape.GetDim(1),
                    OP_LOGE(context->GetNodeName(),
                    "QuantMatmulDequant get m-dim of y and y_scale not match, please check."), return false);
    return true;
}

bool QuantMatmulDequantTiling::GetPlatformInfo(gert::TilingContext* context)
{
    auto compileInfoPtr = reinterpret_cast<const QuantMatmulDequantCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    _Params.CoreNum = compileInfoPtr->coreNum;
    _Params.UBSize = compileInfoPtr->ubSize;
    _Params.L1Size = compileInfoPtr->l1Size;
    _Params.L0ASize = compileInfoPtr->l0ASize;
    _Params.L0BSize = compileInfoPtr->l0BSize;
    _Params.L0CSize = compileInfoPtr->l0CSize;
    return true;
}

bool QuantMatmulDequantTiling::GetTilingData(gert::TilingContext* context)
{
    _Params.originKAligned512 = (_Params.originK + L0_ADDR_ALIGN - 1) / L0_ADDR_ALIGN * L0_ADDR_ALIGN;
    _Params.originKAligned32 = (_Params.originK + K_FRACTAL_INT8 - 1) / K_FRACTAL_INT8 * K_FRACTAL_INT8;
    // 最低的 0 ~ kTailN 位被设置为 1，其余位保持为 0
    _Params.ubKMask = 0;
    uint32_t kTailN = _Params.originKAligned32 - _Params.originK;
    for (uint32_t i = 0; i < K_FRACTAL_INT8; i++) {
        _Params.ubKMask = _Params.ubKMask << 1;
        if (i < kTailN) {
            _Params.ubKMask = (_Params.ubKMask | 1);
        }
    }
    OP_CHECK_IF(_Params.originK % FLOAT16_PERBLOCK != 0,
                    OP_LOGE(context->GetNodeName(),
                    "QuantMatmulDequant get k-dim not align to 16, please check."), return false);
    OP_CHECK_IF(_Params.originN % FLOAT16_PERBLOCK != 0,
                    OP_LOGE(context->GetNodeName(),
                    "QuantMatmulDequant get n-dim not align to 16, please check."), return false);
    OP_CHECK_IF(_Params.originM == 0,
                    OP_LOGE(context->GetNodeName(),
                    "QuantMatmulDequant get m-dim should not be 0, please check."), return false);

    if (isGrouped
        &&(SWIFT_M_MIN <= _Params.originM && _Params.originM <=SWIFT_M_MAX)
        &&(SWIFT_K_MIN <= _Params.originK && _Params.originK <=SWIFT_K_MAX)
        &&(SWIFT_N_MIN <= _Params.originN && _Params.originN <=SWIFT_N_MAX)
        &&(!_Params.perToken)
        &&(_Params.smoothScale)) {
        tilingKey = QuantMatmulDequantTilingKey::SWIFT;
        int32_t tmpThreshold = std::roundf((float)_Params.originK / (float)_Params.originN * GMM_GEMV_THRESHOLD_C1 +
                                            GMM_GEMV_THRESHOLD_C2);
        _Params.swiftGEMVThreshold = std::min(GMM_GMEV_THRESHOLD_MAX, std::max(GMM_GMEV_THRESHOLD_MIN, tmpThreshold));
        
        _Params.processXKBaseNMax = _Params.UBSize
                                / (NM_FRACTAL_INT8 * sizeof(int16_t) + NM_FRACTAL_INT8 * sizeof(int16_t) +
                                NM_FRACTAL_INT8 * sizeof(float) + (_Params.smoothScale ? (sizeof(int16_t) +
                                sizeof(float)) : 0))
                                / NUMBER_256 * NUMBER_256;
        _Params.processXKloopPerfracM = (_Params.originKAligned32 + _Params.processXKBaseNMax - 1) /
                                        _Params.processXKBaseNMax;
    
        // case2
        uint32_t fracMSwift = (SWIFT_M_MAX + NM_FRACTAL_INT8 - 1) / NM_FRACTAL_INT8;
        uint32_t chosen = 0;
        uint32_t mte2Min = (_Params.fracN << NUMBER_3) + (fracMSwift << 0);
        for (int i = 1; i < NUMBER_4; i++) {
            uint32_t mte2Now = (_Params.fracN << (NUMBER_3-i)) + (fracMSwift << i);
            if (mte2Now < mte2Min) {
                chosen = i;
                mte2Min = mte2Now;
            }
        }
        _Params.MCoreNum = 1 << (NUMBER_3-chosen);
        _Params.NCoreNum = 1 << chosen;
        uint32_t singleCoreM = (fracMSwift + _Params.MCoreNum - 1) / _Params.MCoreNum;
        uint32_t singleCoreN = (_Params.fracN + _Params.NCoreNum - 1) / _Params.NCoreNum;
        uint32_t singleCoreNTail = (_Params.fracN-1) % _Params.NCoreNum + 1;
        uint32_t singleCoreMNFractal = (singleCoreM) * (singleCoreN);
        uint32_t l0CMNFractal = NUMBER_256 - NUMBER_16;
        _Params.baseMNum = 1;
        _Params.baseNNum_2 = 1;
        int32_t baseM = 1;
        int32_t baseN = 1;
        if (singleCoreMNFractal > l0CMNFractal) {
            float MNPieces = (float)singleCoreMNFractal / (float)l0CMNFractal;
            float MDivN = (float)singleCoreM / (float)singleCoreN;
            float tmp = std::sqrt(MNPieces/ MDivN);
            if (tmp == 0) {
                return false;
            }
            baseM = static_cast<uint32_t>(floor((float)singleCoreM / (MDivN*tmp)));
            baseN = static_cast<uint32_t>(floor((float)singleCoreN / tmp));

            _Params.baseMNum = (singleCoreM + baseM  - 1) / baseM;
            _Params.baseNNum_2 = (singleCoreN + baseN - 1) / baseN;
            if (singleCoreM > singleCoreN) {
                uint32_t BaseMFractalN_ = (singleCoreM + _Params.baseMNum - 1) / _Params.baseMNum + 1;
                while (_Params.baseNNum_2 > 1 && BaseMFractalN_ * ((singleCoreN + _Params.baseNNum_2 - NUMBER_2) /
                    (_Params.baseNNum_2 - 1)) <= l0CMNFractal) {
                    _Params.baseNNum_2 -= 1;
                }
                uint32_t BaseNFractalN_ = (singleCoreN + _Params.baseNNum_2 - 1) / _Params.baseNNum_2;
                while (_Params.baseMNum > 1 && BaseNFractalN_ * ((singleCoreM + _Params.baseMNum - NUMBER_2) /
                    (_Params.baseMNum - 1)) <= l0CMNFractal) {
                    _Params.baseMNum -= 1;
                }
            } else {
                uint32_t BaseNFractalN_ = (singleCoreN + _Params.baseNNum_2 - 1) / _Params.baseNNum_2 ;
                while (_Params.baseMNum > 1 && BaseNFractalN_ * ((singleCoreM + _Params.baseMNum - NUMBER_2) /
                    (_Params.baseMNum - 1)) <= l0CMNFractal) {
                    _Params.baseMNum -= 1;
                }
                uint32_t BaseMFractalN_ = (singleCoreM + _Params.baseMNum - 1) / _Params.baseMNum + 1;
                while (_Params.baseNNum_2 > 1 && BaseMFractalN_ * ((singleCoreN + _Params.baseNNum_2 - NUMBER_2) /
                    (_Params.baseNNum_2 - 1)) <= l0CMNFractal) {
                    _Params.baseNNum_2 -= 1;
                }
            }
        }

        uint32_t BaseMNumL0AB = (singleCoreM + NUMBER_128 / NUMBER_2 - 1) / (NUMBER_128 / NUMBER_2);
        uint32_t BaseNNumL0AB = (singleCoreN + NUMBER_128 / NUMBER_2 - 1) / (NUMBER_128 / NUMBER_2);
        if (_Params.baseMNum < BaseMNumL0AB) _Params.baseMNum = BaseMNumL0AB;
        if (_Params.baseNNum_2 < BaseNNumL0AB) _Params.baseNNum_2 = BaseNNumL0AB;

        baseM = (singleCoreM + _Params.baseMNum - 1) / _Params.baseMNum;
        _Params.baseN_2 = (singleCoreN + _Params.baseNNum_2 - 1) / _Params.baseNNum_2;

        uint32_t l0AMKFractal = NUMBER_128 / NUMBER_2 / baseM;
        uint32_t l0BNKFractal = NUMBER_128 / NUMBER_2 / _Params.baseN_2;
        uint32_t MKPieces = (_Params.fracK + l0AMKFractal - 1) / l0AMKFractal;
        uint32_t NKPieces = (_Params.fracK + l0BNKFractal - 1) / l0BNKFractal;
        _Params.baseKNum_2 = MKPieces > NKPieces ? MKPieces : NKPieces;
        _Params.baseK_2 = (_Params.fracK + _Params.baseKNum_2 - 1) / _Params.baseKNum_2;
        _Params.baseKTail_2 = (_Params.fracK - 1) % _Params.baseKNum_2 + 1;

        fracMSwift = (SWIFT_M_MIN + NM_FRACTAL_INT8 - 1) / NM_FRACTAL_INT8;
        chosen = 0;
        mte2Min = (_Params.fracN << NUMBER_3) + (fracMSwift << 0);
        for (int i = 1; i < NUMBER_4; i++) {
            uint32_t mte2Now = (_Params.fracN << (NUMBER_3-i)) + (fracMSwift << i);
            if (mte2Now < mte2Min) {
                chosen = i;
                mte2Min = mte2Now;
            }
        }
        _Params.MCoreNum = 1 << (NUMBER_3-chosen);
        _Params.NCoreNum = 1 << chosen;
        singleCoreM = (fracMSwift + _Params.MCoreNum - 1) / _Params.MCoreNum;
        singleCoreN = (_Params.fracN + _Params.NCoreNum - 1) / _Params.NCoreNum;
        singleCoreNTail = (_Params.fracN-1) % _Params.NCoreNum + 1;
        singleCoreMNFractal = (singleCoreM) * (singleCoreN);
        l0CMNFractal = NUMBER_256 - NUMBER_16;
        _Params.baseMNum = 1;
        _Params.baseNNum = 1;

        BaseMNumL0AB = (singleCoreM + NUMBER_128 / NUMBER_2 - 1) / (NUMBER_128 / NUMBER_2);
        BaseNNumL0AB = (singleCoreN + NUMBER_128 / NUMBER_2 - 1) / (NUMBER_128 / NUMBER_2);
        if (_Params.baseMNum < BaseMNumL0AB) _Params.baseMNum = BaseMNumL0AB;
        if (_Params.baseNNum < BaseNNumL0AB) _Params.baseNNum = BaseNNumL0AB;

        baseM = (singleCoreM + _Params.baseMNum - 1) / _Params.baseMNum;
        baseN = (singleCoreN + _Params.baseNNum - 1) / _Params.baseNNum;

        l0AMKFractal = NUMBER_128 / NUMBER_2 / baseM;
        l0BNKFractal = NUMBER_128 / NUMBER_2 / baseN;
        MKPieces = (_Params.fracK + l0AMKFractal - 1) / l0AMKFractal;
        NKPieces = (_Params.fracK + l0BNKFractal - 1) / l0BNKFractal;
        _Params.baseKNum = MKPieces > NKPieces ? MKPieces : NKPieces;
        _Params.baseK = (_Params.fracK + _Params.baseKNum - 1) / _Params.baseKNum;
        _Params.baseKTail = (_Params.fracK - 1) % _Params.baseKNum + 1;
        _Params.singleCoreN = singleCoreN;
        _Params.singleCoreNTail = singleCoreNTail;

        _Params.singleCoreFracN = (_Params.fracN + _Params.CoreNum - 1) / _Params.CoreNum;
        _Params.singleCoreFracNTail = (_Params.fracN - 1) % _Params.CoreNum + 1;
        _Params.baseFracK = NUMBER_16;
        _Params.baseFracN = NUMBER_4;
    } else if (isGrouped) {
        tilingKey = QuantMatmulDequantTilingKey::GROUPED;
        _Params.processXKBaseNMax = (_Params.UBSize - (_Params.perToken ? (NM_FRACTAL_INT8 * BLOCK_SIZE) : 0))
                                / (NM_FRACTAL_INT8 * sizeof(int16_t) + NM_FRACTAL_INT8 * sizeof(int16_t) +
                                NM_FRACTAL_INT8 * sizeof(float) + (_Params.smoothScale ? (sizeof(int16_t) +
                                sizeof(float)) : 0))
                                / NUMBER_256 * NUMBER_256;
        _Params.singleCoreFracN = (_Params.fracN + _Params.CoreNum - 1) / _Params.CoreNum;
        _Params.singleCoreFracNTail = (_Params.fracN - 1) % _Params.CoreNum + 1;
        _Params.baseFracK = NUMBER_16;
        _Params.baseFracN = NUMBER_4;
    } else if (_Params.originM > GEMV_THRESHOLD || (_Params.originKAligned512 * _Params.originM) >
            (_Params.L1Size / NUMBER_2)) {
        tilingKey = QuantMatmulDequantTilingKey::NORMAL;
        _Params.fracM = (_Params.originM + NM_FRACTAL_INT8 - 1) / NM_FRACTAL_INT8;
        _Params.tailM = (_Params.originM - 1) % NM_FRACTAL_INT8 + 1;

        _Params.processXKBaseNMax = (_Params.UBSize - (_Params.perToken ? (NM_FRACTAL_INT8 * BLOCK_SIZE) : 0))
                                    / (NM_FRACTAL_INT8 * sizeof(int16_t) + NM_FRACTAL_INT8 * sizeof(int16_t) +
                                    NM_FRACTAL_INT8 * sizeof(float) + (_Params.smoothScale ? (sizeof(int16_t) +
                                    sizeof(float)) : 0))
                                    / NUMBER_256 * NUMBER_256;
        _Params.processXKloopPerfracM = (_Params.originKAligned32 + _Params.processXKBaseNMax - 1) /
                                        _Params.processXKBaseNMax;
        _Params.processXKloop = (_Params.fracM * _Params.processXKloopPerfracM + _Params.CoreNum - 1) /
                                _Params.CoreNum;
        _Params.processXKloopPerfracM = _Params.processXKloop * _Params.CoreNum / _Params.fracM;
        _Params.processXKBaseN = (_Params.fracK + _Params.processXKloopPerfracM - 1) /
                                _Params.processXKloopPerfracM * K_FRACTAL_INT8;
        _Params.processXKTailN = _Params.fracK % _Params.processXKloopPerfracM;
        _Params.processXKTailN = _Params.processXKTailN == 0 ? _Params.processXKloopPerfracM : _Params.processXKTailN;

        uint32_t chosen = 0;
        uint32_t mte2Min = (_Params.fracN << NUMBER_3) + (_Params.fracM << 0);
        for (int i = 1;i < NUMBER_4; i++) {
            uint32_t mte2Now = (_Params.fracN << (NUMBER_3-i)) + (_Params.fracM << i);
            if (mte2Now < mte2Min) {
                chosen = i;
                mte2Min = mte2Now;
            }
        }
        _Params.MCoreNum = 1 << (NUMBER_3-chosen);
        _Params.NCoreNum = 1 << chosen;
        _Params.singleCoreM = (_Params.fracM + _Params.MCoreNum - 1) / _Params.MCoreNum;
        _Params.singleCoreN = (_Params.fracN + _Params.NCoreNum - 1) / _Params.NCoreNum;
        _Params.singleCoreMTail = (_Params.fracM-1) % _Params.MCoreNum + 1;
        _Params.singleCoreNTail = (_Params.fracN-1) % _Params.NCoreNum + 1;
        uint32_t l0CMNFractal = NUMBER_256;
        uint32_t oriBaseMN = static_cast<uint32_t>(floor(std::sqrt((float)l0CMNFractal)));
        uint32_t extraN = _Params.perToken ? 1 : 0;
        uint32_t singleCoreMNFractal = (_Params.singleCoreM + 1) * (_Params.singleCoreN + extraN);
        _Params.baseMNum = 1;
        _Params.baseNNum = 1;
        if (singleCoreMNFractal > l0CMNFractal) {
            _Params.baseMNum = (_Params.singleCoreM + (oriBaseMN - 1)  - 1) / (oriBaseMN - 1);
            _Params.baseNNum = (_Params.singleCoreN + (oriBaseMN - extraN) - 1) / (oriBaseMN - extraN);
            if (_Params.singleCoreM > _Params.singleCoreN) {
                uint32_t BaseMFractalN_ = (_Params.singleCoreM + _Params.baseMNum - 1) / _Params.baseMNum + 1;
                while (_Params.baseNNum > 1 && BaseMFractalN_ * ((_Params.singleCoreN + _Params.baseNNum - NUMBER_2) /
                    (_Params.baseNNum - 1) + extraN) <= l0CMNFractal) {
                    _Params.baseNNum -= 1;
                }
                uint32_t BaseNFractalN_ = (_Params.singleCoreN + _Params.baseNNum - 1) / _Params.baseNNum + extraN;
                while (_Params.baseMNum > 1 && BaseNFractalN_ * ((_Params.singleCoreM + _Params.baseMNum - NUMBER_2) /
                    (_Params.baseMNum - 1) + 1) <= l0CMNFractal) {
                    _Params.baseMNum -= 1;
                }
            } else {
                uint32_t BaseNFractalN_ = (_Params.singleCoreN + _Params.baseNNum - 1) / _Params.baseNNum + extraN;
                while (_Params.baseMNum > 1 && BaseNFractalN_ * ((_Params.singleCoreM + _Params.baseMNum - NUMBER_2) /
                    (_Params.baseMNum - 1) + 1) <= l0CMNFractal) {
                    _Params.baseMNum -= 1;
                }
                uint32_t BaseMFractalN_ = (_Params.singleCoreM + _Params.baseMNum - 1) / _Params.baseMNum + 1;
                while (_Params.baseNNum > 1 && BaseMFractalN_ * ((_Params.singleCoreN + _Params.baseNNum - NUMBER_2) /
                    (_Params.baseNNum - 1) + extraN) <= l0CMNFractal) {
                    _Params.baseNNum -= 1;
                }
            }
        }

        uint32_t BaseMNumL0AB = (_Params.singleCoreM + NUMBER_128 / NUMBER_2 - 1) / (NUMBER_128 / NUMBER_2);
        uint32_t BaseNNumL0AB = (_Params.singleCoreN + NUMBER_128 / NUMBER_2 - 1) / (NUMBER_128 / NUMBER_2);
        if (_Params.baseMNum < BaseMNumL0AB) _Params.baseMNum = BaseMNumL0AB;
        if (_Params.baseNNum < BaseNNumL0AB) _Params.baseNNum = BaseNNumL0AB;

        uint32_t baseM = (_Params.singleCoreM + _Params.baseMNum - 1) / _Params.baseMNum;
        uint32_t baseN = (_Params.singleCoreN + _Params.baseNNum - 1) / _Params.baseNNum;
        uint32_t l0AMKFractal = NUMBER_128 / NUMBER_2 / baseM;
        uint32_t l0BNKFractal = NUMBER_128 / NUMBER_2 / baseN;
        uint32_t MKPieces = (_Params.fracK + l0AMKFractal - 1) / l0AMKFractal;
        uint32_t NKPieces = (_Params.fracK + l0BNKFractal - 1) / l0BNKFractal;
        _Params.baseKNum = MKPieces > NKPieces ? MKPieces : NKPieces;
    } else {
        tilingKey = QuantMatmulDequantTilingKey::GEMV;
        _Params.singleCoreFracN = (_Params.fracN + _Params.CoreNum - 1) / _Params.CoreNum;
        _Params.singleCoreFracNTail = (_Params.fracN - 1) % _Params.CoreNum + 1;
        _Params.baseFracK = NUMBER_16;
        _Params.baseFracN = NUMBER_4;
        _Params.baseFracNL0C = _Params.UBSize / sizeof(int32_t) / L0C_FRACTAL / L0C_FRACTAL / _Params.originM;
        _Params.ubBaseK = (_Params.UBSize - (_Params.perToken ? _Params.originM * BLOCK_SIZE : 0))
                            / (_Params.originM * NUMBER_3 + NUMBER_4 +
                                (_Params.smoothScale ? (NUMBER_2 + NUMBER_4) : 0))
                            / NUMBER_256 * NUMBER_256;
        _Params.ubIterK = (_Params.originK + _Params.ubBaseK - 1) / _Params.ubBaseK;
        _Params.ubBaseKTail = (_Params.originK - 1) % _Params.ubBaseK + 1;
    }

    if (_Params.dynamicQuant) {
        _Params.dynamicBaseK = (_Params.UBSize / NUMBER_2 - BLOCK_SIZE * NUMBER_2 - BLOCK_SIZE * NUMBER_2) /
                               (1 + (_Params.smoothScale ? 1 : 0)) / NUMBER_2 / NUMBER_256 * NUMBER_256;
        _Params.dynamicBaseK = _Params.dynamicBaseK > REDUCE_THRESHOLD ? REDUCE_THRESHOLD : _Params.dynamicBaseK;
        _Params.dynamicIterK = (_Params.originK + _Params.dynamicBaseK - 1) / _Params.dynamicBaseK;
        _Params.dynamicBaseKTail = (_Params.originK - 1) % _Params.dynamicBaseK + 1;
    }

    return true;
}

bool QuantMatmulDequantTiling::SetTilingData(gert::TilingContext* context)
{
    tilingData.set_CoreNum(_Params.CoreNum);
    tilingData.set_perToken(_Params.perToken);
    tilingData.set_dynamicQuant(_Params.dynamicQuant);
    tilingData.set_smoothScale(_Params.smoothScale);
    tilingData.set_originE(_Params.originE);
    tilingData.set_originM(_Params.originM);
    tilingData.set_originN(_Params.originN);
    tilingData.set_originK(_Params.originK);
    tilingData.set_originKAligned32(_Params.originKAligned32);
    tilingData.set_originKAligned512(_Params.originKAligned512);
    tilingData.set_fracN(_Params.fracN);
    tilingData.set_fracK(_Params.fracK);
    // dynamic
    tilingData.set_dynamicBaseK(_Params.dynamicBaseK);
    tilingData.set_dynamicIterK(_Params.dynamicIterK);
    tilingData.set_dynamicBaseKTail(_Params.dynamicBaseKTail);
    // gemv
    tilingData.set_singleCoreFracN(_Params.singleCoreFracN);
    tilingData.set_singleCoreFracNTail(_Params.singleCoreFracNTail);
    tilingData.set_baseFracN(_Params.baseFracN);
    tilingData.set_baseFracK(_Params.baseFracK);
    tilingData.set_baseFracNL0C(_Params.baseFracNL0C);
    tilingData.set_ubBaseK(_Params.ubBaseK);
    tilingData.set_ubIterK(_Params.ubIterK);
    tilingData.set_ubBaseKTail(_Params.ubBaseKTail);
    // normal
    tilingData.set_fracM(_Params.fracM);
    tilingData.set_tailM(_Params.tailM);
    tilingData.set_processXKBaseNMax(_Params.processXKBaseNMax);
    tilingData.set_processXKBaseN(_Params.processXKBaseN);
    tilingData.set_processXKloop(_Params.processXKloop);
    tilingData.set_processXKloopPerfracM(_Params.processXKloopPerfracM);
    tilingData.set_processXKTailN(_Params.processXKTailN);
    tilingData.set_MMmod(_Params.MMmod);
    tilingData.set_MCoreNum(_Params.MCoreNum);
    tilingData.set_NCoreNum(_Params.NCoreNum);
    tilingData.set_singleCoreM(_Params.singleCoreM);
    tilingData.set_singleCoreN(_Params.singleCoreN);
    tilingData.set_singleCoreMTail(_Params.singleCoreMTail);
    tilingData.set_singleCoreNTail(_Params.singleCoreNTail);
    tilingData.set_baseMNum(_Params.baseMNum);
    tilingData.set_baseNNum(_Params.baseNNum);
    tilingData.set_baseKNum(_Params.baseKNum);
    // swift
    tilingData.set_swiftGEMVThreshold(_Params.swiftGEMVThreshold);
    tilingData.set_baseNNum_2(_Params.baseNNum_2);
    tilingData.set_baseKNum_2(_Params.baseKNum_2);
    tilingData.set_baseK(_Params.baseK);
    tilingData.set_baseKTail(_Params.baseKTail);
    tilingData.set_baseK_2(_Params.baseK_2);
    tilingData.set_baseKTail_2(_Params.baseKTail_2);
    tilingData.set_ubKMask(_Params.ubKMask);
    tilingData.set_isXScaleHalf(_Params.isXScaleHalf);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return true;
}

bool QuantMatmulDequantTiling::SetLaunchInfo(gert::TilingContext* context)
{
    context->SetBlockDim(_Params.CoreNum);

    context->SetTilingKey(static_cast<uint64_t>(tilingKey));

    int64_t workspaceSize = SYS_WORKSPACE_310P
                            + _Params.CoreNum * BLOCK_SIZE
                            + ((_Params.dynamicQuant && _Params.perToken) ? _Params.originM * BLOCK_SIZE : 0);
    if (tilingKey != QuantMatmulDequantTilingKey::GEMV) {
        workspaceSize += (_Params.originM + NM_FRACTAL_INT8 - 1) / NM_FRACTAL_INT8 * NM_FRACTAL_INT8 *
                          _Params.originKAligned32;
    }
    AddWorkspace(context, workspaceSize);
    return true;
}

bool QuantMatmulDequantTiling::GetCheckAttr(gert::TilingContext* context)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    //  optional
    const std::string x_quant_mode = static_cast<std::string>(attrs->GetStr(0));
    const bool *transpose_weight = attrs->GetAttrPointer<bool>(1);

    OP_CHECK_IF(x_quant_mode != "pertoken" && x_quant_mode != "pertensor",
                    OP_LOGE(context->GetNodeName(),
                    "QuantMatmulDequant attr x_quant_mode only support \"pertoken\" or \"pertensor\", please check."),
                    return false);
    OP_CHECK_IF(*transpose_weight == false,
                    OP_LOGE(context->GetNodeName(),
                    "QuantMatmulDequant attr transpose_weight only support true, please check."), return false);

    _Params.perToken = x_quant_mode == "pertoken" ? true : false;
    return true;
}

ge::graphStatus QuantMatmulDequantTiling::runTiling(gert::TilingContext* context, bool is_grouped)
{
    isGrouped = is_grouped;
    // 310P AscendC platformINFO
    OP_CHECK_IF(!GetPlatformInfo(context),
                    OP_LOGE(context->GetNodeName(), "get platforminfo fail."),
                    return ge::GRAPH_FAILED);
    // attr
    OP_CHECK_IF(!GetCheckAttr(context),
                    OP_LOGE(context->GetNodeName(), "check attr fail."),
                    return ge::GRAPH_FAILED);
    // shape
    OP_CHECK_IF(!CheckInOutShapes(context),
                    OP_LOGE(context->GetNodeName(), "check shape fail."),
                    return ge::GRAPH_FAILED);
    // calculate tilingdata
    OP_CHECK_IF(!GetTilingData(context),
                    OP_LOGE(context->GetNodeName(), "get tiling data fail."),
                    return ge::GRAPH_FAILED);

    // tilingdata
    OP_CHECK_IF(!SetTilingData(context),
                    OP_LOGE(context->GetNodeName(), "set tiling data fail."),
                    return ge::GRAPH_FAILED);

    // launchinfo: tilingkey, workspace, blockdim
    OP_CHECK_IF(!SetLaunchInfo(context),
                    OP_LOGE(context->GetNodeName(), "set launchinfo fail."),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForQuantGroupedMatmulDequant(gert::TilingContext* context)
{
    QuantMatmulDequantTiling tiling_handle;
    context->SetScheduleMode(BATCH_MODE);
    return tiling_handle.runTiling(context, true);
}

ge::graphStatus TilingPrepareForQuantGroupedMatmulDequant(gert::TilingParseContext* context)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto compileInfoPtr = context->GetCompiledInfo<QuantMatmulDequantCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);

    std::string val;
    platformInfoPtr->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_fix_pipe_l0c2out", val);
    OP_CHECK_IF(!val.empty(), OP_LOGE(context->GetNodeName(),
                    "QuantMatmulDequant support only ASCEND310P for now"), return false);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    platformInfoPtr->GetPlatformRes("version", "SoC_version", compileInfoPtr->socVersionStr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNum();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0ASize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0BSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0CSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfoPtr->l2Size);

    OP_CHECK_IF((compileInfoPtr->coreNum == 0 || compileInfoPtr->ubSize == 0 || \
                    compileInfoPtr->l1Size == 0 || compileInfoPtr->l0CSize == 0 || compileInfoPtr->l0ASize == 0 || \
                    compileInfoPtr->l0BSize == 0),
                    OP_LOGE(context->GetNodeName(),
                    "platform invalid, coreNum=%u, ubSize=%lu, l1Size=%lu, l0CSize=%lu, l0ASize=%lu, l0BSize=%lu",
                    compileInfoPtr->coreNum, compileInfoPtr->ubSize, compileInfoPtr->l1Size,
                    compileInfoPtr->l0CSize, compileInfoPtr->l0ASize, compileInfoPtr->l0BSize),
                    return ge::GRAPH_FAILED);

    OP_LOGI(context->GetNodeName(), "Parse compile info success, soc: %d",
            static_cast<int>(compileInfoPtr->socVersion));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(QuantGroupedMatmulDequant).Tiling(TilingForQuantGroupedMatmulDequant).TilingParse<QuantMatmulDequantCompileInfo>(TilingPrepareForQuantGroupedMatmulDequant);

}  // namespace optiling
