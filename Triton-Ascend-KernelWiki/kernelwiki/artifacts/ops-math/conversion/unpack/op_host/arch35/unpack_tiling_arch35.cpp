/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "unpack_tiling_arch35.h"
#include "op_host/tiling_base_util.h"
#include "register/op_impl_registry.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "conversion/split_v/op_host/arch35/split_v_tiling_arch35.h"

using namespace ge;

namespace optiling {
constexpr int64_t SPLIT_DIM_INDEX_SAME_LEN = 0;

graphStatus TilingPrepareForUnpack(gert::TilingParseContext* context) {
  auto compile_info = context->GetCompiledInfo<UnpackCompileInfo>();
  OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
    OP_LOGD(context->GetNodeName(), "AscendC Tiling Starting!");
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compile_info->maxCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compile_info->maxCoreNum <= 0),
                    OP_LOGE(context->GetNodeName(), "The core num is invalid."),
                    return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compile_info->ubSizePlatform = static_cast<uint32_t>(ubSize);
    OP_CHECK_IF((compile_info->ubSizePlatform <= 0),
                    OP_LOGE(context->GetNodeName(), "The ubSize is invalid."),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UnpackTiling::GetInputParamsSameLen() {
  auto xInput = context_->GetInputShape(0);
  OP_CHECK_NULL_WITH_CONTEXT(context_, xInput);
  const gert::Shape& xInputShape = Ops::Base::EnsureNotScalar(xInput->GetStorageShape());
  auto attrs = context_->GetAttrs();
  OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
  auto attr0 = attrs->GetInt(0);
  OP_CHECK_NULL_WITH_CONTEXT(context_, attr0);
  auto attr1 = attrs->GetInt(1);
  OP_CHECK_NULL_WITH_CONTEXT(context_, attr1);
  inputShape_ = xInputShape;
  numSplit_ = *attr0;
  splitDim_ = *attr1;
  int64_t inputXDimNum = static_cast<int64_t>(inputShape_.GetDimNum());

  OP_CHECK_IF((splitDim_ < -inputXDimNum || splitDim_ >= inputXDimNum),
      OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "axis",
          std::to_string(splitDim_).c_str(),
          ("The value of axis must be within the range [-" + std::to_string(inputXDimNum) + 
           ", " + std::to_string(inputXDimNum - 1) + "].").c_str()), 
      return ge::GRAPH_FAILED);
  if (splitDim_ < 0) {
      splitDim_ = splitDim_ + inputXDimNum;
  }

  OP_CHECK_IF(numSplit_ <= 0,
      OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "num",
          std::to_string(numSplit_).c_str(),
          "The value of num must be greater than 0."),
      return ge::GRAPH_FAILED);

  const int64_t dim = inputShape_.GetDim(splitDim_);
  OP_CHECK_IF(numSplit_ != dim, 
      OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "num",
          std::to_string(numSplit_).c_str(),
          std::to_string(dim).c_str()),
      return GRAPH_FAILED);

  if (numSplit_ == 1) {
    // 如果不切分, 则把splitDim设置成0, 提高搬运效率
    splitDim_ = 0;
  }
  return ge::GRAPH_SUCCESS;
}


ge::graphStatus UnpackTiling::InitParamsSameLen(int32_t maxCoreNum, uint32_t ubSize){
  coreNum_ = std::min(maxCoreNum, static_cast<int32_t>(MAX_CORE_COUNT));
  ubSize_ = ubSize;

  auto xInputDesc = context_->GetInputDesc(0);
  OP_CHECK_NULL_WITH_CONTEXT(context_, xInputDesc);
  ge::DataType xDtype = xInputDesc->GetDataType();
  xDtypeSize_ = ge::GetSizeByDataType(xDtype);
OP_CHECK_IF(xDtypeSize_ <= 0,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "x",
                    Ops::Base::ToString(xDtype).c_str(),
                    "The dtype size of x must be greater than 0."),
                return ge::GRAPH_FAILED);

  if (GetInputParamsSameLen() != ge::GRAPH_SUCCESS) {
      return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForUnpack(gert::TilingContext* context) {
  UnpackTiling tilingUnpack(context);
  OP_LOGD(context->GetNodeName(), "begin to do TilingForUnpack");
  auto compile_info = static_cast<const UnpackCompileInfo*>(context->GetCompileInfo());

    OP_LOGD(context->GetNodeName(), "DSL/TIK Tiling4Unpack is Null! Running AscendC tiling!");
    uint32_t maxCoreNum = compile_info->maxCoreNum;
    uint32_t ubSizePlatform = compile_info->ubSizePlatform;
OP_CHECK_IF(ubSizePlatform <= 0U, OP_LOGE(context->GetNodeName(),
                    "get ubSize <= 0 error"),
                    return GRAPH_FAILED);

    OP_CHECK_IF(tilingUnpack.DoSplitTiling(maxCoreNum, ubSizePlatform) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(),
                    "AscendC split tiling function call failed"), 
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Unpack).Tiling(TilingForUnpack)
               .TilingParse<UnpackCompileInfo>(TilingPrepareForUnpack);
}  // namespace optiling