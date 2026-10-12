/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
* \file strided_slice_v2_tiling.cc
* \brief
*/
#include "strided_slice_v2_tiling_arch35.h"
#include <numeric>
#include "common/inc/op_host/math_log.h"

namespace {
const std::string OP_NAME = "StridedSliceV2";
const int INDEX_X = 0;
const int INDEX_BEGIN = 1;
const int INDEX_END = 2;
const int INDEX_AXES = 3;
const int INDEX_STRIDES = 4;
const int INDEX_Y = 0;
static const size_t IDX_MASK_BEGIN = 0;
static const size_t IDX_MASK_END = 1;
static const size_t IDX_MASK_ELLIPSIS = 2;
static const size_t IDX_MASK_NEW_AXIS = 3;
static const size_t IDX_MASK_SHRINK_AXIS = 4;
}  // namespace

namespace optiling {

static void ConstructSliceShape(const gert::StorageShape *storage, gert::Shape &param) {
   const gert::Shape &shape = Ops::Base::EnsureNotScalar(storage->GetStorageShape());
   int32_t dimNum = static_cast<int32_t>(shape.GetDimNum());
   param.SetDimNum(dimNum);
   for (int32_t i = 0; i < dimNum; i++) {
       param[i] = shape.GetDim(i);
   }
}

static bool CheckStride(ops::QuickVector &stride, const gert::TilingContext *context) {
    auto compileInfo = context->GetCompileInfo<StridedSliceV2CompileInfo>();
   OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

   for (size_t i = 0; i < stride.GetDimNum(); i++) {
       if (stride[i] == 0) {
           OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
               OP_NAME, "stride", "0",
               "The value of stride cannot be 0.");
           return false;
       }
   }
   return true;
}

static void MakeSameDims(SliceParametersRuntime2 *parametersPtr) {
   auto &parameters = *parametersPtr;
   bool sameSize = parameters.inputShape.GetDimNum() == parameters.beginList.GetDimNum() &&
                   parameters.inputShape.GetDimNum() == parameters.endList.GetDimNum() &&
                   parameters.inputShape.GetDimNum() == parameters.strideList.GetDimNum();
   if (!sameSize) {
       return;
   }

   parameters.outputShape.SetDimNum(0);
   for (size_t i = 0; i < parameters.inputShape.GetDimNum(); i++) {
       auto interval = parameters.endList[i] - parameters.beginList[i];
       auto strideI = parameters.strideList[i];
       if (strideI == 0) {
           strideI = 1;
       }
       int64_t outputSize = interval / strideI + (interval % strideI != 0 ? 1 : 0);
       parameters.outputShape.AppendDim(outputSize);
   }
}

template <typename T>
static void PositiveAxisImpl(int32_t inputDims, const gert::Tensor* axesTensor, std::vector<int64_t>& newAxes) {
   int32_t axesSize = static_cast<int32_t>(axesTensor->GetShapeSize());
   const T* data = axesTensor->GetData<T>();
   if (data == nullptr) {
       OP_LOGE(OP_NAME, "Failed to get tensor data, data is null.");
       return;
   }

   for (int32_t i = 0; i < axesSize; i++) {
       int32_t value = static_cast<int32_t>(data[i]);
       if (value >= 0 && value < inputDims) {
           newAxes.push_back(value);
       } else if (value < 0 && value >= -inputDims) {
           newAxes.push_back(value + inputDims);
       }
   }
}

static std::vector<int64_t> ConstructValidAxis(const gert::Tensor* axesTensor, int32_t inputDims) {
   std::vector<int64_t> newAxes;
   if (!axesTensor || axesTensor->GetShapeSize() == 0) {
       newAxes.resize(inputDims);
       std::iota(newAxes.begin(), newAxes.end(), 0);
       return newAxes;
   }

   if (axesTensor->GetDataType() == ge::DT_INT32) {
       PositiveAxisImpl<int32_t>(inputDims, axesTensor, newAxes);
   } else {
       PositiveAxisImpl<int64_t>(inputDims, axesTensor, newAxes);
   }
   return newAxes;
}

static int64_t GetConstIndexValue(const gert::Tensor* tensor, int32_t idx, int64_t defaultValue = 0) {
   if (!tensor) {
       OP_LOGE(OP_NAME, "Tensor is null.");
       return defaultValue;
   }

   int64_t value = defaultValue;
   const auto dataType = tensor->GetDataType();
   if (dataType == ge::DT_INT32) {
       const int32_t* data = tensor->GetData<int32_t>();
       if (data == nullptr) {
           OP_LOGE(OP_NAME, "Failed to get tensor data, data is null.");
           return defaultValue;
       }
       value = static_cast<int64_t>(data[idx]);
   } else if (dataType == ge::DT_INT64) {
       const int64_t* data = tensor->GetData<int64_t>();
       if (data == nullptr) {
           OP_LOGE(OP_NAME, "Failed to get tensor data, data is null.");
           return defaultValue;
       }
       value = data[idx];
   } else {
       OP_LOGE_FOR_INVALID_DTYPE(
            OP_NAME, "end, begin or stride", ge::TypeUtils::DataTypeToSerialString(dataType).c_str(),
            "int32 or int64");
       return defaultValue;
   }

   OP_LOGD(OP_NAME.c_str(), "const tensor[%d] is %ld.", idx, value);
   return value;
}

static void InitListWithDimNum(ops::QuickVector& list, int32_t dimNum, int64_t initValue = 0) {
   for (int32_t i = 0; i < dimNum; i++) {
       list.AppendDim(initValue);
   }
}

static void ConstructStrideList(const gert::Tensor* strideTensor, int32_t dimNum,
                               const std::vector<int64_t>& axes, ops::QuickVector& list_strides) {
   // Initialize all strides to 1
   InitListWithDimNum(list_strides, dimNum, 1);

   if (!strideTensor) {
       OP_LOGD(OP_NAME.c_str(), "Stride tensor is null. Set stride as 1.");
       return;
   }

   // Update list_strides with const value of strideTensor
   const int32_t strideSize = static_cast<int32_t>(strideTensor->GetShapeSize());
   const int32_t axesSize = static_cast<int32_t>(axes.size());

   for (int32_t i = 0; i < axesSize && i < strideSize; i++) {
       int64_t axesValue = axes[i];
       list_strides.SetDim(axesValue, GetConstIndexValue(strideTensor, i));
   }
}

static void ConstructBeginList(const gert::Tensor* beginTensor, const gert::Shape& xShape,
                              const std::vector<int64_t>& axes, ops::QuickVector& beginList) {
   // Initialize beginList with 0
   const int32_t dimNum = static_cast<int32_t>(xShape.GetDimNum());
   InitListWithDimNum(beginList, dimNum, 0);

   // Update beginList with const value of beginTensor
   const int32_t beginsSize = static_cast<int32_t>(beginTensor->GetShapeSize());
   const int32_t axesSize = static_cast<int32_t>(axes.size());

   for (int32_t i = 0; i < axesSize && i < beginsSize; i++) {
       int64_t axesValue = axes[i];
       int64_t inputDim = xShape.GetDim(axesValue);
       beginList.SetDim(axesValue, GetConstIndexValue(beginTensor, i, inputDim));
   }
}

static void ConstructEndList(const gert::Tensor* endTensor, const gert::Shape& xShape,
                            const std::vector<int64_t>& axes, ops::QuickVector& endList) {
   // Initialize endList with input_shape
   const int32_t dimNum = static_cast<int32_t>(xShape.GetDimNum());
   for (int32_t i = 0; i < dimNum; i++) {
       endList.AppendDim(xShape.GetDim(i));
   }

   // Update endList with const value of endTensor
   const int32_t endSize = static_cast<int32_t>(endTensor->GetShapeSize());
   const int32_t axesSize = static_cast<int32_t>(axes.size());

   for (int32_t i = 0; i < axesSize && i < endSize; i++) {
       int64_t axesValue = axes[i];
       int64_t inputDim = xShape.GetDim(axesValue);
       endList.SetDim(axesValue, GetConstIndexValue(endTensor, i, inputDim));
   }
}

static ge::graphStatus ConstructSliceParam(const gert::TilingContext *context, SliceParametersRuntime2 &sliceParam) {
   // Construct slice_param.input, slice_param.output_shape
   const gert::StorageShape *xStorage = context->GetInputShape(INDEX_X);
   OP_CHECK_NULL_WITH_CONTEXT(context, xStorage);
   const gert::StorageShape *yStorage = context->GetOutputShape(INDEX_Y);
   OP_CHECK_NULL_WITH_CONTEXT(context, yStorage);

   ConstructSliceShape(xStorage, sliceParam.inputShape);
   ConstructSliceShape(yStorage, sliceParam.outputShape);

   // Construct slice_param.begin_list, slice_param.end_list, slice_param.stride_list
   int32_t inputDimNum = static_cast<int32_t>(sliceParam.inputShape.GetDimNum());
   std::vector<int64_t> axes = ConstructValidAxis(context->GetOptionalInputTensor(INDEX_AXES), inputDimNum);

   const gert::Tensor *tensorBegin = context->GetInputTensor(INDEX_BEGIN);
   OP_CHECK_NULL_WITH_CONTEXT(context, tensorBegin);
   ConstructBeginList(tensorBegin, sliceParam.inputShape, axes, sliceParam.beginList);

   const gert::Tensor *tensorEnd = context->GetInputTensor(INDEX_END);
   OP_CHECK_NULL_WITH_CONTEXT(context, tensorEnd);
   ConstructEndList(tensorEnd, sliceParam.inputShape, axes, sliceParam.endList);

   const gert::Tensor *tensorStrides = context->GetOptionalInputTensor(INDEX_STRIDES);
   ConstructStrideList(tensorStrides, inputDimNum, axes, sliceParam.strideList);

   // Check slice_param.stride_list valid, only support value==1
   if (!CheckStride(sliceParam.strideList, context)) {
       return ge::GRAPH_FAILED;
   }
   return ge::GRAPH_SUCCESS;
}

static void ReconstructSliceParamByInferShape(ops::StridedSliceParams &inputParams, gert::Shape &shapeOutput,
                                             SliceParametersRuntime2 &sliceParam) {
   sliceParam.beginList = inputParams.begin;
   sliceParam.endList = inputParams.end;
   sliceParam.strideList = inputParams.strides;

   sliceParam.inputShape.SetDimNum(0);
   for (size_t i = 0; i < inputParams.input_shape.GetDimNum(); i++) {
       sliceParam.inputShape.AppendDim(inputParams.input_shape.GetDim(i));
   }

   for (size_t i = 0; i < shapeOutput.GetDimNum(); i++) {
       sliceParam.outputShape.AppendDim(shapeOutput.GetDim(i));
   }
}

static int64_t CalculateStrideValue(int64_t begin, int64_t end, int64_t stride) {
   return end > begin ? std::min(stride, end - begin) : std::max(stride, end - begin);
}

static void MakePerformanceParamsNeg(SliceParametersRuntime2 &param) {
   OP_LOGI("", "before handle negative perf slice params: %s", param.to_string().c_str());

   SliceParametersRuntime2 perfParams;
   size_t perfSize = 0;

   for (size_t i = 0; i < param.inputShape.GetDimNum(); i++) {
       const auto inputShapeI = param.inputShape[i];
       const auto outputShapeI = param.outputShape[i];
       const auto beginI = param.beginList[i];
       const auto endI = param.endList[i];
       const auto stride_i = CalculateStrideValue(beginI, endI, param.strideList[i]);

       // Skip size-1 dimensions (except first)
       if (inputShapeI == 1 && outputShapeI == 1 && i != 0) {
           continue;
       }

       // Continuous stride=-1 axis fusion
       if (i == 0 || inputShapeI != outputShapeI || stride_i != -1 || perfParams.strideList[perfSize - 1] != -1) {
           perfParams.inputShape.AppendDim(inputShapeI);
           perfParams.outputShape.AppendDim(outputShapeI);
           perfParams.beginList.AppendDim(beginI);
           perfParams.endList.AppendDim(endI);
           perfParams.strideList.AppendDim(stride_i);
           perfSize++;
           continue;
       }

       const auto perfIndex = perfSize - 1;
       perfParams.inputShape[perfIndex] *= inputShapeI;
       perfParams.outputShape[perfIndex] *= outputShapeI;
       perfParams.beginList[perfIndex] = perfParams.beginList[perfIndex] * inputShapeI + inputShapeI - 1;
       perfParams.endList[perfIndex] = perfParams.endList[perfIndex] * inputShapeI + inputShapeI - 1;
       perfParams.strideList[perfIndex] = -1;
   }

   param = perfParams;

   // Fix negative stride for size-1 output dimensions
   for (size_t i = 0; i < param.inputShape.GetDimNum(); i++) {
       if (param.outputShape[i] == 1 && param.strideList[i] < 0) {
           param.strideList[i] = 1;
           param.endList[i] = param.beginList[i] + 1;
       }
   }
}

static bool ShouldAdjustLastStride(const SliceParametersRuntime2 &param, size_t lastDim) {
   const auto inputLastDim = param.inputShape[lastDim];
   const auto beginLastDim = param.beginList[lastDim];
   const auto strideLastDim = param.strideList[lastDim];
   const auto outputLastDim = param.outputShape[lastDim];

   return strideLastDim > 1 &&
          inputLastDim % strideLastDim == 0 &&
          beginLastDim / strideLastDim == 0 &&
          outputLastDim == inputLastDim / strideLastDim;
}

static void AdjustLastStrideDimension(SliceParametersRuntime2 &param) {
   size_t th = param.inputShape.GetDimNum() - 1;
   const auto inputLastDim = param.inputShape[th];
   const auto beginLastDim = param.beginList[th];
   const auto strideLastDim = param.strideList[th];
   const auto outputLastDim = param.outputShape[th];

   // Split the last dimension
   param.inputShape[th] = inputLastDim / strideLastDim;
   param.inputShape.AppendDim(strideLastDim);

   param.beginList[th] = beginLastDim / strideLastDim;
   param.beginList.AppendDim(beginLastDim % strideLastDim);

   param.strideList[th] = 1;
   param.strideList.AppendDim(1);

   param.outputShape.AppendDim(1);

   param.endList[th] = param.beginList[th] + outputLastDim;
   param.endList.AppendDim(param.beginList[th + 1] + 1);
}

static void MakePerformanceParams(SliceParametersRuntime2 &param, bool isAdjustLastStride) {
   if (param.outputShape.GetDimNum() == 0) {
       return;
   }

   // Check for negative strides
   bool hasNegStride = false;
   for (size_t i = 0; i < param.inputShape.GetDimNum(); i++) {
       if (param.strideList[i] < 0) {
           hasNegStride = true;
           break;
       }
   }

   // Adjust last dimension if needed
   if (isAdjustLastStride && !hasNegStride) {
       size_t lastDim = param.inputShape.GetDimNum() - 1;
       if (ShouldAdjustLastStride(param, lastDim)) {
           AdjustLastStrideDimension(param);
       }
   }

   SliceParametersRuntime2 perfParams;
   size_t perfSize = 0;

   for (size_t i = 0; i < param.inputShape.GetDimNum(); i++) {
       const auto inputShapeI = param.inputShape[i];
       const auto outputShapeI = param.outputShape[i];
       const auto beginI = param.beginList[i];
       const auto endI = param.endList[i];
       const auto stride_i = endI > beginI ? std::min(param.strideList[i], endI - beginI) : param.strideList[i];
       // Fuse continuous stride=1 dimensions
       if (i == 0 || inputShapeI != outputShapeI || stride_i != 1 || perfParams.strideList[perfSize - 1] != 1) {
           perfParams.inputShape.AppendDim(inputShapeI);
           perfParams.outputShape.AppendDim(outputShapeI);
           perfParams.beginList.AppendDim(beginI);
           perfParams.endList.AppendDim(endI);
           perfParams.strideList.AppendDim(stride_i);
           perfSize++;
           continue;
       }

       const auto perfIndex = perfSize - 1;
       perfParams.inputShape[perfIndex] *= inputShapeI;
       perfParams.outputShape[perfIndex] *= outputShapeI;
       perfParams.beginList[perfIndex] *= inputShapeI;
       perfParams.endList[perfIndex] *= inputShapeI;
       perfParams.strideList[perfIndex] = 1;
   }

   param = perfParams;

   if (hasNegStride) {
       MakePerformanceParamsNeg(param);
   }
}

ge::graphStatus StrideSliceV2TilingForAscendC(gert::TilingContext* context, int64_t coreNum, int64_t ubSize,
                                             int64_t cachelineSize, SliceParametersRuntime2& sliceParam,
                                             const ge::DataType& dtype) {
   StrideSliceTiling tilingObject(context);
   if (tilingObject.Init(coreNum, ubSize, cachelineSize, sliceParam, dtype) != ge::GRAPH_SUCCESS) {
       return ge::GRAPH_FAILED;
   }
   return tilingObject.RunStrideSliceTiling();
}

static ge::graphStatus GetMaskValue(
    gert::TilingContext* context, const gert::RuntimeAttrs* attrs, size_t attrIndex, uint64_t& maskValue)
{
    const int64_t* mask = attrs->GetAttrPointer<int64_t>(attrIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, mask);
    maskValue = static_cast<uint64_t>(*mask);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4StridedSliceV2(gert::TilingContext *context) {
   SliceParametersRuntime2 sliceParam;
   if (ConstructSliceParam(context, sliceParam) != ge::GRAPH_SUCCESS) {
       return ge::GRAPH_FAILED;
   }

   // Get mask attributes
   auto attrs = context->GetAttrs();
   OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

   uint64_t beginValue{0}, endValue{0}, ellipsisValue{0}, newAxisValue{0}, shrinkAxisValue{0};
   CHECK_RET_SUCC(GetMaskValue(context, attrs, IDX_MASK_BEGIN, beginValue));
   CHECK_RET_SUCC(GetMaskValue(context, attrs, IDX_MASK_END, endValue));
   CHECK_RET_SUCC(GetMaskValue(context, attrs, IDX_MASK_ELLIPSIS, ellipsisValue));
   CHECK_RET_SUCC(GetMaskValue(context, attrs, IDX_MASK_NEW_AXIS, newAxisValue));
   CHECK_RET_SUCC(GetMaskValue(context, attrs, IDX_MASK_SHRINK_AXIS, shrinkAxisValue));

   // Infer shape
   const gert::StorageShape *xStorage = context->GetInputShape(INDEX_X);
   OP_CHECK_NULL_WITH_CONTEXT(context, xStorage);
   const gert::Shape &shapeInput = Ops::Base::EnsureNotScalar(xStorage->GetStorageShape());

   ops::StridedSliceParams inputParams = {
       shapeInput,
       sliceParam.beginList,
       sliceParam.endList,
       sliceParam.strideList,
       beginValue,
       endValue,
       ellipsisValue,
       newAxisValue,
       shrinkAxisValue,
       true, true, true,
       sliceParam.isBeginConst,
       sliceParam.isEndConst,
       shapeInput
   };

   gert::Shape shapeOutput;
   if (!ops::InferShape(inputParams, &shapeOutput)) {
       OP_LOGE(OP_NAME, "InferShape fail.");
       return ge::GRAPH_FAILED;
   }

   // Reconstruct slice_param by infer shape
   ReconstructSliceParamByInferShape(inputParams, shapeOutput, sliceParam);

   // Align slice param dims
   MakeSameDims(&sliceParam);
   OP_LOGI(context->GetNodeName(), "align slice params: %s", sliceParam.to_string().c_str());

   // Optimize performance slice param
   MakePerformanceParams(sliceParam, true);
   OP_LOGI(context->GetNodeName(), "perf slice params: %s", sliceParam.to_string().c_str());

   // Infer tiling mode
   auto compileInfo = context->GetCompileInfo<StridedSliceV2CompileInfo>();
   OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
   OP_LOGD(context->GetNodeName(), "compile info: %s.", compileInfo->to_string().c_str());

   auto xDesc = context->GetInputDesc(INDEX_X);
   OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);

   return StrideSliceV2TilingForAscendC(context, compileInfo->coreNum, compileInfo->ubSize,
                                        compileInfo->cacheLineSize, sliceParam, xDesc->GetDataType());
}

ge::graphStatus TilingPrepare4StridedSliceV2(gert::TilingParseContext* context) {
   OP_LOGD(context->GetNodeName(), "Start parse StridedSliceV2Tiling.");

   auto ci = context->GetCompiledInfo<StridedSliceV2CompileInfo>();
   OP_CHECK_NULL_WITH_CONTEXT(context, ci);

   auto platformInfo = context->GetPlatformInfo();
   OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);

   auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
   ci->coreNum = ascendcPlatform.GetCoreNumAiv();
   if (ci->coreNum <= 0) {
       OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
               context->GetNodeName(), "core number", std::to_string(ci->coreNum),
               "The value of core number cannot <= 0.");
       return ge::GRAPH_FAILED;
   }

   uint64_t ubSize;
   ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
   ci->ubSize = static_cast<int64_t>(ubSize);
   if (ci->ubSize <= 0) {
       OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
               context->GetNodeName(), "ubSize", std::to_string(ci->ubSize),
               "The value of ubSize cannot <= 0.");
       return ge::GRAPH_FAILED;
   } 

   ci->cacheLineSize = Ops::Base::GetCacheLineSize(context);
   if (ci->cacheLineSize <= 0) {
       OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
               context->GetNodeName(), "cacheLineSize", std::to_string(ci->cacheLineSize),
               "The value of cacheLineSize cannot <= 0.");
       return ge::GRAPH_FAILED;
   }  

   return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(StridedSliceV2)
   .Tiling(Tiling4StridedSliceV2)
   .TilingParse<StridedSliceV2CompileInfo>(TilingPrepare4StridedSliceV2)
   .TilingInputsDataDependency({INDEX_BEGIN, INDEX_END, INDEX_AXES, INDEX_STRIDES});
} // namespace optiling