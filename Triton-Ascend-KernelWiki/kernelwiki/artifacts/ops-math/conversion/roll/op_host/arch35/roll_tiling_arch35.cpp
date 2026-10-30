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
 * \file roll_tiling_arch35.cpp
 * \brief
 */
#include "roll_tiling_arch35.h"
#include "op_host/tiling_base_util.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"

namespace optiling {
constexpr int64_t INPUT_X_IDX = 0;
constexpr int64_t ATTR_SHIFTS_IDX = 0;
constexpr int64_t ATTR_DIMS_IDX = 1;
constexpr int64_t OUTPUT_Y_IDX = 0;
constexpr int64_t BUFFER_NUM = 2;
constexpr int64_t ALIVE_NODE = 2;
constexpr int64_t CONSTANT_TWO = 2;
constexpr int64_t ALIGN_BYTE = 32;
constexpr int64_t ALIGN_COUNT = 32;
constexpr int64_t SMALL_TAIL_CONDITION = 64;
constexpr int64_t UINT16_MAX_NUM = 65535;
constexpr uint64_t TILING_KEY_FOR_SIMD_ONE_DIM = 10000;
constexpr uint64_t TILING_KEY_FOR_SIMD_BEFOR_H = 20000;
constexpr uint64_t TILING_KEY_FOR_SIMD_AFTER_H_ALIGN = 30000;
constexpr uint64_t TILING_KEY_FOR_SIMD_AFTER_H_UNALIGN = 30001; // 非对齐
constexpr uint64_t TILING_KEY_FOR_SIMD_SPLIT_W = 40000;
constexpr uint64_t TILING_KEY_FOR_SIMD_SMALL_TAIL_SHIFTW = 50000;
constexpr uint64_t TILING_KEY_FOR_SIMD_SMALL_TAIL_NOT_SHIFTW = 50001;
constexpr uint64_t TILING_KEY_FOR_EMPTY_TENSOR = 60000;
constexpr size_t RESERVED_WORKSPACE_SIZE = static_cast<size_t>(16 * 1024 * 1024);
constexpr int64_t SIMT_DCACHE_SIZE = static_cast<int64_t>(64 * 1024);

static const std::vector<ge::DataType> X_SUPPORT_DTYPE = {ge::DT_INT8,  ge::DT_UINT8,   ge::DT_INT32, ge::DT_UINT32,
                                                          ge::DT_INT64, ge::DT_FLOAT16, ge::DT_BF16,  ge::DT_FLOAT};

inline static int64_t Mod(int64_t a, int64_t b)
{
    int64_t r = a % b;
    if (r < 0) {
        r += (b < 0) ? -b : b;
    }
    return r;
}

ge::graphStatus RollTilingClass::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        aicoreParams_.numBlocks = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        aicoreParams_.ubSize = ubSizePlatForm;
    } else {
        auto compileInfoPtr = reinterpret_cast<const RollCompileInfoArch35*>(context_->GetCompileInfo());
        OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfoPtr);
        aicoreParams_.numBlocks = compileInfoPtr->core_num;
        aicoreParams_.ubSize = compileInfoPtr->ub_size;
    }
    cacheLineSize_ = Ops::Base::GetCacheLineSize(context_);
    OP_LOGD(context_, "cacheLineSize_ is: %ld", cacheLineSize_);
    if (cacheLineSize_ == 0LL) {
        std::string valueMsg = "0";
        std::string reasonMsg = "Failed to get cacheLine size";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "cacheLineSize",
            valueMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    vectorSize_ = static_cast<int64_t>(Ops::Base::GetVRegSize(context_));
    OP_LOGD(context_, "vectorSize_ is: %ld", vectorSize_);
    if (vectorSize_ == 0LL) {
        std::string valueMsg = "0";
        std::string reasonMsg = "Failed to get vector size";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "vectorSize",
            valueMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RollTilingClass::GetShapeAttrsInfo()
{
    // 获取输入shape
    xShapePtr_ = context_->GetInputShape(INPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr_);

    // 获取输出shape
    yShapePtr_ = context_->GetOutputShape(OUTPUT_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShapePtr_);

    // 获取属性
    auto attrs = context_->GetAttrs();
    shiftsListPtr_ = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_SHIFTS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shiftsListPtr_);
    dimsListPtr_ = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_DIMS_IDX);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RollTilingClass::CheckAndGetInputParam()
{
    const gert::Shape xShape = Ops::Base::EnsureNotScalar(xShapePtr_->GetStorageShape());
    OP_LOGD(context_, "Input x shape is: %s", Ops::Base::ToString(xShape).c_str());
    const gert::Shape yShape = Ops::Base::EnsureNotScalar(yShapePtr_->GetStorageShape());
    OP_LOGD(context_, "Output y shape is: %s", Ops::Base::ToString(yShape).c_str());
    if (xShape != yShape) {
        std::string shapeMsg = Ops::Base::ToString(xShape) + " and " + Ops::Base::ToString(yShape);
        std::string reasonMsg = "Input x shape should be equal to output y shape";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "x and y",
            shapeMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    dimNum_ = xShape.GetDimNum();
    OP_LOGD(context_, "DimNum is: %ld", dimNum_);
    for (int32_t i = 0; i < dimNum_; i++) {
        shapes_[i] = xShape.GetDim(i);
    }

    // 最大支持8维
    if (dimNum_ > MAX_DIM_NUM) {
        std::string dimMsg = std::to_string(dimNum_);
        std::string reasonMsg = "should be less than or equal to 8 dims";
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "x",
            dimMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    totalEmelents_ = xShape.GetShapeSize();
    OP_LOGD(context_, "total emelents is: %ld.", totalEmelents_);
    dtypeSize_ = static_cast<int64_t>(ge::GetSizeByDataType(context_->GetInputDesc(INPUT_X_IDX)->GetDataType()));
    OP_LOGD(context_, "Input x dtype size is: %ld.", dtypeSize_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RollTilingClass::CheckAttr()
{
    const int64_t* shiftsData = reinterpret_cast<const int64_t*>(shiftsListPtr_->GetData());
    int64_t shiftsSize = shiftsListPtr_->GetSize();
    OP_LOGD(context_, "shiftsSize is: %ld", shiftsSize);
    if (totalEmelents_ == 0) {
        // 空tensor，直接返回
        tilingKey = TILING_KEY_FOR_EMPTY_TENSOR;
        OP_LOGD(context_, "None tensor, tilingBranch is %u", tilingKey);
        return ge::GRAPH_SUCCESS;
    }
    if (dimsListPtr_ == nullptr || dimsListPtr_->GetSize() == 0) {
        if (shiftsSize > 1) {
            std::string valueMsg = std::to_string(shiftsSize);
            std::string reasonMsg = "shifts size should be 1 when dims is empty";
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "shifts",
                valueMsg.c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
        dimNum_ = 1;
        shifts_[0] = Mod(shiftsData[0], totalEmelents_);
    } else {
        int64_t dimsSize = dimsListPtr_->GetSize();
        OP_LOGD(context_, "dimsSize is: %ld", dimsSize);
        if (shiftsSize != dimsSize) {
            std::string valuesMsg = std::to_string(shiftsSize) + " and " + std::to_string(dimsSize);
            std::string reasonMsg = "the size of shifts should be equal to the size of dims";
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context_->GetNodeName(), "shifts and dims",
                valuesMsg.c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
        const int64_t* dimsData = reinterpret_cast<const int64_t*>(dimsListPtr_->GetData());
        for (int32_t i = 0; i < dimsSize; i++) {
            int64_t dimData = dimsData[i];
            int64_t shiftData = shiftsData[i];
            if (dimData >= dimNum_ || dimData < -dimNum_) {
                std::string valueMsg = std::to_string(dimData);
                std::string correctMsg = "in range of [" + std::to_string(-dimNum_) + ", " +
                    std::to_string(dimNum_ - 1) + "]";
                OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "dims",
                    valueMsg.c_str(), correctMsg.c_str());
                return ge::GRAPH_FAILED;
            }
            if (dimData < 0) {
                dimData = dimData + dimNum_;
            }
            shifts_[dimData] = Mod(shifts_[dimData] + shiftData, shapes_[dimData]);
        }
    }
    MergeAxes(shapes_, shifts_, dimNum_); // 合并连续的0轴
    RemoveShapeOne(shapes_, shifts_, dimNum_); // 移除为1的shape
    strides_[dimNum_ - 1] = 1;
    for (int32_t i = dimNum_ - CONSTANT_TWO; i >= 0; --i) {
        strides_[i] = strides_[i + 1] * shapes_[i + 1];
    }
    return ge::GRAPH_SUCCESS;
}

void RollTilingClass::MergeAxes(int64_t shape[], int64_t shifts[], int64_t& dimNum)
{
    if (dimNum <= 0) {
        return;
    }

    if (dimNum == 1) {
        shape[0] = totalEmelents_;
        return;
    }

    int64_t new_shape[dimNum];
    int64_t new_shifts[dimNum];
    int64_t new_size = 0;

    int64_t current_merge_size = 1;
    bool in_merge_zone = false;

    for (int32_t i = 0; i < dimNum; ++i) {
        if (shifts[i] == 0) {
            if (!in_merge_zone) {
                in_merge_zone = true;
                current_merge_size = shape[i];
            } else {
                current_merge_size *= shape[i];
            }

            if (i == dimNum - 1 || shifts[i + 1] != 0) {
                new_shape[new_size] = current_merge_size;
                new_shifts[new_size] = 0;
                new_size++;
                in_merge_zone = false;
            }
        } else {
            if (in_merge_zone) {
                new_shape[new_size] = current_merge_size;
                new_shifts[new_size] = 0;
                new_size++;
                in_merge_zone = false;
            }
            new_shape[new_size] = shape[i];
            new_shifts[new_size] = shifts[i];
            new_size++;
        }
    }

    for (int32_t i = 0; i < new_size; ++i) {
        shape[i] = new_shape[i];
        shifts[i] = new_shifts[i];
    }
    // 更新实际维度数
    dimNum = new_size;
}

void RollTilingClass::RemoveShapeOne(int64_t shape[], int64_t shifts[], int64_t& dimNum)
{
    if(dimNum <= 1) {
        return;
    }

    int64_t new_shape[dimNum];
    int64_t new_shift[dimNum];
    int64_t new_size = 0;

    for(int32_t i = 0; i < dimNum; ++i) {
        if(shape[i] > 1) {
            new_shape[new_size] = shape[i];
            new_shift[new_size] = shifts[i];
            new_size++;
        }
    }

    for(int32_t i = 0; i < new_size; ++i) {
        shape[i] = new_shape[i];
        shifts[i] = new_shift[i];
    }
    dimNum = new_size;
}

void RollTilingClass::SplitCore()
{
    int64_t cacheLineLen = cacheLineSize_ / dtypeSize_; // 切核最小长度
    // 找切核的轴
    int64_t useBlockDim_ = aicoreParams_.numBlocks; // 设备可用的核数
    OP_LOGD(context_, "useBlockDim_ is: %ld", useBlockDim_);
    int64_t currentProd = 1;
    blockSplitAxis_ = 0;    // 默认在0轴
    bool isSuccess = false; // 未进入分核

    for (int32_t i = 0; i < dimNum_; i++) { // 先找到能分核的轴
        if (shapes_[i] * strides_[i] <= cacheLineLen) {
            break;
        }
        isSuccess = true;
        currentProd *= shapes_[i];
        blockSplitAxis_ = i;
        if (currentProd >= useBlockDim_) {
            break;
        }
    }
    OP_LOGD(context_, "currentProd is: %ld", currentProd);
    if (!isSuccess) {
        // 不需要分核，单核
        blockFactor_ = shapes_[0];     // 核切分轴上切分的长度
        blockCount_ = 1;               // 使用的核数
        blockTailFactor_ = shapes_[0]; // 尾核
    } else {
        blockFactor_ = (currentProd + useBlockDim_ - 1) / useBlockDim_;    // 核切分轴上切分的长度
        blockCount_ = (currentProd + blockFactor_ - 1) / blockFactor_;     // 使用的核数
        blockTailFactor_ = currentProd - (blockCount_ - 1) * blockFactor_; // 尾核
    }

    OP_LOGD(context_, "blockFactor_ is: %ld", blockFactor_);
    OP_LOGD(context_, "blockCount_ is: %ld", blockCount_);
    OP_LOGD(context_, "blockTailFactor_ is: %ld", blockTailFactor_);
}

void RollTilingClass::SplitUb(UbParam& ubparam, bool isTail)
{
    int64_t wSize = shapes_[dimNum_ - 1];
    int64_t maxElementSize = 0;
    if (tilingKey == TILING_KEY_FOR_SIMD_SMALL_TAIL_SHIFTW || tilingKey == TILING_KEY_FOR_SIMD_SMALL_TAIL_NOT_SHIFTW) {
        // 公共坐标空间，double buffer，input & output各留一个寄存器空间
        maxElementSize = (aicoreParams_.ubSize - vectorSize_) / BUFFER_NUM / ALIVE_NODE - vectorSize_;
        // maxElementSize不能超过uint16最大值减掉input和output的寄存器空间
        if (maxElementSize >= (UINT16_MAX_NUM * dtypeSize_ - vectorSize_ * 2)) {
            maxElementSize = UINT16_MAX_NUM * dtypeSize_ - vectorSize_ * 2;
        }
    } else {
        maxElementSize = aicoreParams_.ubSize / BUFFER_NUM;
        int64_t wByte = (wSize * dtypeSize_ + ALIGN_BYTE - 1) / ALIGN_BYTE * ALIGN_BYTE;
        if ((wSize - shifts_[dimNum_ - 1]) * dtypeSize_ % ALIGN_BYTE) {
            wByte += vectorSize_;
        }
        wSize = wByte / dtypeSize_;
    }
    OP_LOGD(context_, "vectorSize_ is: %ld", vectorSize_);
    OP_LOGD(context_, "maxElementSize is: %ld", maxElementSize);

    // b8的用b16替代
    int64_t usedDtypeSize_ = 2;
    if (dtypeSize_ > usedDtypeSize_) {
        usedDtypeSize_ = dtypeSize_;
    }

    maxElements_ = maxElementSize / usedDtypeSize_; // 长度

    OP_LOGD(context_, "maxElements_ is: %ld", maxElements_);
    OP_LOGD(context_, "wSize is: %ld", wSize);
    // 找切ub的轴
    ubparam.UbSplitAxis = dimNum_ - 1; // 默认最后一根轴
    int64_t currentProd = 1;
    for (int32_t i = dimNum_ - 1; i >= blockSplitAxis_; i--) { // 最多切到切核的轴
        if (i == dimNum_ - 1) {                                // 用更新的wSize切分
            currentProd *= wSize;
        } else {
            currentProd *= shapes_[i];
        }
        ubparam.UbSplitAxis = i;
        if (currentProd > maxElements_) {
            break;
        }
    }
    int64_t perUbFactor = strides_[ubparam.UbSplitAxis] / shapes_[dimNum_ - 1] * wSize; // 每个UB切分的基本大小
    if (ubparam.UbSplitAxis == dimNum_ - 1) {                                           // 注意尾轴切分
        perUbFactor = 1;
    }
    OP_LOGD(context_, "perUbFactor is: %ld", perUbFactor);
    int64_t singleUbNeedNum = maxElements_ / perUbFactor;
    OP_LOGD(context_, "singleUbNeedNum is: %ld", singleUbNeedNum);
    if (ubparam.UbSplitAxis == blockSplitAxis_) { // 切核与切UB在同一轴，需要重新处理
        if (isTail) {                             // 尾核
            if (blockTailFactor_ <= singleUbNeedNum) {
                ubparam.UbFactor = blockTailFactor_;
                ubparam.UbCount = 1;
                ubparam.UbTailFactor = blockTailFactor_;
            } else {
                ubparam.UbFactor = singleUbNeedNum;
                ubparam.UbCount = (blockTailFactor_ + singleUbNeedNum - 1) / singleUbNeedNum;
                ubparam.UbTailFactor = blockTailFactor_ - (ubparam.UbCount - 1) * ubparam.UbFactor;
            }
        } else {
            if (blockFactor_ <= singleUbNeedNum) {
                ubparam.UbFactor = blockFactor_;
                ubparam.UbCount = 1;
                ubparam.UbTailFactor = blockFactor_;
            } else {
                ubparam.UbFactor = singleUbNeedNum;
                ubparam.UbCount = (blockFactor_ + singleUbNeedNum - 1) / singleUbNeedNum;
                ubparam.UbTailFactor = blockFactor_ - (ubparam.UbCount - 1) * ubparam.UbFactor;
            }
        }
    } else { // 不在同一轴
        ubparam.UbFactor = singleUbNeedNum;
        ubparam.UbCount = (shapes_[ubparam.UbSplitAxis] + ubparam.UbFactor - 1) / ubparam.UbFactor;
        ubparam.UbTailFactor = shapes_[ubparam.UbSplitAxis] - (ubparam.UbCount - 1) * ubparam.UbFactor;
    }
    OP_LOGD(context_, "UbSplitAxis is: %ld", ubparam.UbSplitAxis);
    OP_LOGD(context_, "UbFactor is: %ld", ubparam.UbFactor);
    OP_LOGD(context_, "UbCount is: %ld", ubparam.UbCount);
    OP_LOGD(context_, "UbTailFactor is: %ld", ubparam.UbTailFactor);
}

void RollTilingClass::upDateParam(
    int64_t index, int64_t srcOffset, int64_t blockCount, int64_t blockLen, int64_t srcStride, int64_t dstOffset)
{
    moveparam.srcOffset[index] = srcOffset;
    moveparam.blockCount[index] = blockCount;
    moveparam.blockLen[index] = blockLen;
    moveparam.srcStride[index] = srcStride;
    moveparam.dstOffset[index] = dstOffset;
}

void RollTilingClass::CalMoveParam() // 切在H之前 or H轴满载
{
    int64_t h = shapes_[dimNum_ - 2];
    int64_t w = shapes_[dimNum_ - 1];
    int64_t wLen = w;
    int64_t shiftH = shifts_[dimNum_ - 2];
    int64_t shiftW = shifts_[dimNum_ - 1];

    // w需要对齐
    if (((w * dtypeSize_) % ALIGN_BYTE)) {
        int64_t wByte = (w * dtypeSize_ + ALIGN_BYTE - 1) / ALIGN_BYTE * ALIGN_BYTE;
        w = wByte / dtypeSize_;
    }

    // 搬入初始参数
    moveparam.mte3Count = 1;
    // H和W维都做shift，dimNum_ - 2表示H维
    if (shifts_[dimNum_ - 1] > 0 && shifts_[dimNum_ - 2] > 0) {
        moveparam.mte3Count = 4; // H&W都shift，4套坐标
        upDateParam(0, 0, h - shiftH, wLen - shiftW, w, shiftH * wLen + shiftW);
        upDateParam(1, wLen - shiftW, h - shiftH, shiftW, w, shiftH * wLen);
        upDateParam(2, (h - shiftH) * w, shiftH, wLen - shiftW, w, shiftW);     // 第三个坐标
        upDateParam(3, (h - shiftH) * w + wLen - shiftW, shiftH, shiftW, w, 0); // 第四个坐标
    } else if (shifts_[dimNum_ - 1] > 0) {
        moveparam.mte3Count = 2; // W shift，2套坐标
        upDateParam(0, 0, h, wLen - shiftW, w, shiftW);
        upDateParam(1, wLen - shiftW, h, shiftW, w, 0);
    } else {
        moveparam.mte3Count = 2; // H shift，2套坐标
        upDateParam(0, 0, h - shiftH, wLen, w, shiftH * wLen);
        upDateParam(1, (h - shiftH) * w, shiftH, wLen, w, 0);
    }
}

void RollTilingClass::SplitCoreforSimd()
{
    if (shifts_[dimNum_ - 1] != 0) {
        basicElements_ = 1;
    } else {
        basicElements_ = shifts_[dimNum_ - 1];
    }
    maxElements_ = aicoreParams_.ubSize / BUFFER_NUM / dtypeSize_;
    maxElements_ = maxElements_ / ALIGN_COUNT * ALIGN_COUNT;
    OP_LOGD(context_, "basicElements is: %ld", basicElements_);
    perCoreElements_ = (totalEmelents_ + aicoreParams_.numBlocks - 1) / aicoreParams_.numBlocks;
    needCoreNum_ = (totalEmelents_ + perCoreElements_ - 1) / perCoreElements_;
    lastCoreElements_ = totalEmelents_ - (needCoreNum_ - 1) * perCoreElements_;
}

ge::graphStatus RollTilingClass::DoOpTiling()
{
    auto ret = CheckAndGetInputParam();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = CheckAttr();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    // 处理空tensor的情况
    if (tilingKey == TILING_KEY_FOR_EMPTY_TENSOR) {
        return ge::GRAPH_SUCCESS;
    }

    // 一、切UB之前就能判断的场景
    // H*W*数据类型字节小于cacheline,shapes_[dimNum_ - 2]表示H轴的shape
    if ((dimNum_ > 0 && (shapes_[dimNum_ - 1] * dtypeSize_ < SMALL_TAIL_CONDITION) && shifts_[dimNum_ - 1] > 0) ||
        ((dimNum_ > 1) && (shapes_[dimNum_ - 1] * shapes_[dimNum_ - 2] * dtypeSize_ < cacheLineSize_))) {
        // 小尾轴场景
        tilingKey = (shifts_[dimNum_ - 1] > 0) ? TILING_KEY_FOR_SIMD_SMALL_TAIL_SHIFTW :
                                                 TILING_KEY_FOR_SIMD_SMALL_TAIL_NOT_SHIFTW;
        OP_LOGD(context_, "tilingBranch is: %ld", tilingKey);
        SplitCore();
        SplitUb(mainCoreUbParam_, false);
        SplitUb(tailCoreUbParam_, true);
        return ge::GRAPH_SUCCESS;
    } else if (dimNum_ == 1) {
        // 只有一维
        tilingKey = TILING_KEY_FOR_SIMD_ONE_DIM;
        SplitCoreforSimd();
        return ge::GRAPH_SUCCESS;
    }

    // 二、先切UB之后才能判断出场景
    SplitCore();
    SplitUb(mainCoreUbParam_, false);
    SplitUb(tailCoreUbParam_, true);

    UbParam ubparam = mainCoreUbParam_;
    if (ubparam.UbSplitAxis < dimNum_ - 2) {
        // dim需要大于2维并且切分的轴在c轴及之前
        tilingKey = TILING_KEY_FOR_SIMD_BEFOR_H;
        // 该场景需要计算搬运参数
        CalMoveParam();
    } else if (ubparam.UbSplitAxis == dimNum_ - 1) {
        // 切到W
        tilingKey = TILING_KEY_FOR_SIMD_SPLIT_W;
        // 重新切核
        SplitCoreforSimd();
    } else if (
        (ubparam.UbSplitAxis == dimNum_ - 2 &&
         ((shapes_[dimNum_ - 1] - shifts_[dimNum_ - 1]) * dtypeSize_ % ALIGN_BYTE == 0)) ||
        (ubparam.UbSplitAxis == dimNum_ - 2 && (shifts_[dimNum_ - 1] == 0))) {
        // 切到H轴且shift W对齐，dimNum_ - 2 表示H轴
        tilingKey = TILING_KEY_FOR_SIMD_AFTER_H_ALIGN;
    } else {
        // 切到H轴 shift非对齐
        tilingKey = TILING_KEY_FOR_SIMD_AFTER_H_UNALIGN;
        // 重新切核
        SplitCoreforSimd();
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RollTilingClass::GetWorkspaceSize()
{
    auto workSpaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workSpaces);
    workSpaces[0] = RESERVED_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RollTilingClass::PostTiling()
{
    if (tilingKey == TILING_KEY_FOR_EMPTY_TENSOR) {
        needCoreNum_ = 1;
        OP_LOGD(context_, "None tensor, needCoreNum_ is %ld", needCoreNum_);
    }
    tilingData_ = context_->GetTilingData<RollTilingData>();
    tilingData_->needCoreNum = needCoreNum_;
    tilingData_->dimNum = dimNum_;
    tilingData_->basicElements = basicElements_;
    tilingData_->maxElements = maxElements_;
    tilingData_->perCoreElements = perCoreElements_;
    tilingData_->lastCoreElements = lastCoreElements_;
    for (int32_t i = 0; i < dimNum_; i++) {
        tilingData_->shapes[i] = shapes_[i];
        tilingData_->shifts[i] = shifts_[i];
        tilingData_->strides[i] = strides_[i];
    }

    tilingData_->blockCount = blockCount_;
    tilingData_->blockFactor = blockFactor_;
    tilingData_->blockTailFactor = blockTailFactor_;
    tilingData_->blockSplitAxis = blockSplitAxis_;

    tilingData_->mainCoreUbParam = mainCoreUbParam_;
    tilingData_->tailCoreUbParam = tailCoreUbParam_;
    tilingData_->moveparam = moveparam;
    context_->SetTilingKey(GetTilingKey());
    if (tilingKey == TILING_KEY_FOR_EMPTY_TENSOR) {
        context_->SetBlockDim(needCoreNum_);
    } else if (
        tilingKey == TILING_KEY_FOR_SIMD_ONE_DIM || tilingKey == TILING_KEY_FOR_SIMD_SPLIT_W ||
        tilingKey == TILING_KEY_FOR_SIMD_AFTER_H_UNALIGN) {
        context_->SetBlockDim(needCoreNum_);
    } else {
        context_->SetBlockDim(blockCount_);
    }
    PrintTiling();
    return ge::GRAPH_SUCCESS;
}

void RollTilingClass::PrintTiling() const
{
    if (tilingKey == TILING_KEY_FOR_EMPTY_TENSOR) {
        OP_LOGI(context_, "Roll tilingKey is %ld", tilingKey);
        OP_LOGI(context_, "BlockDim is %d", needCoreNum_);
        return;
    }

    OP_LOGD(context_, "cacheLineSize_ %ld", cacheLineSize_);

    OP_LOGD(context_, "vectorSize_ %ld", vectorSize_);

    OP_LOGD(context_, "Roll tilingKey is %ld", tilingKey);

    OP_LOGD(
        context_,
        "Roll tilingData: blockCount = %ld, blockFactor = %ld, blockTailFactor = %ld, "
        "blockSplitAxis = %ld",
        tilingData_->blockCount, tilingData_->blockFactor, tilingData_->blockTailFactor, tilingData_->blockSplitAxis);

    OP_LOGD(
        context_,
        "Roll tilingData->mainCoreubparam: UbSplitAxis = %ld, UbCount = %ld, UbFactor = %ld, UbTailFactor = %ld",
        tilingData_->mainCoreUbParam.UbSplitAxis, tilingData_->mainCoreUbParam.UbCount,
        tilingData_->mainCoreUbParam.UbFactor, tilingData_->mainCoreUbParam.UbTailFactor);

    OP_LOGD(
        context_,
        "Roll tilingData->tailCoreubparam: UbSplitAxis = %ld, UbCount = %ld, UbFactor = %ld, UbTailFactor = %ld",
        tilingData_->tailCoreUbParam.UbSplitAxis, tilingData_->tailCoreUbParam.UbCount,
        tilingData_->tailCoreUbParam.UbFactor, tilingData_->tailCoreUbParam.UbTailFactor);

    for (int64_t i = 0; i < moveparam.mte3Count; i++) {
        OP_LOGD(
            context_,
            "Roll tilingData->MoveParam: mte3Count = %ld, srcOffset[%ld] = %ld, blockCount[%ld] = %ld, blockLen[%ld] = "
            "%ld"
            "srcStride[%ld] = %ld, dstOffeset[%ld] = %ld",
            tilingData_->moveparam.mte3Count, i, tilingData_->moveparam.srcOffset[i], i,
            tilingData_->moveparam.blockCount[i], i, tilingData_->moveparam.blockLen[i], i,
            tilingData_->moveparam.srcStride[i], i, tilingData_->moveparam.dstOffset[i]);
    }

    OP_LOGD(
        context_,
        "Roll tilingdata is: needCoreNum = %ld, dimNum = %ld, basicElements = %ld, maxElements = "
        "%ld, perCoreElements = %ld, lastCoreElements = %ld",
        tilingData_->needCoreNum, tilingData_->dimNum, tilingData_->basicElements, tilingData_->maxElements,
        tilingData_->perCoreElements, tilingData_->lastCoreElements);
    for (int32_t i = 0; i < dimNum_; i++) {
        OP_LOGD(
            context_, "shape[%d] is: %ld, shifts[%d] is: %ld, strides[%d] is: %ld", i, tilingData_->shapes[i], i,
            tilingData_->shifts[i], i, tilingData_->strides[i]);
    }
    return;
}

uint64_t RollTilingClass::GetTilingKey() const
{
    return tilingKey;
}

ge::graphStatus RollTilingArch35(gert::TilingContext* context)
{
    OP_LOGI(context->GetNodeName(), "tiling running.");

    const RollCompileInfoArch35* compile_info =
        reinterpret_cast<const RollCompileInfoArch35*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
    OP_LOGD(context->GetNodeName(), "runing regbase soc version tiling func");
    RollTilingClass tiling(context);
    return tiling.DoTiling();
}

ge::graphStatus TilingPrepare4RollArch35(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "begin to get compile info for Roll.");
    auto compile_info = context->GetCompiledInfo<RollCompileInfoArch35>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
    OP_LOGD(context->GetNodeName(), "Roll on regbase soc version no need to parse compile info.");
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compile_info->core_num = ascendcPlatform.GetCoreNumAiv();
    OP_LOGD(context->GetNodeName(), "core_num is: %d.", compile_info->core_num);
    compile_info->is_ascendc = true;
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compile_info->ub_size = static_cast<int64_t>(ubSize);
    OP_LOGD(context->GetNodeName(), "ub_size is: %d.", compile_info->ub_size);
    compile_info->socVersion = ascendcPlatform.GetSocVersion();
    OP_LOGD(context->GetNodeName(), "TilingPrepare4Roll GRAPH_SUCCESS.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Roll).Tiling(RollTilingArch35).TilingParse<RollCompileInfoArch35>(TilingPrepare4RollArch35);
} // namespace optiling
