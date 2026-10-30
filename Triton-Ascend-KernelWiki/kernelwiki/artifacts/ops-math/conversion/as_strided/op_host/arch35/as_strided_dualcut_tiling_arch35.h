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
 * \file as_strided_dualcut_tiling_arch35.h
 * \brief as_strided_dualcut_tiling_arch35
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_AS_STRIDED_DUALCUT_TILING_ARCH35_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_AS_STRIDED_DUALCUT_TILING_ARCH35_H_
#include <cstdint>
#include <vector>
#include "log/log.h"
#include "util/math_util.h"
namespace optiling {
const int64_t SHAPE_ARRAY_LEN = 10;
const int64_t SHAPE_NDDMA_LEN = 5;
constexpr int64_t BLOCK_BYTES = 32;

struct AxisInf {
    AxisInf(int64_t dim_, int64_t stride_, int64_t idx_, gert::TilingContext* context)
        : context_(context), dim(dim_), stride(stride_), idx(idx_), code(1 << idx_), conter(0)
    {
    }

    void PrintDebug()
    {
        OP_LOGI(context_, "(%ld, %ld, %u, 0x%x)", dim, stride, idx, code);
        this->conter += 1;
    }

    void ResetInf(int64_t dim_, int64_t stride_, int64_t idx_)
    {
        this->dim = dim_;
        this->stride = stride_;
        this->idx = idx_;
        this->code = 1 << idx_;
    }

    gert::TilingContext* context_ = nullptr;
    int64_t dim;
    int64_t stride;
    uint32_t idx;
    unsigned int code;
    int conter;
};

void PrintAxisList(std::vector<AxisInf>& axisList)
{
    for (auto tmpAxis : axisList) {
        tmpAxis.PrintDebug();
    }
}

class AxisCutter {
public:
    AxisCutter(std::vector<AxisInf>& axisList_, int alignedFactor_, gert::TilingContext* context)
    {
        context_ = context;
        this->initDimNums = axisList_.size();
        this->alignedFactor = alignedFactor_;
        for (uint32_t i = 0; i < this->initDimNums; i++) {
            if (axisList_[i].idx < this->initDimNums - 1) {
                this->axisList.push_back(axisList_[i]);
            } else {
                int64_t alignedFinalDim =
                    (axisList_[i].dim + this->alignedFactor - 1) / this->alignedFactor * this->alignedFactor;
                this->originFinalDim = axisList_[i].dim;
                this->axisList.push_back(AxisInf(alignedFinalDim, axisList_[i].stride, axisList_[i].idx, context));
            }
        }
    }

    void PrintDebug();
    void FindCutAxis(int ubBound);
    void RemoveAxis(unsigned int setCode);
    AxisInf* GetCutAxis();

public:
    gert::TilingContext* context_ = nullptr;

    int conter{0};
    uint32_t initDimNums{0};
    int alignedFactor{0};
    int originFinalDim{0};
    unsigned int ubSet{0};
    unsigned int cutSet{0};
    int cutIdx = -1;
    bool cutFinal{false};
    AxisInf innerCutAxis = AxisInf(0, 0, 0, context_);
    AxisInf outerCutAxis = AxisInf(0, 0, 0, context_);
    AxisInf* cutAxisPtr{NULL};
    std::vector<AxisInf> axisList;
};

void AxisCutter::FindCutAxis(int ubBound)
{
    int idx = (this->axisList).size() - 1;
    int64_t prod = 1;
    this->ubSet = 0;
    for (; idx >= 0; idx--) {
        prod *= this->axisList[idx].dim;
        if (prod > ubBound) {
            break;
        }
        this->ubSet = this->ubSet | this->axisList[idx].code;
    }
    this->cutSet = (idx >= 0) ? this->axisList[idx].code : 0;
    this->cutIdx = idx;
    this->cutAxisPtr = (this->cutIdx >= 0) ? &this->axisList[cutIdx] : &this->axisList[0];
    this->cutFinal = (cutAxisPtr->idx == (this->initDimNums - 1));
    int64_t cutDim = cutAxisPtr->dim;
    prod = prod / cutDim;

    int64_t innerDim = ubBound / prod;
    int64_t outerDim = (cutDim + innerDim - 1) / innerDim;

    this->innerCutAxis.ResetInf(innerDim, cutAxisPtr->stride, cutAxisPtr->idx);
    this->outerCutAxis.ResetInf(outerDim, innerDim * cutAxisPtr->stride, cutAxisPtr->idx);
}

void AxisCutter::RemoveAxis(unsigned int setCode)
{
    std::vector<AxisInf> tmpAxisList;
    for (auto tmpAxis : this->axisList) {
        if ((tmpAxis.code & setCode) == 0) {
            tmpAxisList.push_back(tmpAxis);
        }
    }
    this->axisList.clear();
    this->axisList = tmpAxisList;
}

void AxisCutter::PrintDebug()
{
    OP_LOGI(context_, "AxisList: ");
    PrintAxisList(this->axisList);
    OP_LOGI(context_, "InnerCutAxis: ");
    this->innerCutAxis.PrintDebug();
    OP_LOGI(context_, "OuterCutAxis: ");
    this->outerCutAxis.PrintDebug();
    this->conter += 1;
}

AxisInf* AxisCutter::GetCutAxis()
{
    return &this->axisList[this->cutIdx];
}

class DualCutAxisSeeker {
public:
    DualCutAxisSeeker(int64_t* shape, int64_t* strides, int dimNums_, int dtSize_, gert::TilingContext* context)
    {
        context_ = context;
        this->dtSize = dtSize_;
        if(dtSize_ <= 0) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "input",
                std::to_string(dtSize_).c_str(),
                "The dtype size of input must be greater than 0.");
            dtSize_ = 1;
        }
        this->alignedFactor = BLOCK_BYTES / dtSize_;
        this->dimNums = dimNums_;
        for (int i = 0; i < dimNums_; i++) {
            this->outputAxis.push_back(AxisInf(shape[i], strides[i], i, context));
            this->reorderAxis.push_back(AxisInf(shape[i], strides[i], i, context));
        }
        std::stable_sort(this->reorderAxis.begin(), this->reorderAxis.end(), [](const AxisInf a, const AxisInf b) {
            return (a.stride == b.stride) ? a.dim > b.dim : (a.stride > b.stride);
        });
        this->outputCutter = new AxisCutter(this->outputAxis, alignedFactor, context);
        this->inputCutter = new AxisCutter(this->reorderAxis, alignedFactor, context);
    }

    ~DualCutAxisSeeker()
    {
        OP_LOGI(context_, "Delete AxisCutter Start");
        delete this->outputCutter;
        delete this->inputCutter;
        OP_LOGI(context_, "Delete AxisCutter End");
    }

    bool FindDualCutAxis(int ubSize, int bufferNum);
    int64_t ComputeSetAxisProd(unsigned int axisSet);
    int64_t ComputeRemainUB(int64_t ubSize, unsigned int axisSet);
    unsigned int FindCommonUbAxis(const AxisCutter* cutter1, const AxisCutter* cutter2);
    unsigned int FindCommonCutAxis(const AxisCutter* cutter1, const AxisCutter* cutter2);
    void PrintDebug();
    void ComputeCutAxis(unsigned int ubAxisSet, unsigned ubNum);
    void UpdateCommonAxis(unsigned int commonSet);

    bool CutAxis(unsigned int ubAxisSet, int64_t remainNums);
    bool CutTwoAxis(
        unsigned int ubAxisSet, int64_t remainNums, std::vector<AxisInf>& innerAxis, std::vector<AxisInf>& outerAxis);
    bool CutOneAxis(
        unsigned int ubAxisSet, int64_t remainNums, std::vector<AxisInf>& innerAxis, std::vector<AxisInf>& outerAxis);
    void SetCutAxisIdx();
    void GenTilingData();
    void ComputeBlockTiling(int coreNum);

    int ComputeOutputShape(AxisInf& axis);

public:
    gert::TilingContext* context_ = nullptr;

    std::vector<AxisInf> outputAxis;
    std::vector<AxisInf> reorderAxis;
    //  The gmAxis is for multi core loop
    std::vector<AxisInf> gmAxis;
    std::vector<AxisInf> ubAxis;

    AxisCutter* outputCutter = nullptr;
    AxisCutter* inputCutter = nullptr;

    int dtSize{0};
    int alignedFactor{0};
    int dimNums{0};
    int conter{0};

    // Tiling data:
    int32_t gmShape[SHAPE_ARRAY_LEN] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    int32_t gmInStride[SHAPE_ARRAY_LEN] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int32_t gmOutStride[SHAPE_ARRAY_LEN] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    int32_t ubShape[SHAPE_NDDMA_LEN] = {1, 1, 1, 1, 1};
    int64_t ubInStride[SHAPE_NDDMA_LEN] = {0};
    int32_t ubOutStride[SHAPE_NDDMA_LEN] = {0};

    int cutAxisNums{0};
    int outerAxisNums{0};

    int innerAxisNums{0};
    int cutAxisIdx01{-1};
    int cutAxisIdx02{-1};
    int cutAxisTail01{0};
    int cutAxisTail02{0};

    int dimsPerCore{0};
    int useCoreNum{0};
    int dimsPerCoreTail{0};
};

void DualCutAxisSeeker::PrintDebug()
{
    OP_LOGI(context_, "OutputCutter:");
    this->outputCutter->PrintDebug();
    OP_LOGI(context_, "InputCutter:");
    this->inputCutter->PrintDebug();

    OP_LOGI(context_, "OutputAxis:");
    PrintAxisList(this->outputAxis);
    OP_LOGI(context_, "InputAxis:");
    PrintAxisList(this->reorderAxis);
    this->conter += 1;
}

void DualCutAxisSeeker::SetCutAxisIdx()
{
    int cutAxisIdx[2] = {-1, -1};
    int cutAxisTail[2] = {-1, -1};

    AxisInf* outerCutAxisPtr = nullptr;

    int baseIdx = SHAPE_NDDMA_LEN - this->ubAxis.size();

    for (int i = 0; i < this->cutAxisNums; i++) {
        int outerId = this->outerAxisNums - this->cutAxisNums + i;
        outerCutAxisPtr = &this->gmAxis[outerId];
        for (uint32_t j = 0; j < this->ubAxis.size(); j++) {
            if (this->ubAxis[j].idx == outerCutAxisPtr->idx) {
                cutAxisIdx[i] = j + baseIdx;
                cutAxisTail[i] =
                    this->outputAxis[outerCutAxisPtr->idx].dim - (outerCutAxisPtr->dim - 1) * this->ubAxis[j].dim;
                break;
            }
        }
    }
    this->cutAxisIdx01 = cutAxisIdx[0];
    this->cutAxisIdx02 = cutAxisIdx[1];
    this->cutAxisTail01 = cutAxisTail[0];
    this->cutAxisTail02 = cutAxisTail[1];
}

void DualCutAxisSeeker::GenTilingData()
{
    this->outerAxisNums = this->gmAxis.size();
    this->innerAxisNums = this->ubAxis.size();
    for (int i = 0; i < this->outerAxisNums; i++) {
        gmShape[i] = this->gmAxis[i].dim;
        gmInStride[i] = this->gmAxis[i].stride;
        gmOutStride[i] = ComputeOutputShape(this->gmAxis[i]);
    }
    int baseIdx = SHAPE_NDDMA_LEN - this->ubAxis.size();
    for (uint32_t i = 0; i < this->ubAxis.size(); i++) {
        ubShape[baseIdx + i] = this->ubAxis[i].dim;
        ubInStride[baseIdx + i] = this->ubAxis[i].stride;
        ubOutStride[baseIdx + i] = ComputeOutputShape(this->ubAxis[i]);
    }
    // update cut outer axis
    for (int i = 0; i < this->cutAxisNums; i++) {
        uint32_t idx = this->gmAxis[static_cast<size_t>(this->outerAxisNums - 1 - i)].idx;
        for (auto tmpAxis : this->ubAxis) {
            if (tmpAxis.idx == idx) {
                this->gmOutStride[static_cast<size_t>(this->outerAxisNums - 1 - i)] *= tmpAxis.dim;
                break;
            }
        }
    }
    SetCutAxisIdx();
}

void DualCutAxisSeeker::ComputeBlockTiling(int coreNum)
{
    int64_t totalOuterDim = 1;
    for (auto tmpAxis : this->gmAxis) {
        totalOuterDim *= tmpAxis.dim;
    }
    this->dimsPerCore = Ops::Base::CeilDiv(totalOuterDim, static_cast<int64_t>(coreNum));
    this->useCoreNum = (totalOuterDim + this->dimsPerCore - 1) / this->dimsPerCore;
    this->dimsPerCoreTail = totalOuterDim - (this->useCoreNum - 1) * this->dimsPerCore;
}

int DualCutAxisSeeker::ComputeOutputShape(AxisInf& axis)
{
    int idx = axis.idx;
    int prod = 1;
    for (uint32_t i = idx + 1; i < this->outputAxis.size(); i++) {
        prod *= this->outputAxis[i].dim;
    }
    return prod;
}

bool DualCutAxisSeeker::FindDualCutAxis(int ubSize, int bufferNum)
{
    if(bufferNum == 0) {
        OP_LOGE(context_, "the bufferNum is equal zero");
        bufferNum = 2;
    }
    int ubNum = ubSize / this->dtSize / bufferNum;
    int ubBound = 0;
    unsigned int ubAxisSet = 0;
    unsigned int joinUbAxisSet = 0;
    if (this->outputAxis[dimNums - 1].dim <= alignedFactor) {
        OP_LOGW(context_, "Last dim is smaller than 32B, no need do dual cut!");
        return false;
    }
    for (int findLoops = SHAPE_ARRAY_LEN; findLoops >= 0; findLoops--) {
        int64_t remainUB = this->ComputeRemainUB(ubNum, ubAxisSet);
        ubBound = static_cast<int>(std::floor(std::sqrt(remainUB)));
        if (ubBound <= 0) {
            OP_LOGI(context_, "ubBound is invalid.");
            break;
        }
        this->outputCutter->FindCutAxis(ubBound);
        this->inputCutter->FindCutAxis(ubBound);

        unsigned int commonUbAxisSet = this->FindCommonUbAxis(this->outputCutter, this->inputCutter);
        joinUbAxisSet = this->outputCutter->ubSet | this->inputCutter->ubSet;

        if (commonUbAxisSet == 0) {
            unsigned int commonCutAxisSet = this->FindCommonCutAxis(this->outputCutter, this->inputCutter);
            if (commonCutAxisSet == 0) {
                OP_LOGI(context_, "Cut two axis.");
                break;
            } else {
                int64_t currentJoinUbSize = this->ComputeSetAxisProd(joinUbAxisSet | ubAxisSet | commonCutAxisSet);
                if (currentJoinUbSize >= ubNum) {
                    OP_LOGI(context_, "Cut one axis.");
                    break;
                } else {
                    ubAxisSet = ubAxisSet | commonCutAxisSet;
                    this->UpdateCommonAxis(commonCutAxisSet);
                }
            }
        } else {
            ubAxisSet = ubAxisSet | commonUbAxisSet;
            this->UpdateCommonAxis(commonUbAxisSet);
        }
    }
    ubAxisSet = ubAxisSet | joinUbAxisSet;
    int64_t remainNums = this->ComputeRemainUB(ubNum, ubAxisSet);
    OP_LOGD(context_, "remainNums = %ld", remainNums);
    return this->CutAxis(ubAxisSet, remainNums);
}

bool DualCutAxisSeeker::CutAxis(unsigned int ubAxisSet, int64_t remainNums)
{
    std::vector<AxisInf> innerAxis;
    std::vector<AxisInf> outerAxis;

    bool successCut = true;
    if (this->outputCutter->cutSet != this->inputCutter->cutSet) {
        OP_LOGD(context_, "CutTwoAxis");
        successCut = this->CutTwoAxis(ubAxisSet, remainNums, innerAxis, outerAxis);
    } else {
        OP_LOGD(context_, "CutOneAxis");
        successCut = this->CutOneAxis(ubAxisSet, remainNums, innerAxis, outerAxis);
    }
    OP_CHECK_IF(!successCut, OP_LOGW(context_, "cutting failed, back to Sole Cut."), return false);

    this->cutAxisNums = innerAxis.size();

    unsigned int cutAxisSet = this->inputCutter->cutSet | this->outputCutter->cutSet;
    for (auto tmpAxis : this->outputAxis) {
        if ((tmpAxis.code & ubAxisSet) != 0) { // in ubSet
            this->ubAxis.push_back(tmpAxis);
        } else if ((tmpAxis.code & cutAxisSet) == 0) { // not in ubSet, not cut
            this->gmAxis.push_back(tmpAxis);
        }
    }

    // move big stride axis into gm if more than 5 axis in ub
    if (this->ubAxis.size() + innerAxis.size() > SHAPE_NDDMA_LEN) {
        std::sort(this->ubAxis.begin(), this->ubAxis.end(), [](const AxisInf a, const AxisInf b) {
            return a.stride == b.stride ? a.dim > b.dim : a.stride < b.stride;
        });
        int totalSize = this->ubAxis.size() + innerAxis.size();
        for (int i = 0; i < totalSize - SHAPE_NDDMA_LEN; i++) {
            // final axis should in ub, if get final axis, push back it in to innerAxis.
            if (this->ubAxis[this->ubAxis.size() - 1].idx == this->outputAxis.size() - 1) {
                innerAxis.push_back(this->ubAxis[this->ubAxis.size() - 1]);
                i--;
            } else {
                this->gmAxis.push_back(this->ubAxis[this->ubAxis.size() - 1]);
            }
            this->ubAxis.pop_back();
        }
    }

    for (auto tmpAxis : innerAxis) {
        this->ubAxis.push_back(tmpAxis);
    }

    std::sort(this->gmAxis.begin(), this->gmAxis.end(), [](const AxisInf a, const AxisInf b) { return a.idx < b.idx; });
    std::sort(this->ubAxis.begin(), this->ubAxis.end(), [](const AxisInf a, const AxisInf b) { return a.idx < b.idx; });

    for (auto tmpAxis : outerAxis) {
        this->gmAxis.push_back(tmpAxis);
    }
    return true;
}

bool DualCutAxisSeeker::CutTwoAxis(
    unsigned int ubAxisSet, int64_t remainNums, std::vector<AxisInf>& innerAxis, std::vector<AxisInf>& outerAxis)
{
    OP_LOGI(context_, "cutTwo ubAxisSet: 0x%x", ubAxisSet);
    AxisInf outputCutOuterAixs(0, 0, 0, context_);
    AxisInf outputCutInnerAixs(0, 0, 0, context_);
    AxisInf inputCutOuterAixs(0, 0, 0, context_);
    AxisInf inputCutInnerAixs(0, 0, 0, context_);

    outputCutOuterAixs.ResetInf(
        this->outputCutter->outerCutAxis.dim, this->outputCutter->outerCutAxis.stride,
        this->outputCutter->outerCutAxis.idx);
    outputCutInnerAixs.ResetInf(
        this->outputCutter->innerCutAxis.dim, this->outputCutter->innerCutAxis.stride,
        this->outputCutter->innerCutAxis.idx);
    inputCutOuterAixs.ResetInf(
        this->inputCutter->outerCutAxis.dim, this->inputCutter->outerCutAxis.stride,
        this->inputCutter->outerCutAxis.idx);
    inputCutInnerAixs.ResetInf(
        this->inputCutter->innerCutAxis.dim, this->inputCutter->innerCutAxis.stride,
        this->inputCutter->innerCutAxis.idx);

    if (this->inputCutter->cutFinal || this->outputCutter->cutFinal) {
        if (this->inputCutter->cutFinal) {
            inputCutInnerAixs.dim = (inputCutInnerAixs.dim < alignedFactor) ?
                                        ((inputCutInnerAixs.dim + alignedFactor - 1) / alignedFactor * alignedFactor) :
                                        (inputCutInnerAixs.dim / alignedFactor * alignedFactor);
            OP_CHECK_IF(
                (inputCutInnerAixs.dim == 0),
                OP_LOGW(context_, "inputCutInnerAixs.dim get 0, back to Sole Cut."),
                return false);
            outputCutInnerAixs.dim = remainNums / inputCutInnerAixs.dim; // If get zero back to single cut
        } else if (this->outputCutter->cutFinal) {
            outputCutInnerAixs.dim =
                (outputCutInnerAixs.dim < alignedFactor) ?
                    ((outputCutInnerAixs.dim + alignedFactor - 1) / alignedFactor * alignedFactor) :
                    (outputCutInnerAixs.dim / alignedFactor * alignedFactor);
            OP_CHECK_IF(
                (outputCutInnerAixs.dim == 0),
                OP_LOGW(context_, "outputCutInnerAixs.dim get 0, back to Sole Cut."),
                return false);
            inputCutInnerAixs.dim = remainNums / outputCutInnerAixs.dim; // If get zero back to single cut
        }
        inputCutOuterAixs.dim =
            (this->inputCutter->cutAxisPtr->dim + inputCutInnerAixs.dim - 1) / inputCutInnerAixs.dim;
        outputCutOuterAixs.dim =
            (this->outputCutter->cutAxisPtr->dim + outputCutInnerAixs.dim - 1) / outputCutInnerAixs.dim;
        inputCutOuterAixs.stride = inputCutInnerAixs.dim * inputCutInnerAixs.stride;
        outputCutOuterAixs.stride = outputCutInnerAixs.dim * outputCutInnerAixs.stride;
    }

    OP_LOGD(context_, "CutTwoAxis success");
    outerAxis.push_back(outputCutOuterAixs);
    outerAxis.push_back(inputCutOuterAixs);
    innerAxis.push_back(outputCutInnerAixs);
    innerAxis.push_back(inputCutInnerAixs);

    return true;
}

bool DualCutAxisSeeker::CutOneAxis(
    unsigned int ubAxisSet, int64_t remainNums, std::vector<AxisInf>& innerAxis, std::vector<AxisInf>& outerAxis)
{
    OP_LOGI(context_, "cutOne ubAxisSet: 0x%x", ubAxisSet);
    int64_t innerDim = (this->inputCutter->cutFinal) ? (remainNums / alignedFactor * alignedFactor) : remainNums;
    
    // If zero, throw bad cut.
    OP_CHECK_IF(
        (innerDim == 0), OP_LOGW(context_, "innerDim get 0, back to Sole Cut."), return false);
    int64_t innerStride = this->inputCutter->cutAxisPtr->stride;
    int64_t axisIdx = this->inputCutter->cutAxisPtr->idx;

    int64_t outerDim = (this->inputCutter->cutAxisPtr->dim + innerDim - 1) / innerDim;
    int64_t outerStride = innerStride * innerDim;

    AxisInf cutInnerAixs(innerDim, innerStride, axisIdx, context_);
    AxisInf cutOuterAixs(outerDim, outerStride, axisIdx, context_);

    outerAxis.push_back(cutOuterAixs);
    innerAxis.push_back(cutInnerAixs);
    return true;
}

void DualCutAxisSeeker::UpdateCommonAxis(unsigned int commonSet)
{
    this->outputCutter->RemoveAxis(commonSet);
    this->inputCutter->RemoveAxis(commonSet);
}

int64_t DualCutAxisSeeker::ComputeSetAxisProd(unsigned int axisSet)
{
    int64_t factor = 1;
    for (auto tmpAxis : this->outputAxis) {
        if ((tmpAxis.code & axisSet) != 0) {
            factor *= tmpAxis.dim;
        }
    }
    return factor;
}

int64_t DualCutAxisSeeker::ComputeRemainUB(int64_t ubSize, unsigned int axisSet)
{
    uint32_t finalDimCode = this->outputAxis[dimNums - 1].code;
    int32_t finalDimAlign = this->outputAxis[dimNums - 1].dim;
    finalDimAlign = ((finalDimAlign + this->alignedFactor - 1) / this->alignedFactor) * this->alignedFactor;
    if ((axisSet & finalDimCode) > 0) {
        axisSet = axisSet ^ finalDimCode;
        return ubSize / (ComputeSetAxisProd(axisSet) * finalDimAlign);
    }
    return ubSize / ComputeSetAxisProd(axisSet);
}

unsigned int DualCutAxisSeeker::FindCommonUbAxis(const AxisCutter* cutter1, const AxisCutter* cutter2)
{
    unsigned int commonAxisUB = cutter1->ubSet & cutter2->ubSet;
    unsigned int cutCommonAxis1 = cutter1->cutSet & cutter2->ubSet;
    unsigned int cutCommonAxis2 = cutter2->cutSet & cutter1->ubSet;
    return commonAxisUB | cutCommonAxis1 | cutCommonAxis2;
}

unsigned int DualCutAxisSeeker::FindCommonCutAxis(const AxisCutter* cutter1, const AxisCutter* cutter2)
{
    return cutter1->cutSet & cutter2->cutSet;
}
} // namespace optiling

#endif
