/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file select_simt.h
 * \brief
 */
#ifndef SELECT_SIMT_H
#define SELECT_SIMT_H

#include "kernel_operator.h"
#include "select_struct.h"
#ifdef __CCE_AICORE__
#include "simt_api/asc_simt.h"
#endif

namespace SelectOp {
using namespace AscendC;

constexpr uint32_t THREAD_NUM_LAUNCH_BOUND = 2048;

template <typename T>
class SelectSimt {
 public:
  __aicore__ inline SelectSimt(){};
  __aicore__ inline void Init(GM_ADDR condition, GM_ADDR x1, GM_ADDR x2, GM_ADDR y, 
                              const SelectSimtTilingData* tilingData);
  __aicore__ inline void Process();

 private:
  static __simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND) inline void OpSelectSimt(int32_t needCoreNum, 
    int32_t threadNum, int64_t aSize, uint64_t bSize, int64_t currentCoreElements, uint64_t m, uint64_t shift, 
    uint64_t xyIndexBase, __gm__ uint8_t* condition,__gm__ T* x1, __gm__ T* x2, __gm__ T* y);

 private:
  GlobalTensor<uint8_t> conditionGm_;
  GlobalTensor<T> x1Gm_;
  GlobalTensor<T> x2Gm_;
  GlobalTensor<T> yGm_;

  const SelectSimtTilingData* tilingData_ = nullptr;
};

template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND) inline void SelectSimt<T>::OpSelectSimt(int32_t needCoreNum, 
    int32_t threadNum, int64_t aSize, uint64_t bSize, int64_t currentCoreElements, uint64_t m, uint64_t shift, 
    uint64_t xyIndexBase, __gm__ uint8_t* condition,__gm__ T* x1, __gm__ T* x2, __gm__ T* y) {
  for (uint64_t index = static_cast<uint64_t>(threadIdx.x); index < currentCoreElements;
      index += static_cast<uint32_t>(blockDim.x)) {
    uint64_t xyIndex = xyIndexBase + index;
    uint64_t conditionIdx = Simt::UintDiv(xyIndex, m, shift);
    y[xyIndex] = condition[conditionIdx] ? x1[xyIndex] : x2[xyIndex];
  }
}


template <typename T>
__aicore__ inline void SelectSimt<T>::Init(GM_ADDR condition, GM_ADDR x1, GM_ADDR x2, 
                                          GM_ADDR y, const SelectSimtTilingData* tilingData) {
  tilingData_ = tilingData;
  conditionGm_.SetGlobalBuffer((__gm__ uint8_t*)condition);
  x1Gm_.SetGlobalBuffer((__gm__ T*)x1);
  x2Gm_.SetGlobalBuffer((__gm__ T*)x2);
  yGm_.SetGlobalBuffer((__gm__ T*)y);
}

template <typename T>
__aicore__ inline void SelectSimt<T>::Process() {
  int32_t blockIdx = static_cast<int32_t>(GetBlockIdx());
  int32_t needCoreNum = static_cast<int32_t>(tilingData_->needCoreNum);
  int32_t threadNum = static_cast<int32_t>(tilingData_->threadNum);
  int64_t aSize = static_cast<int64_t>(tilingData_->aSize);
  uint64_t bSize = static_cast<uint64_t>(tilingData_->bSize);
  int64_t currentCoreElements = static_cast<int64_t>(tilingData_->perCoreElements);
  if (blockIdx == tilingData_->needCoreNum - 1) {
    currentCoreElements = static_cast<int64_t>(tilingData_->lastCoreElements);
  }
  uint64_t m {0};
  uint64_t shift {0};
  uint64_t xyIndexBase = blockIdx * tilingData_->perCoreElements;

  GetUintDivMagicAndShift(m, shift, bSize);

  asc_vf_call<OpSelectSimt>(dim3(threadNum), needCoreNum, threadNum, aSize, bSize, currentCoreElements, m, shift, 
  xyIndexBase, (__gm__ uint8_t*) (conditionGm_.GetPhyAddr()), (__gm__ T*) (x1Gm_.GetPhyAddr()), (__gm__ T*) (x2Gm_.GetPhyAddr()),
  (__gm__ T*) (yGm_.GetPhyAddr()));
}
}
#endif  // SELECT_SIMT_H
