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
 * \file fag_copy_cube_in_post_preload.h
 * \brief
 */
 
 #ifndef FAG_COPY_CUBE_IN_POST_ROPE_H
 #define FAG_COPY_CUBE_IN_POST_ROPE_H
  
 #include "../../cube_in_buffer/fag_cube_in_buffer_post.h"
  
 namespace AscendC {
 namespace Impl {
 namespace Detail {
 template <typename IMPL, class INPUT_TYPE, const auto &MM_CFG>
 class FagCopyCubeInS1S2PostPreload : public CopyCubeInBase<IMPL, MM_CFG, INPUT_TYPE> {
     MATMUL_USE_MODULE_ON(CubeInBuffer, INPUT_TYPE::TAG);
     MATMUL_USE_MODULE(Context);
     MATMUL_USE_MODULE_ON(CopyCubeInParams, INPUT_TYPE::TAG);
     MATMUL_USE_MODULE_ON(MatmulTensorInfo, INPUT_TYPE::TAG);
     MATMUL_USE_MODULE_ON(DataCopyUtils, INPUT_TYPE::TAG);
     MATMUL_USE_MODULE(MatmulUserDefineInfo);
     using SRC_T = typename INPUT_TYPE::T;
     using TRANS_T = typename INPUT_TYPE::TRANS_T;
  
 public:
     __aicore__ inline FagCopyCubeInS1S2PostPreload()
     {
     }
  
     __aicore__ inline void Init()
     {
         totalRow_ = MATMUL_MODULE(CopyCubeInParams)->GetTotalRow();
         totalCol_ = MATMUL_MODULE(CopyCubeInParams)->GetTotalCol();
         isTranspose_ = INPUT_TYPE::isTrans;
     }
  
     __aicore__ inline void SetInput(const LocalTensor<SRC_T> &localMatrix, bool isTranspose)
     {
         isTranspose_ = isTranspose;
         if (isTranspose_) {
             int32_t temp = 0;
             temp = orgWidth_;
             orgWidth_ = orgHeight_;
             orgHeight_ = temp;
         }
         MATMUL_MODULE(MatmulTensorInfo)->SetLocalTensor(localMatrix, isTranspose);
     }
  
     __aicore__ inline void SetInput(const GlobalTensor<SRC_T> &globalMatrix, bool isTranspose)
     {
         isTranspose_ = isTranspose;
         if (isTranspose_) {
             int32_t temp = 0;
             temp = orgWidth_;
             orgWidth_ = orgHeight_;
             orgHeight_ = temp;
         }
         MATMUL_MODULE(MatmulTensorInfo)->SetGlobalTensor(globalMatrix, isTranspose);
     }
  
     __aicore__ inline LocalTensor<SRC_T> LoadData(int curRow, int curCol, int tileHeight, int tileWidth,
                                                   int batchNum = 1)
     {
         // 获取用户设置的flag
         FagTscmRopeFlagData flag = MATMUL_MODULE(MatmulUserDefineInfo)->GetSelfDefineData();
         LocalTensor<SRC_T> l1;
         MATMUL_MODULE(Context)->rightMatrixEncodingTableIdx = static_cast<int32_t>(flag.rightMatrixEncodingTableIdx);
  
         if constexpr (INPUT_TYPE::TAG == InputTypeTag::B) {
             uint8_t tscmIndex = MM2_ENCODING_TABLE[flag.rightMatrixEncodingTableIdx];
             l1.SetAddr(gTscmArray[tscmIndex].srcAddr);
             return l1;
         } else {
             l1.SetAddr(MATMUL_MODULE(MatmulTensorInfo)->GetLocalTensor().address_);
             return l1;
         }
     }
  
     __aicore__ inline void ClearLoadData(const LocalTensor<TRANS_T> &aMatrix = LocalTensor<TRANS_T>{}, int32_t curRow = 0,
                                          int32_t curCol = 0)
     {
         if constexpr (!PhyPosIsUB(INPUT_TYPE::pos) && !PhyPosIsL1(INPUT_TYPE::pos)) {
             int posL1 = MATMUL_MODULE(CubeInBuffer)->GetIterIndex(curRow, curCol);
             MATMUL_MODULE(CubeInBuffer)->FreeTensor(posL1, aMatrix);
         }
     }
  
     // Destroy
     __aicore__ inline void Destroy()
     {
         if constexpr (!PhyPosIsL1(INPUT_TYPE::pos)) {
             MATMUL_MODULE(CubeInBuffer)->Destroy();
         }
     }
  
 public:
     bool isTranspose_{false};
  
 private:
     int32_t totalRow_;
     int32_t totalCol_;
     int32_t orgWidth_;
     int32_t orgHeight_;
 };
  
 } // namespace Detail
 } // namespace Impl
 } // namespace AscendC
  
 #endif