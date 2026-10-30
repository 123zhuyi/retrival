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
 * \file vf_antiquant.h
 * \brief
 */
#ifndef VF_ANTIQUANT_H
#define VF_ANTIQUANT_H
namespace AscendC {
template <typename Q_T, typename KV_T, uint32_t baseSize, bool hasOffset>
__aicore__ inline void PfaAntiquantVFImplW8D64(LocalTensor<KV_T>& antiqInUb, LocalTensor<Q_T>& antiqResUb,
                                         LocalTensor<Q_T>& antiqOffsetUb, LocalTensor<Q_T>& antiqScaleUb,
                                         uint32_t dealRowCount) {
  static_assert(baseSize == 64); // 64 :headSize

  __VEC_SCOPE__ {
    __ubuf__ uint8_t* ubSrcAddr = (__ubuf__ uint8_t*)(antiqInUb.GetPhyAddr());
    __ubuf__ Q_T* ubDstAddr = (__ubuf__ Q_T*)(antiqResUb.GetPhyAddr());
    __ubuf__ Q_T* ubOffsetAddr = (__ubuf__ Q_T*)antiqOffsetUb.GetPhyAddr();
    __ubuf__ Q_T* ubScaleAddr = (__ubuf__ Q_T*)antiqScaleUb.GetPhyAddr();

    MicroAPI::RegTensor<KV_T> vKvData;
    MicroAPI::RegTensor<Q_T> vOffset;
    MicroAPI::RegTensor<Q_T> vScale;
    MicroAPI::RegTensor<Q_T> vRes;

    MicroAPI::MaskReg kvMaskAll = MicroAPI::CreateMask<KV_T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg qMaskLower64 = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::VL64>();
    MicroAPI::MaskReg qMaskLower128 = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::VL128>();
    MicroAPI::MaskReg qMaskAll = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::ALL>(); // Q_T 所有元素（共128个）
    MicroAPI::MaskReg qMaskHigher64;

    static constexpr MicroAPI::CastTrait castTrait = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                      MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    uint32_t blockStride = dealRowCount + 1;
    uint32_t repeatStride = 1;
    uint32_t fp16ElementCntPerBlock = 16; // 16:size of fp16

    MicroAPI::MaskXor(qMaskHigher64, qMaskLower64, qMaskAll, qMaskAll); // qMaskAll与qMaskLower64异或得到qMaskHigher64

    __ubuf__ Q_T* ubDstAddrOdd = ubDstAddr;
    __ubuf__ uint8_t* ubSrcTemp = ubSrcAddr;

    if constexpr (hasOffset) {
      MicroAPI::DataCopy<Q_T, MicroAPI::LoadDist::DIST_NORM>(vOffset, ubOffsetAddr);
    }
    MicroAPI::DataCopy<Q_T, MicroAPI::LoadDist::DIST_NORM>(vScale, ubScaleAddr);

    // 对D64优化，相邻2行合并计算；+1兼容奇数行场景
    for (uint16_t i = 0; i < (dealRowCount + 1) / 2; i++) {
      __ubuf__ Q_T* ubDstAddrEven = (__ubuf__ Q_T*)(antiqResUb.GetPhyAddr());
      MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK_B8>(
          (MicroAPI::RegTensor<uint8_t>&)vKvData, ubSrcTemp, baseSize * 2);

      if constexpr (std::is_same<Q_T, bfloat16_t>::value) {
        //Bf16 没有指令直接支持
      } else {
        MicroAPI::Cast<Q_T, KV_T, castTrait>(vRes, vKvData, kvMaskAll);
      }

      if constexpr (hasOffset) {
        MicroAPI::Add<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes, vRes, vOffset, qMaskLower128);
      }

      MicroAPI::Mul<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes, vRes, vScale, qMaskLower128);

      MicroAPI::DataCopy<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
          ubDstAddrOdd, vRes, blockStride, repeatStride, qMaskLower64);

      // 额外+1是因为有解bank冲突的行
      ubDstAddrEven += (i * 2 + 1) * fp16ElementCntPerBlock - ((dealRowCount + 1) * baseSize); // 2:相邻2行合并计算
      ubDstAddrOdd += fp16ElementCntPerBlock;

      MicroAPI::DataCopy<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
          ubDstAddrEven, vRes, blockStride, repeatStride, qMaskHigher64);
    }
  }
}

template <typename Q_T, typename KV_T, uint32_t baseSize, bool hasOffset = false>
__aicore__ inline void PfaAntiquantVFImplW8Norm(LocalTensor<KV_T>& antiqInUb, LocalTensor<Q_T>& antiqResUb,
                                         LocalTensor<Q_T>& antiqOffsetUb, LocalTensor<Q_T>& antiqScaleUb,
                                         uint32_t dealRowCount) {
  static_assert(baseSize % 128 == 0); // 128:headSize

  __VEC_SCOPE__ {
    __ubuf__ uint8_t* ubSrcAddr = (__ubuf__ uint8_t*)(antiqInUb.GetPhyAddr());
    __ubuf__ Q_T* ubDstAddr = (__ubuf__ Q_T*)(antiqResUb.GetPhyAddr());
    __ubuf__ Q_T* ubOffsetAddr = (__ubuf__ Q_T*)antiqOffsetUb.GetPhyAddr();
    __ubuf__ Q_T* ubScaleAddr = (__ubuf__ Q_T*)antiqScaleUb.GetPhyAddr();

    MicroAPI::RegTensor<KV_T> vKvData;
    MicroAPI::RegTensor<Q_T> vOffset;
    MicroAPI::RegTensor<Q_T> vScale;
    MicroAPI::RegTensor<Q_T> vRes;

    MicroAPI::MaskReg kvMaskAll = MicroAPI::CreateMask<KV_T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg qMaskAll = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::ALL>(); // Q_T 所有元素（共128个）

    static constexpr MicroAPI::CastTrait castTrait = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                      MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    uint32_t blockStride = dealRowCount + 1;
    uint32_t repeatStride = 1;
    const uint32_t loops = baseSize / 128; // 128:step

    for (uint16_t j = 0; j < loops; j++) {
      __ubuf__ Q_T* ubDstAddrTmp = ubDstAddr + blockStride * 128 * j; // 128:step
      __ubuf__ uint8_t* ubSrcTemp = ubSrcAddr + j * 128; // 128:step

      if constexpr (hasOffset) {
        MicroAPI::DataCopy<Q_T, MicroAPI::LoadDist::DIST_NORM>(vOffset, ubOffsetAddr + j * 128); // 128:step
      }
      MicroAPI::DataCopy<Q_T, MicroAPI::LoadDist::DIST_NORM>(vScale, ubScaleAddr + j * 128); // 128:step

      for (uint16_t i = 0; i < dealRowCount; i++) {
        MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK_B8>(
            (MicroAPI::RegTensor<uint8_t>&)vKvData, ubSrcTemp, baseSize);

        if constexpr (std::is_same<Q_T, bfloat16_t>::value) {
          // Bf16 没有指令直接支持
        } else {
          MicroAPI::Cast<Q_T, KV_T, castTrait>(vRes, vKvData, kvMaskAll);
        }

        if constexpr (hasOffset) {
          MicroAPI::Add<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes, vRes, vOffset, qMaskAll);
        }

        MicroAPI::Mul<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vRes, vRes, vScale, qMaskAll);

        MicroAPI::DataCopy<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ubDstAddrTmp, vRes, blockStride, repeatStride, qMaskAll);
      }
    }
  }
}

template <typename Q_T, typename KV_T, uint32_t baseSize, bool hasOffset = false>
__aicore__ inline void AntiquantVFImplPfa(LocalTensor<KV_T>& antiqInUb, LocalTensor<Q_T>& antiqResUb,
                                       LocalTensor<Q_T>& antiqOffsetUb, LocalTensor<Q_T>& antiqScaleUb,
                                       uint32_t dealRowCount) {
  if constexpr (baseSize == 64) { // 64: headSize
    PfaAntiquantVFImplW8D64<Q_T, KV_T, baseSize, hasOffset>(antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb,
                                                            dealRowCount);
  } else {
    PfaAntiquantVFImplW8Norm<Q_T, KV_T, baseSize, hasOffset>(antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb,
                                                           dealRowCount);
  }
}

template <typename Q_T, typename KV_T, uint32_t baseSize, bool hasOffset = false>
__aicore__ inline void PfaAntiquantVF(LocalTensor<KV_T>& antiqInUb, LocalTensor<Q_T>& antiqResUb,
                                   LocalTensor<Q_T>& antiqOffsetUb, LocalTensor<Q_T>& antiqScaleUb,
                                   uint32_t dealRowCount) {
  AntiquantVFImplPfa<Q_T, KV_T, baseSize, hasOffset>(antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb, dealRowCount);
}

};  // namespace AscendC
#endif // VF_ANTIQUANT_H