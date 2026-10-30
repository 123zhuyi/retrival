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
 * \file masked_select_v3_apt.h
 * \brief masked_selectv3_apt kernel proc
 */
#ifndef MASKED_SELECT_V3_APT_H
#define MASKED_SELECT_V3_APT_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
using namespace AscendC;
namespace MaskedSelectV3 {
#define IS_1_BYTES_TYPE is_same<T, int8_t>::value || is_same<T, uint8_t>::value
#define IS_2_BYTES_TYPE                                                                     \
    is_same<T, int16_t>::value || is_same<T, uint16_t>::value || is_same<T, half>::value || \
        is_same<T, bfloat16_t>::value
#define IS_4_BYTES_TYPE is_same<T, int32_t>::value || is_same<T, uint32_t>::value || is_same<T, float>::value
#define IS_8_BYTES_TYPE is_same<T, int64_t>::value || is_same<T, uint64_t>::value || is_same<T, double>::value

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t DUPLICATE_ALIGN = 256;                         // 单位字节
constexpr uint64_t SHAPEOUT_DIMNUM_WITH_UINT64FLAG = 0x80000001; // 如果shapeout的类型为uint64，则要求在第32位置1

template <typename Tp, Tp v>
struct integral_constant {
    static constexpr Tp value = v;
};

using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;

template <typename, typename>
struct is_same : public false_type {};

template <typename Tp>
struct is_same<Tp, Tp> : public true_type {};

constexpr uint32_t SHAPEOUT_SIZE = 2;
constexpr uint32_t HEAD_BLOCK_SIZE = 64;
constexpr uint32_t OFFSET_SHIFT_BITS = 3; // offset偏移量移位输，<<3 等价于 *8

template <typename T>
class KernelMaskedSelectV3 {
public:
    __aicore__ inline KernelMaskedSelectV3() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR mask, GM_ADDR y, GM_ADDR shapeout, GM_ADDR workspace,
                                uint32_t formerNum, uint32_t formerLength, uint32_t formertileNum,
                                uint32_t formertileLength, uint32_t formerlasttileLength, uint32_t tailNum,
                                uint32_t tailLength, uint32_t tailtileNum, uint32_t tailtileLength,
                                uint32_t taillasttileLength)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->numBlocks = GetBlockNum();
        __gm__ T* globalWorkTensor = (__gm__ T*)((__gm__ uint64_t*)workspace +
                                                 this->numBlocks * (HEAD_BLOCK_SIZE / sizeof(uint64_t)));

        blockIdx = GetBlockIdx();
        if (blockIdx + 1 > this->numBlocks) {
            return;
        }
        this->formerNum = formerNum;
        this->formerLength = formerLength;
        this->formertileNum = formertileNum;
        this->formertileLength = formertileLength;
        this->formerlasttileLength = formerlasttileLength;

        this->tailNum = tailNum;
        this->tailLength = tailLength;
        this->tailtileNum = tailtileNum;
        this->tailtileLength = tailtileLength;
        this->taillasttileLength = taillasttileLength;

        if (blockIdx < this->formerNum) { // 分到大块核的处理
            this->tileLength = this->formertileLength;
            this->lasttileLength = this->formerlasttileLength;
            this->tileNum = this->formertileNum;
            xGlobal.SetGlobalBuffer((__gm__ T*)x + this->formerLength * blockIdx, this->formerLength);
            maskGlobal.SetGlobalBuffer((__gm__ uint8_t*)mask + this->formerLength * blockIdx, this->formerLength);
            workGlobal.SetGlobalBuffer(globalWorkTensor + this->formerLength * blockIdx, this->formerLength);
        } else { // 分到小块核的处理，需要处理的数据量比大核少alignNum个
            this->tileLength = this->tailtileLength;
            this->lasttileLength = this->taillasttileLength;
            this->tileNum = this->tailtileNum;

            xGlobal.SetGlobalBuffer(
                (__gm__ T*)x + this->formerLength * this->formerNum + this->tailLength * (blockIdx - this->formerNum),
                this->tailLength);
            maskGlobal.SetGlobalBuffer((__gm__ uint8_t*)mask + this->formerLength * this->formerNum +
                                           this->tailLength * (blockIdx - this->formerNum),
                                       this->tailLength);
            workGlobal.SetGlobalBuffer(globalWorkTensor + this->formerLength * this->formerNum +
                                           this->tailLength * (blockIdx - this->formerNum),
                                       this->tailLength);
        }
        shapeoutGlobal.SetGlobalBuffer((__gm__ uint64_t*)shapeout, SHAPEOUT_SIZE);
        offsetGlobal.SetGlobalBuffer((__gm__ uint64_t*)workspace,
                                     numBlocks * 8); // 8 = HEAD_BLOCK_SIZE / sizeof(uint64_t)

        pipe.InitBuffer(inQueueXPing, 1, this->tileLength * sizeof(T));
        pipe.InitBuffer(inQueueXPong, 1, this->tileLength * sizeof(T));
        pipe.InitBuffer(inQueueMaskPing, 1, this->tileLength * sizeof(uint8_t));
        pipe.InitBuffer(inQueueMaskPong, 1, this->tileLength * sizeof(uint8_t));
        pipe.InitBuffer(outQueueYPing, 1, this->tileLength * sizeof(T));
        pipe.InitBuffer(outQueueYPong, 1, this->tileLength * sizeof(T));

        pipe.InitBuffer(offsetSelfBuffer, 1, HEAD_BLOCK_SIZE);
    }

    __aicore__ inline void MoveOutputDataFromWorkspaceToGM(GM_ADDR y)
    {
        pipe.Reset();
        pipe.InitBuffer(
            shapeoutDimBuffer, 1,
            Ops::Base::GetUbBlockSize()); // 32，单位字节。不考虑对齐的应该为 SHAPEOUT_SIZE * sizeof(uint64_t)
        pipe.InitBuffer(offsetOtherBuffer, 1, this->numBlocks * HEAD_BLOCK_SIZE);
        pipe.InitBuffer(moveQue, BUFFER_NUM, this->tileLength * sizeof(T));

        uint64_t ind = 0;
        LocalTensor<uint64_t> offsetUbIn = offsetOtherBuffer.AllocTensor<uint64_t>();
        DataCopyExtParams copyParams1{1, static_cast<uint32_t>(this->numBlocks * HEAD_BLOCK_SIZE), 0, 0, 0};
        DataCopyPadExtParams<uint64_t> padParams1{false, 0, 0, 0};
        DataCopyPad<uint64_t>(offsetUbIn, offsetGlobal, copyParams1, padParams1);
        offsetOtherBuffer.EnQue(offsetUbIn);
        LocalTensor<uint64_t> offsetUbOut = offsetOtherBuffer.DeQue<uint64_t>();

        PipeBarrier<PIPE_ALL>();
        for (int32_t i = 0; i < blockIdx; i++) {
            ind += offsetUbOut.GetValue(i << OFFSET_SHIFT_BITS);
        }
        offsetOtherBuffer.FreeTensor(offsetUbOut);

        yGlobal.SetGlobalBuffer((__gm__ T*)y + ind, this->outOffset);
        // 搬运至GM
        int32_t loopCount = this->outOffset / this->tileLength;
        int32_t tailLoopLength = this->outOffset % this->tileLength;
        // GYW 先处理可以整分的。
        for (int32_t i = 0; i < loopCount; ++i) {
            CopyInMove(i, this->tileLength);
            CopyOutMove(i, this->tileLength);
        }
        // 剩余不能被整分处理
        if (tailLoopLength > 0) {
            CopyInMove(loopCount, tailLoopLength);
            CopyOutMove(loopCount, tailLoopLength);
        }
        if (this->blockIdx == this->numBlocks - 1) {
            LocalTensor<uint64_t> shapeoutDimUb = shapeoutDimBuffer.AllocTensor<uint64_t>();
            shapeoutDimUb.SetValue(0, SHAPEOUT_DIMNUM_WITH_UINT64FLAG);
            shapeoutDimUb.SetValue(1, ind + this->outOffset);
            PipeBarrier<PIPE_ALL>();
            DataCopyExtParams copyParamsShapeout{1, static_cast<uint32_t>(SHAPEOUT_SIZE * sizeof(uint64_t)), 0, 0, 0};
            DataCopyPad<uint64_t>(shapeoutGlobal, shapeoutDimUb, copyParamsShapeout);
            shapeoutDimBuffer.FreeTensor(shapeoutDimUb);
        }
    }

    __aicore__ inline void Process(GM_ADDR y, GM_ADDR shapeout)
    {
        if (blockIdx + 1 > this->numBlocks) {
            return;
        }
        int32_t loopCount = this->tileNum;
        // GYW 先处理可以整分的。
        for (int32_t i = 0; i < loopCount / BUFFER_NUM; i++) {
            CopyIn(BUFFER_NUM * i, inQueueXPing, inQueueMaskPing);
            CopyIn(BUFFER_NUM * i + 1, inQueueXPong, inQueueMaskPong);
            Compute(BUFFER_NUM * i, inQueueXPing, inQueueMaskPing, outQueueYPing, rsvdCntPing);
            Compute(BUFFER_NUM * i + 1, inQueueXPong, inQueueMaskPong, outQueueYPong, rsvdCntPong);
            CopyOut2WorkSpacePingPong();
        }
        if (loopCount % BUFFER_NUM == 1) {
            CopyIn(loopCount - 1, inQueueXPing, inQueueMaskPing);
            Compute(loopCount - 1, inQueueXPing, inQueueMaskPing, outQueueYPing, rsvdCntPing);
            CopyOut2WorkSpace();
        }

        // workspace 写入 offset
        LocalTensor<uint64_t> offsetSelfUb = offsetSelfBuffer.AllocTensor<uint64_t>();
        offsetSelfUb.SetValue(0, this->outOffset);
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(HEAD_BLOCK_SIZE), 0, 0, 0};
        DataCopyPad<uint64_t>(offsetGlobal[blockIdx << OFFSET_SHIFT_BITS], offsetSelfUb[0], copyParams);
        offsetSelfBuffer.EnQue(offsetSelfUb);
        LocalTensor<uint64_t> offsetSelfUbOut = offsetSelfBuffer.DeQue<uint64_t>();
        offsetSelfBuffer.FreeTensor(offsetSelfUbOut);

        SyncAll();

        MoveOutputDataFromWorkspaceToGM(y);
    }

private:
    __aicore__ inline void CopyInMove(int32_t progress, int32_t length)
    {
        LocalTensor<T> xLocal = moveQue.AllocTensor<T>();

        DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(xLocal, workGlobal[progress * (this->tileLength)], copyParams, padParams);

        moveQue.EnQue(xLocal);
    }

    __aicore__ inline void CopyOutMove(int32_t progress, int32_t length)
    {
        LocalTensor<T> yLocal = moveQue.DeQue<T>();

        DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(T)), 0, 0, 0};
        DataCopyPad(yGlobal[progress * (this->tileLength)], yLocal, copyParams);

        moveQue.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyIn(int32_t progress, TQue<QuePosition::VECIN, 1>& inQueueXTemp,
                                  TQue<QuePosition::VECIN, 1>& inQueueMaskTemp)
    {
        LocalTensor<T> xLocal = inQueueXTemp.AllocTensor<T>();
        LocalTensor<uint8_t> maskLocal = inQueueMaskTemp.AllocTensor<uint8_t>();
        uint32_t ind = progress * this->tileLength;
        uint32_t length = this->tileLength;
        if (progress == this->tileNum - 1) {
            // 最后一个block，最后一个tile
            length = this->lasttileLength;
            Duplicate(maskLocal[(length - 1) / DUPLICATE_ALIGN * DUPLICATE_ALIGN], static_cast<uint8_t>(0),
                      static_cast<int32_t>(DUPLICATE_ALIGN));
            PipeBarrier<PIPE_ALL>();
        }

        DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(xLocal, xGlobal[ind], copyParams, padParams);

        {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(length), 0, 0, 0};
            DataCopyPadExtParams<uint8_t> padParams{true, 0, 0, 0};
            DataCopyPad(maskLocal, maskGlobal[ind], copyParams, padParams);
        }

        inQueueXTemp.EnQue(xLocal);
        inQueueMaskTemp.EnQue(maskLocal);
    }

    __aicore__ inline uint16_t GetVLSize() { return Ops::Base::GetVRegSize() / sizeof(T); };

    template <typename T1>
    __aicore__ inline void DoMaskedSelectV3VF(__local_mem__ void* dstLocal, __local_mem__ void* srcLocal,
                                              __local_mem__ void* mask, const uint32_t& processLength, uint64_t& count)
    {
        uint16_t vregLength = GetVLSize();
        uint16_t copyInInputVregLength = vregLength;
        if (IS_8_BYTES_TYPE) {
            copyInInputVregLength *= 2;
        }
        uint32_t maskProcessLength = vregLength;
        uint16_t mainLoopCount = CeilDivision(processLength, vregLength);
        uint8_t compareScalarValue = 1;

        __VEC_SCOPE__
        {
            AscendC::Reg::ClearSpr<AscendC::SpecialPurposeReg::AR>();
            AscendC::Reg::RegTensor<T1> inputVreg, resultVreg;
            AscendC::Reg::UnalignReg ureg0;
            AscendC::Reg::RegTensor<uint8_t> maskVreg;
            AscendC::Reg::MaskReg cmpMaskReg, cmpHelpMaskReg, processLengthMaskReg, compareMaskReg, cmpHelpHighMaskReg;
            compareMaskReg = AscendC::Reg::UpdateMask<uint8_t>(maskProcessLength);
            for (uint16_t i = 0; i < mainLoopCount; i++) {
                // copy mask -> maskVreg, vregLength
                AscendC::Reg::DataCopy(maskVreg, (__local_mem__ uint8_t*&)mask + i * vregLength);
                // compare scalar
                AscendC::Reg::CompareScalar(cmpHelpMaskReg, maskVreg, compareScalarValue, compareMaskReg);

                if constexpr (IS_8_BYTES_TYPE) {
                    AscendC::Reg::MaskUnPack(cmpMaskReg, cmpHelpMaskReg);
                    AscendC::Reg::MaskMov(cmpHelpMaskReg, cmpMaskReg);
                    AscendC::Reg::MaskUnPack(cmpMaskReg, cmpHelpMaskReg);
                    AscendC::Reg::MaskMov(cmpHelpMaskReg, cmpMaskReg);
                    AscendC::Reg::MaskInterleave<T1>(cmpMaskReg, cmpHelpHighMaskReg, cmpHelpMaskReg, cmpHelpMaskReg);
                } else {
                    if constexpr (IS_1_BYTES_TYPE) {
                        AscendC::Reg::MaskMov(cmpMaskReg, cmpHelpMaskReg);
                    }
                    if constexpr (IS_2_BYTES_TYPE) {
                        AscendC::Reg::MaskUnPack(cmpMaskReg, cmpHelpMaskReg);
                    }
                    if constexpr (IS_4_BYTES_TYPE) {
                        AscendC::Reg::MaskUnPack(cmpMaskReg, cmpHelpMaskReg);
                        AscendC::Reg::MaskMov(cmpHelpMaskReg, cmpMaskReg);
                        AscendC::Reg::MaskUnPack(cmpMaskReg, cmpHelpMaskReg);
                    }
                }
                AscendC::Reg::DataCopy(inputVreg, (__local_mem__ T1*&)srcLocal + i * copyInInputVregLength);
                AscendC::Reg::GatherMask<T1, Reg::GatherMaskMode::STORE_REG>(resultVreg, inputVreg, cmpMaskReg);
                AscendC::Reg::DataCopyUnAlign<T1, Reg::PostLiteral::POST_MODE_UPDATE>((__local_mem__ T1*&)dstLocal,
                                                                                      resultVreg, ureg0);
            }
            AscendC::Reg::DataCopyUnAlignPost((__local_mem__ T1*&)dstLocal, ureg0);
        }
        count = (AscendC::Reg::GetSpr<AscendC::SpecialPurposeReg::AR>()) / sizeof(T);
    }

    __aicore__ inline void Compute(int32_t progress, TQue<QuePosition::VECIN, 1>& inQueueXTemp,
                                   TQue<QuePosition::VECIN, 1>& inQueueMaskTemp,
                                   TQue<QuePosition::VECOUT, 1>& outQueueYTemp, uint64_t& rsvdCntTemp)
    {
        LocalTensor<T> xLocal = inQueueXTemp.DeQue<T>();
        LocalTensor<uint8_t> maskLocal = inQueueMaskTemp.DeQue<uint8_t>();
        LocalTensor<T> yLocal = outQueueYTemp.AllocTensor<T>();

        uint32_t length = this->tileLength;
        if (progress == this->tileNum - 1) {
            length = this->lasttileLength;
        }

        if constexpr (IS_8_BYTES_TYPE) {
            DoMaskedSelectV3VF<uint32_t>((__local_mem__ uint32_t*)yLocal.GetPhyAddr(),
                                         (__local_mem__ uint32_t*)xLocal.GetPhyAddr(),
                                         (__local_mem__ uint8_t*)maskLocal.GetPhyAddr(), length, rsvdCntTemp);
        } else {
            DoMaskedSelectV3VF<T>((__local_mem__ T*)yLocal.GetPhyAddr(), (__local_mem__ T*)xLocal.GetPhyAddr(),
                                  (__local_mem__ uint8_t*)maskLocal.GetPhyAddr(), length, rsvdCntTemp);
        }
        outQueueYTemp.EnQue<T>(yLocal);
        inQueueXTemp.FreeTensor(xLocal);
        inQueueMaskTemp.FreeTensor(maskLocal);
    }

    __aicore__ inline void CopyOut2WorkSpace()
    {
        if (rsvdCntPing == 0) {
            return;
        }
        LocalTensor<T> yLocal = outQueueYPing.DeQue<T>();

        DataCopyExtParams copyParams{1, static_cast<uint32_t>(rsvdCntPing * sizeof(T)), 0, 0, 0};
        DataCopyPad(workGlobal[outOffset], yLocal, copyParams);
        outOffset += rsvdCntPing;

        outQueueYPing.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut2WorkSpacePingPong()
    {
        LocalTensor<T> yLocalPing = outQueueYPing.DeQue<T>();
        LocalTensor<T> yLocalPong = outQueueYPong.DeQue<T>();

        if (rsvdCntPing != 0) {
            DataCopyExtParams copyParamsPing{1, static_cast<uint32_t>(rsvdCntPing * sizeof(T)), 0, 0, 0};
            DataCopyPad(workGlobal[outOffset], yLocalPing, copyParamsPing);
            outOffset += rsvdCntPing;
        }
        if (rsvdCntPong != 0) {
            DataCopyExtParams copyParamsPong{1, static_cast<uint32_t>(rsvdCntPong * sizeof(T)), 0, 0, 0};
            DataCopyPad(workGlobal[outOffset], yLocalPong, copyParamsPong);
            outOffset += rsvdCntPong;
        }

        outQueueYPing.FreeTensor(yLocalPing);
        outQueueYPong.FreeTensor(yLocalPong);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueXPing;
    TQue<QuePosition::VECIN, 1> inQueueXPong;
    TQue<QuePosition::VECIN, 1> inQueueMaskPing;
    TQue<QuePosition::VECIN, 1> inQueueMaskPong;
    TQue<QuePosition::VECOUT, 1> outQueueYPing;
    TQue<QuePosition::VECOUT, 1> outQueueYPong;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> moveQue;

    TQue<QuePosition::VECOUT, 1> shapeoutDimBuffer;
    TQue<QuePosition::VECIN, 1> offsetOtherBuffer;
    TQue<QuePosition::VECOUT, 1> offsetSelfBuffer;

    GlobalTensor<T> xGlobal;
    GlobalTensor<T> yGlobal;
    GlobalTensor<uint64_t> shapeoutGlobal;
    GlobalTensor<uint8_t> maskGlobal;
    GlobalTensor<T> workGlobal;
    GlobalTensor<uint64_t> offsetGlobal;

    // 输入
    uint32_t numBlocks;
    uint32_t formerNum;
    uint32_t formerLength;
    uint32_t formertileNum;
    uint32_t formertileLength;
    uint32_t formerlasttileLength;
    uint32_t tailNum;
    uint32_t tailLength;
    uint32_t tailtileNum;
    uint32_t tailtileLength;
    uint32_t taillasttileLength;

    // 本block/核的
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t lasttileLength;

    uint64_t rsvdCntPing = 0;
    uint64_t rsvdCntPong = 0;
    uint64_t outOffset = 0;
    uint32_t blockIdx = 0;
};
} // namespace MaskedSelectV3

#endif // MASKED_SELECT_V3_APT_H
