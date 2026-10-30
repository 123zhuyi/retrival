/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/DeviceUtils.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace npu_ops_transformer_ext
{

    namespace ScoreNormalize
    {

        #include <iostream>
        #include <stdio.h>
        #include "kernel_operator.h"
        #include "tiling/platform/platform_ascendc.h"
        #include "rotary/moe_routing.h"
        #include "rotary/dtype_convert.h"
        using namespace AscendC;

        constexpr int BUFFER_NUM = 2;
        #define UB_BUFFER_SIZE 194560
        template <typename T>
        class KernelFusedMoeGate
        {
        public:
            __aicore__ inline KernelFusedMoeGate() {}
            __aicore__ inline void Init(GM_ADDR topk_logits_gather, int32_t ROWS, int32_t per_count, int32_t block_cnt, TPipe* pipe)
            {
                BASE_SIZE = 32 / sizeof(T);
                BASE_SIZE_FP32 = 32 / sizeof(float);
                core_id_ = GetBlockIdx();
                TILE_LENGTH = per_count;

                per_dim_ = per_count;
                block_cnt_ = block_cnt;

                int32_t rows_per_core = int(ROWS / block_cnt_);
                rows_per_core_per_loop_ = 1;
                rows_per_core = min(rows_per_core, 254);
                align_logit_cnt = DivCeil(per_dim_, BASE_SIZE) * BASE_SIZE;
                if (rows_per_core >= 2)
                {
                    int32_t limit_rows_per_core = UB_BUFFER_SIZE / (align_logit_cnt * BUFFER_NUM * 8);
                    rows_per_core_per_loop_ = min(rows_per_core, limit_rows_per_core);
                }
                tail_rows_last_loop_per_core_ = rows_per_core_per_loop_;

                align_reduce_cnt = DivCeil(rows_per_core_per_loop_, BASE_SIZE_FP32) * BASE_SIZE_FP32;

                int32_t loop_count = ROWS / (rows_per_core_per_loop_ * block_cnt_);

                full_loop_count_ = loop_count;
                full_rows_offset_ = full_loop_count_ * rows_per_core_per_loop_ * block_cnt_ * TILE_LENGTH;
                rows_offset_ = 0;

                int tail_count = ROWS - (loop_count * block_cnt_ * rows_per_core_per_loop_);

                tail_rows_last_loop_per_core_ = tail_count / block_cnt_;
                tail_count = tail_count % block_cnt_;
                tail_rows_offset_ = tail_count * (tail_rows_last_loop_per_core_ + 1) * TILE_LENGTH;
                if (core_id_ < tail_count || tail_rows_last_loop_per_core_)
                {
                    tail_rows_last_loop_per_core_ += 1;
                    loop_count += 1;
                }
                loop_count_ = loop_count;
                tail_count_ = tail_count;

                topk_logits_gather_Gm.SetGlobalBuffer((__gm__ T *)topk_logits_gather, rows_per_core_per_loop_ * align_logit_cnt);

                pipe -> InitBuffer(inQueue_topk_logits_gather, BUFFER_NUM, rows_per_core_per_loop_ * align_logit_cnt * sizeof(T));
                pipe -> InitBuffer(outQueue_topk_logits, BUFFER_NUM, rows_per_core_per_loop_ * align_logit_cnt * sizeof(T));
                pipe -> InitBuffer(work_buf, rows_per_core_per_loop_ * align_logit_cnt * sizeof(float));
                pipe -> InitBuffer(work_buf1, align_reduce_cnt * sizeof(float));
                pipe -> InitBuffer(work_buf3, rows_per_core_per_loop_ * align_logit_cnt * sizeof(float));
            }

            __aicore__ inline void Process()
            {
                for (int i = 0; i < loop_count_; ++i)
                {
                    CopyIn(i);
                    Compute(i);
                    CopyOut(i);
                }
            }

        private:
            __aicore__ inline void CopyIn(int32_t loop_cnt)
            {
                global_core_index_ = loop_cnt * block_cnt_ + core_id_;
                if (loop_cnt < full_loop_count_)
                {
                    rows_offset_ = global_core_index_ * rows_per_core_per_loop_ * TILE_LENGTH;
                }
                else
                {
                    rows_per_core_per_loop_ = tail_rows_last_loop_per_core_;

                    if (core_id_ < tail_count_)
                    {
                        rows_offset_ = full_rows_offset_ + core_id_ * rows_per_core_per_loop_ * TILE_LENGTH;
                    }
                    else
                    {
                        rows_offset_ = full_rows_offset_ + tail_rows_offset_ + (core_id_ - tail_count_) * rows_per_core_per_loop_ * TILE_LENGTH;
                    }
                }

                LocalTensor<T> topk_logits_gatherLocal = inQueue_topk_logits_gather.AllocTensor<T>();
                DataCopyParams copy_pas{(uint16_t)rows_per_core_per_loop_, (uint16_t)(TILE_LENGTH * sizeof(T)), 0, 0};
                DataCopyPadParams pad_pas{true, 0, uint8_t(align_logit_cnt - TILE_LENGTH), 0};
                DataCopyPad(topk_logits_gatherLocal, topk_logits_gather_Gm[rows_offset_], copy_pas, pad_pas);
                inQueue_topk_logits_gather.EnQue(topk_logits_gatherLocal);
            }

            __aicore__ inline void Compute(int32_t loop_cnt)
            {
                LocalTensor<T> topk_logits_gatherLocal = inQueue_topk_logits_gather.DeQue<T>();
                LocalTensor<T> topk_logitsLocal = outQueue_topk_logits.AllocTensor<T>();
                LocalTensor<float> topk_logits_gatherfp32Local = work_buf.Get<float>();
                LocalTensor<float> reducefp32Local = work_buf1.Get<float>();
                LocalTensor<float> broadcast_reducefp32Local = work_buf3.Get<float>();
                Cast(topk_logits_gatherfp32Local, topk_logits_gatherLocal, RoundMode::CAST_NONE, rows_per_core_per_loop_ * align_logit_cnt);

                WholeReduceSum(reducefp32Local, topk_logits_gatherfp32Local, TILE_LENGTH, rows_per_core_per_loop_, 1, 1, 2);

                uint32_t srcshape[2] = {(uint32_t)rows_per_core_per_loop_, 1};
                uint32_t dstshape[2] = {(uint32_t)rows_per_core_per_loop_, (uint32_t)align_logit_cnt};
                Broadcast<float, 2, 1>(broadcast_reducefp32Local, reducefp32Local, dstshape, srcshape);
                Div(topk_logits_gatherfp32Local, topk_logits_gatherfp32Local, broadcast_reducefp32Local, rows_per_core_per_loop_ * align_logit_cnt);

                Muls(topk_logits_gatherfp32Local, topk_logits_gatherfp32Local, (float)2.5, rows_per_core_per_loop_ * align_logit_cnt);
                Cast(topk_logitsLocal, topk_logits_gatherfp32Local, RoundMode::CAST_ROUND, rows_per_core_per_loop_ * align_logit_cnt);

                outQueue_topk_logits.EnQue<T>(topk_logitsLocal);
                work_buf.FreeTensor(topk_logits_gatherfp32Local);
                work_buf1.FreeTensor(reducefp32Local);
                work_buf3.FreeTensor(broadcast_reducefp32Local);
                inQueue_topk_logits_gather.FreeTensor(topk_logits_gatherLocal);
            }

            __aicore__ inline void CopyOut(int32_t loop_cnt)
            {
                LocalTensor<T> topk_logitsLocal = outQueue_topk_logits.DeQue<T>();
                DataCopyParams copyParams{(uint16_t)rows_per_core_per_loop_, (uint16_t)(TILE_LENGTH * sizeof(T)), 0, 0};
                DataCopyPad(topk_logits_gather_Gm[rows_offset_], topk_logitsLocal, copyParams);

                outQueue_topk_logits.FreeTensor(topk_logitsLocal);
            }

        private:
            TQue<QuePosition::VECIN, BUFFER_NUM> inQueue_topk_logits_gather;
            TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue_topk_logits;
            TBuf<QuePosition::VECCALC> work_buf, work_buf1, work_buf2, work_buf3;
            GlobalTensor<T> topk_logits_gather_Gm;
            int32_t core_id_;
            int32_t TILE_LENGTH;
            int32_t per_dim_;
            int32_t BASE_SIZE, BASE_SIZE_FP32, block_cnt_;
            int32_t align_logit_cnt, align_reduce_cnt;
            int32_t tail_count_;
            int32_t loop_count_, full_loop_count_, full_rows_offset_, dim_loop_count_, rows_per_core_per_loop_, tail_rows_last_loop_per_core_;
            int32_t global_core_index_, core_rows_, rows_offset_, tail_rows_offset_;
        };

        extern "C" __global__ __aicore__ void compute_score_normalize(GM_ADDR topk_logits_gather, int32_t ROWS, int32_t per_count, int32_t block_cnt, int32_t dtype)
        {
            TPipe pipe;
            TYPE_SWITCH(dtype, T, {
                KernelFusedMoeGate<T> op;
                op.Init(topk_logits_gather, ROWS, per_count, block_cnt, &pipe);
                op.Process();
            });
        }

        void score_normalize_launch(uint8_t *topk_logits_gather, int32_t ROWS, int32_t per_count, int32_t blockDim, void *stream, int32_t dtype)
        {
            int32_t real_blockDim = ROWS < blockDim ? ROWS : blockDim;
            compute_score_normalize<<<real_blockDim, nullptr, stream>>>(topk_logits_gather, ROWS, per_count, real_blockDim, dtype);
        }

        void score_normalize_npu(torch::Tensor &topk_logits_gather, int64_t ROWS, int64_t per_count)
        { // 具体参数符合原型定义
            TORCH_CHECK(torch_npu::utils::is_npu(topk_logits_gather), "Input tensor must be on NPU device");
            TORCH_CHECK(topk_logits_gather.scalar_type() == at::kBFloat16, "Only BFloat16 type is supported by score_normalize now");
            TORCH_CHECK(per_count <= 16, "The prameter \"k\" must <= 16 now");

            auto topk_logits_gather_ptr = topk_logits_gather.data_ptr();
            c10::ScalarType scalar_type = topk_logits_gather.scalar_type();
            int32_t dtype = 27; // corresponds to bf16 in AclDataType
            int32_t blockDim = 40;
            auto stream = c10_npu::getCurrentNPUStream().stream(false);
            auto acl_call = [=]() -> int
            {
                score_normalize_launch((uint8_t *)topk_logits_gather_ptr, ROWS, per_count, blockDim, stream, dtype);
                return 0;
            };
            at_npu::native::OpCommand::RunOpApi("FusedGate", acl_call);
        }

        TORCH_LIBRARY_IMPL(npu_ops_transformer_ext, PrivateUse1, m)
        {
            m.impl("score_normalize", score_normalize_npu);
        }

    } // ScoreNormalize
} // npu_ops_transformer_ext