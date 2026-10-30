/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
 * KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
 * NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the
 * License.
 */
/**
 * mhc_post AscendC Kernel实现
 * 
 * mHC论文 (DeepSeek): Manifold-Constrained Hyper-Connections
 * 
 * 计算: output[b*num_streams + s] = branch_output[b] * h_post[s]
 * 每个block处理一个(b, s)组合
 */

#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t BLOCK_SIZE = 32;
constexpr int32_t ALIGN_ELEM = 8;  // fp32对齐元素数

class MhcPostKernel {
public:
    __aicore__ inline MhcPostKernel() {}
    
    __aicore__ inline void Init(
        GM_ADDR branch_output,
        GM_ADDR h_post,
        GM_ADDR output,
        int64_t batch,
        int64_t seq_len,
        int64_t dim,
        int64_t num_streams
    ) {
        this->batch = batch;
        this->seq_len = seq_len;
        this->dim = dim;
        this->num_streams = num_streams;
        
        int64_t block_idx = GetBlockIdx();
        this->batch_idx = block_idx / num_streams;
        this->stream_idx = block_idx % num_streams;
        
        this->batch_elements = seq_len * dim;
        
        int64_t input_offset = this->batch_idx * this->batch_elements;
        this->gm_branch_output.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(branch_output) + input_offset,
            this->batch_elements
        );
        
        this->gm_h_post.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(h_post),
            num_streams
        );
        
        int64_t output_offset = (this->batch_idx * num_streams + this->stream_idx) * this->batch_elements;
        this->gm_output.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(output) + output_offset,
            this->batch_elements
        );
        
        this->tile_length = 256;
        this->tile_num = (this->batch_elements + this->tile_length - 1) / this->tile_length;
        
        pipe.InitBuffer(inQueue, BUFFER_NUM, this->tile_length * sizeof(float));
        pipe.InitBuffer(outQueue, BUFFER_NUM, this->tile_length * sizeof(float));
    }
    
    __aicore__ inline void Process() {
        this->weight_value = this->gm_h_post.GetValue(this->stream_idx);
        
        for (int64_t tile_idx = 0; tile_idx < this->tile_num; ++tile_idx) {
            int64_t offset = tile_idx * this->tile_length;
            int64_t length = this->tile_length;
            
            if (offset + length > this->batch_elements) {
                length = this->batch_elements - offset;
            }
            
            CopyIn(offset, length);
            Compute(length);
            CopyOut(offset, length);
        }
    }
    
private:
    __aicore__ inline void CopyIn(int64_t offset, int64_t length) {
        LocalTensor<float> inLocal = inQueue.AllocTensor<float>();
        
        if (length % ALIGN_ELEM == 0) {
            DataCopy(inLocal, gm_branch_output[offset], length);
        } else {
            DataCopyExtParams params{1, static_cast<uint32_t>(length * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
            DataCopyPad(inLocal, gm_branch_output[offset], params, pad);
        }
        
        inQueue.EnQue(inLocal);
    }
    
    __aicore__ inline void Compute(int64_t length) {
        LocalTensor<float> inLocal = inQueue.DeQue<float>();
        LocalTensor<float> outLocal = outQueue.AllocTensor<float>();
        
        int64_t aligned_len = ((length + ALIGN_ELEM - 1) / ALIGN_ELEM) * ALIGN_ELEM;
        Muls(outLocal, inLocal, this->weight_value, aligned_len);
        
        outQueue.EnQue(outLocal);
        inQueue.FreeTensor(inLocal);
    }
    
    __aicore__ inline void CopyOut(int64_t offset, int64_t length) {
        LocalTensor<float> outLocal = outQueue.DeQue<float>();
        
        if (length % ALIGN_ELEM == 0) {
            DataCopy(gm_output[offset], outLocal, length);
        } else {
            DataCopyExtParams params{1, static_cast<uint32_t>(length * sizeof(float)), 0, 0, 0};
            DataCopyPad(gm_output[offset], outLocal, params);
        }
        
        outQueue.FreeTensor(outLocal);
    }
    
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    
    GlobalTensor<float> gm_branch_output;
    GlobalTensor<float> gm_h_post;
    GlobalTensor<float> gm_output;
    
    int64_t batch, seq_len, dim, num_streams;
    int64_t batch_idx, stream_idx;
    int64_t batch_elements, tile_length, tile_num;
    float weight_value;
};

extern "C" __global__ __aicore__ void mhc_post_kernel(
    GM_ADDR branch_output, GM_ADDR h_post, GM_ADDR output,
    int64_t batch, int64_t seq_len, int64_t dim, int64_t num_streams
) {
    MhcPostKernel op;
    op.Init(branch_output, h_post, output, batch, seq_len, dim, num_streams);
    op.Process();
}

extern "C" void mhc_post_do(
    uint32_t blockDim, void* stream,
    uint8_t* branch_output, uint8_t* h_post, uint8_t* output,
    int64_t batch, int64_t seq_len, int64_t dim, int64_t num_streams
) {
    mhc_post_kernel<<<blockDim, nullptr, stream>>>(
        branch_output, h_post, output, batch, seq_len, dim, num_streams
    );
}
