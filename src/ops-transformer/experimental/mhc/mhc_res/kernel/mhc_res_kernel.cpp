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
 * mhc_res AscendC Kernel - Stream Mixing
 * Computes: out[b*S + t, seq, d] = Σ_s (h_res[s, t] × x[b*S + s, seq, d])
 * This is an S×S stream mixing operation (residual connection mixing)
 * Supports: fp32, fp16, bf16 (via fp32 compute)
 */

#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t UB_SIZE = 192 * 1024;

template <typename T, int32_t ALIGN>
class MhcResKernel {
public:
    __aicore__ inline MhcResKernel() {}

    __aicore__ inline void Init(
        GM_ADDR input,       // [batch * num_streams, seq_len, dim]
        GM_ADDR h_res,       // [num_streams, num_streams] - h_res[s, t] = weight from stream s to stream t
        GM_ADDR output,      // [batch * num_streams, seq_len, dim]
        int64_t batch, int64_t seq_len, int64_t dim, int64_t num_streams,
        int64_t tile_length, int64_t block_dim
    ) {
        this->gm_input = reinterpret_cast<__gm__ T*>(input);
        this->gm_output = reinterpret_cast<__gm__ T*>(output);
        this->batch = batch;
        this->num_streams = num_streams;
        this->stream_elements = seq_len * dim;
        this->tile_length = tile_length;
        this->block_dim = block_dim;
        this->total_tasks = batch * num_streams;  // Each task: one (batch, target_stream) pair

        gm_h.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(h_res), num_streams * num_streams);
        pipe.InitBuffer(inQue, BUFFER_NUM, tile_length * sizeof(T));
        pipe.InitBuffer(outQue, BUFFER_NUM, tile_length * sizeof(T));
        pipe.InitBuffer(tmpBuf, tile_length * sizeof(T));
    }

    __aicore__ inline void Process() {
        int64_t block_idx = GetBlockIdx();
        // Each block processes multiple (batch, target_stream) combinations
        for (int64_t task_idx = block_idx; task_idx < total_tasks; task_idx += block_dim) {
            int64_t batch_idx = task_idx / num_streams;
            int64_t target_stream = task_idx % num_streams;
            ProcessOne(batch_idx, target_stream);
        }
    }

private:
    __aicore__ inline void ProcessOne(int64_t batch_idx, int64_t target_stream) {
        // Output location: out[batch_idx * num_streams + target_stream, ...]
        int64_t out_off = (batch_idx * num_streams + target_stream) * stream_elements;
        gm_out.SetGlobalBuffer(gm_output + out_off, stream_elements);

        int64_t tiles = (stream_elements + tile_length - 1) / tile_length;
        for (int64_t t = 0; t < tiles; ++t) {
            int64_t tile_off = t * tile_length;
            int64_t len = (tile_off + tile_length > stream_elements)
                        ? (stream_elements - tile_off) : tile_length;
            ProcessOneTile(batch_idx, target_stream, tile_off, len);
        }
    }

    __aicore__ inline void ProcessOneTile(int64_t batch_idx, int64_t target_stream, 
                                          int64_t tile_off, int64_t len) {
        LocalTensor<T> acc = outQue.AllocTensor<T>();
        LocalTensor<T> tmp = tmpBuf.Get<T>();
        uint32_t alen = ((len + ALIGN - 1) / ALIGN) * ALIGN;

        // Sum over all source streams: out[t] = Σ_s h_res[s, t] × in[s]
        for (int64_t src_stream = 0; src_stream < num_streams; ++src_stream) {
            // Input location: in[batch_idx * num_streams + src_stream, ...]
            int64_t in_off = (batch_idx * num_streams + src_stream) * stream_elements + tile_off;
            gm_in.SetGlobalBuffer(gm_input + in_off, len);

            CopyIn(len);
            LocalTensor<T> in = inQue.DeQue<T>();
            
            // h_res[src_stream, target_stream]
            T weight = gm_h.GetValue(src_stream * num_streams + target_stream);

            if (src_stream == 0) {
                Muls(acc, in, weight, alen);
            } else {
                Muls(tmp, in, weight, alen);
                Add(acc, acc, tmp, alen);
            }
            inQue.FreeTensor(in);
        }

        outQue.EnQue(acc);
        CopyOut(tile_off, len);
    }

    __aicore__ inline void CopyIn(int64_t len) {
        LocalTensor<T> local = inQue.AllocTensor<T>();
        uint32_t l = static_cast<uint32_t>(len);
        if (l % ALIGN == 0) {
            DataCopy(local, gm_in[0], l);
        } else {
            DataCopyExtParams p{1, l * (uint32_t)sizeof(T), 0, 0, 0};
            DataCopyPadExtParams<T> pad{false, 0, 0, T(0)};
            DataCopyPad(local, gm_in[0], p, pad);
        }
        inQue.EnQue(local);
    }

    __aicore__ inline void CopyOut(int64_t off, int64_t len) {
        LocalTensor<T> local = outQue.DeQue<T>();
        uint32_t l = static_cast<uint32_t>(len);
        if (l % ALIGN == 0) {
            DataCopy(gm_out[off], local, l);
        } else {
            DataCopyExtParams p{1, l * (uint32_t)sizeof(T), 0, 0, 0};
            DataCopyPad(gm_out[off], local, p);
        }
        outQue.FreeTensor(local);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQue;
    TBuf<QuePosition::VECCALC> tmpBuf;
    GlobalTensor<T> gm_in, gm_h, gm_out;
    __gm__ T* gm_input;
    __gm__ T* gm_output;
    int64_t batch, num_streams, stream_elements, tile_length, block_dim, total_tasks;
};

#if __CCE_AICORE__ == 220
template <>
class MhcResKernel<bfloat16_t, 16> {
public:
    __aicore__ inline MhcResKernel() {}

    __aicore__ inline void Init(
        GM_ADDR input, GM_ADDR h_res, GM_ADDR output,
        int64_t batch, int64_t seq_len, int64_t dim, int64_t num_streams,
        int64_t tile_length, int64_t block_dim
    ) {
        this->gm_input = reinterpret_cast<__gm__ bfloat16_t*>(input);
        this->gm_output = reinterpret_cast<__gm__ bfloat16_t*>(output);
        this->batch = batch;
        this->num_streams = num_streams;
        this->stream_elements = seq_len * dim;
        this->tile_length = tile_length;
        this->block_dim = block_dim;
        this->total_tasks = batch * num_streams;

        gm_h.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(h_res), num_streams * num_streams);
        pipe.InitBuffer(inQue, BUFFER_NUM, tile_length * sizeof(bfloat16_t));
        pipe.InitBuffer(outQue, BUFFER_NUM, tile_length * sizeof(bfloat16_t));
        pipe.InitBuffer(accBuf, tile_length * sizeof(float));
        pipe.InitBuffer(tmpBuf, tile_length * sizeof(float));
    }

    __aicore__ inline void Process() {
        int64_t block_idx = GetBlockIdx();
        for (int64_t task_idx = block_idx; task_idx < total_tasks; task_idx += block_dim) {
            int64_t batch_idx = task_idx / num_streams;
            int64_t target_stream = task_idx % num_streams;
            ProcessOne(batch_idx, target_stream);
        }
    }

private:
    __aicore__ inline void ProcessOne(int64_t batch_idx, int64_t target_stream) {
        int64_t out_off = (batch_idx * num_streams + target_stream) * stream_elements;
        gm_out.SetGlobalBuffer(gm_output + out_off, stream_elements);

        int64_t tiles = (stream_elements + tile_length - 1) / tile_length;
        for (int64_t t = 0; t < tiles; ++t) {
            int64_t tile_off = t * tile_length;
            int64_t len = (tile_off + tile_length > stream_elements)
                        ? (stream_elements - tile_off) : tile_length;
            ProcessOneTile(batch_idx, target_stream, tile_off, len);
        }
    }

    __aicore__ inline void ProcessOneTile(int64_t batch_idx, int64_t target_stream,
                                          int64_t tile_off, int64_t len) {
        LocalTensor<float> acc = accBuf.Get<float>();
        LocalTensor<float> tmp = tmpBuf.Get<float>();
        uint32_t alen = ((len + 16 - 1) / 16) * 16;

        for (int64_t src_stream = 0; src_stream < num_streams; ++src_stream) {
            int64_t in_off = (batch_idx * num_streams + src_stream) * stream_elements + tile_off;
            gm_in.SetGlobalBuffer(gm_input + in_off, len);

            CopyIn(len);
            LocalTensor<bfloat16_t> in = inQue.DeQue<bfloat16_t>();
            float weight = gm_h.GetValue(src_stream * num_streams + target_stream);

            Cast(tmp, in, RoundMode::CAST_NONE, alen);
            if (src_stream == 0) {
                Muls(acc, tmp, weight, alen);
            } else {
                Muls(tmp, tmp, weight, alen);
                Add(acc, acc, tmp, alen);
            }
            inQue.FreeTensor(in);
        }

        LocalTensor<bfloat16_t> out = outQue.AllocTensor<bfloat16_t>();
        Cast(out, acc, RoundMode::CAST_RINT, alen);
        outQue.EnQue(out);
        CopyOut(tile_off, len);
    }

    __aicore__ inline void CopyIn(int64_t len) {
        LocalTensor<bfloat16_t> local = inQue.AllocTensor<bfloat16_t>();
        uint32_t l = static_cast<uint32_t>(len);
        if (l % 16 == 0) {
            DataCopy(local, gm_in[0], l);
        } else {
            DataCopyExtParams p{1, l * 2, 0, 0, 0};
            DataCopyPadExtParams<bfloat16_t> pad{false, 0, 0, bfloat16_t(0)};
            DataCopyPad(local, gm_in[0], p, pad);
        }
        inQue.EnQue(local);
    }

    __aicore__ inline void CopyOut(int64_t off, int64_t len) {
        LocalTensor<bfloat16_t> local = outQue.DeQue<bfloat16_t>();
        uint32_t l = static_cast<uint32_t>(len);
        if (l % 16 == 0) {
            DataCopy(gm_out[off], local, l);
        } else {
            DataCopyExtParams p{1, l * 2, 0, 0, 0};
            DataCopyPad(gm_out[off], local, p);
        }
        outQue.FreeTensor(local);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQue;
    TBuf<QuePosition::VECCALC> accBuf;
    TBuf<QuePosition::VECCALC> tmpBuf;
    GlobalTensor<bfloat16_t> gm_in, gm_out;
    GlobalTensor<float> gm_h;
    __gm__ bfloat16_t* gm_input;
    __gm__ bfloat16_t* gm_output;
    int64_t batch, num_streams, stream_elements, tile_length, block_dim, total_tasks;
};
#endif

inline int64_t CalcTile(int64_t elem, int64_t sz, int64_t align) {
    // Need space for: inQue(2) + outQue(2) + tmpBuf(1) = 5 buffers
    int64_t t = UB_SIZE / ((BUFFER_NUM * 2 + 1) * sz);
    t = (t / align) * align;
    if (t > elem) t = ((elem + align - 1) / align) * align;
    return t < align ? align : t;
}

extern "C" __global__ __aicore__ void mhc_res_kernel_fp32(
    GM_ADDR in, GM_ADDR h, GM_ADDR out,
    int64_t b, int64_t s, int64_t d, int64_t n, int64_t tile, int64_t blkDim
) {
    MhcResKernel<float, 8> op;
    op.Init(in, h, out, b, s, d, n, tile, blkDim);
    op.Process();
}

extern "C" __global__ __aicore__ void mhc_res_kernel_fp16(
    GM_ADDR in, GM_ADDR h, GM_ADDR out,
    int64_t b, int64_t s, int64_t d, int64_t n, int64_t tile, int64_t blkDim
) {
    MhcResKernel<half, 16> op;
    op.Init(in, h, out, b, s, d, n, tile, blkDim);
    op.Process();
}

#if __CCE_AICORE__ == 220
extern "C" __global__ __aicore__ void mhc_res_kernel_bf16(
    GM_ADDR in, GM_ADDR h, GM_ADDR out,
    int64_t b, int64_t s, int64_t d, int64_t n, int64_t tile, int64_t blkDim
) {
    MhcResKernel<bfloat16_t, 16> op;
    op.Init(in, h, out, b, s, d, n, tile, blkDim);
    op.Process();
}
#endif

extern "C" void mhc_res_do_fp32(
    uint32_t blockDim, void* stream, uint8_t* in, uint8_t* h, uint8_t* out,
    int64_t b, int64_t s, int64_t d, int64_t n
) {
    int64_t tile = CalcTile(s * d, sizeof(float), 8);
    uint32_t maxBlk = b * n;  // One task per (batch, target_stream)
    if (blockDim == 0 || blockDim > maxBlk) blockDim = maxBlk;
    mhc_res_kernel_fp32<<<blockDim, nullptr, stream>>>(in, h, out, b, s, d, n, tile, blockDim);
}

extern "C" void mhc_res_do_fp16(
    uint32_t blockDim, void* stream, uint8_t* in, uint8_t* h, uint8_t* out,
    int64_t b, int64_t s, int64_t d, int64_t n
) {
    int64_t tile = CalcTile(s * d, sizeof(half), 16);
    uint32_t maxBlk = b * n;
    if (blockDim == 0 || blockDim > maxBlk) blockDim = maxBlk;
    mhc_res_kernel_fp16<<<blockDim, nullptr, stream>>>(in, h, out, b, s, d, n, tile, blockDim);
}

extern "C" void mhc_res_do_bf16(
    uint32_t blockDim, void* stream, uint8_t* in, uint8_t* h, uint8_t* out,
    int64_t b, int64_t s, int64_t d, int64_t n
) {
    int64_t tile = CalcTile(s * d, 2 + 4, 16);  // bf16 + fp32 accumulator
    uint32_t maxBlk = b * n;
    if (blockDim == 0 || blockDim > maxBlk) blockDim = maxBlk;
    mhc_res_kernel_bf16<<<blockDim, nullptr, stream>>>(in, h, out, b, s, d, n, tile, blockDim);
}

extern "C" void mhc_res_do(
    uint32_t blockDim, void* stream, uint8_t* in, uint8_t* h, uint8_t* out,
    int64_t b, int64_t s, int64_t d, int64_t n
) {
    mhc_res_do_fp32(blockDim, stream, in, h, out, b, s, d, n);
}
