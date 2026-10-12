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
 * mhc_post AscendC Kernel - Adaptive Strategy
 *
 * Two strategies for different shape regimes:
 * - Strategy A (per-stream): parallelize over (batch, stream) pairs.
 *   Each task reads the full input row. Best when elem is small/medium.
 * - Strategy B (read-once): parallelize over (batch, tile) pairs.
 *   Each task reads input once, writes N outputs. Best when elem is large.
 *
 * Host function selects strategy based on batch_elements size.
 */

#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t UB_SIZE = 192 * 1024;
constexpr int32_t MAX_STREAMS = 8;

// ============================================================
// Strategy A: Per-stream parallelism
// total_tasks = batch * num_streams
// Each task: read input[b], multiply by weight[n], write output[b*N+n]
// ============================================================
template <typename T, int32_t ALIGN>
class MhcPostPerStream {
public:
    __aicore__ inline MhcPostPerStream() {}

    __aicore__ inline void Init(
        GM_ADDR branch_output, GM_ADDR h_post, GM_ADDR output,
        int64_t batch, int64_t seq_len, int64_t dim, int64_t num_streams,
        int64_t tile_length, int64_t block_dim
    ) {
        gm_input = reinterpret_cast<__gm__ T*>(branch_output);
        gm_output = reinterpret_cast<__gm__ T*>(output);
        gm_weight.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(h_post), num_streams);
        this->batch = batch;
        this->num_streams = num_streams;
        this->batch_elements = seq_len * dim;
        this->tile_length = tile_length;
        this->block_dim = block_dim;
        this->total_tasks = batch * num_streams;
        pipe.InitBuffer(inQue, BUFFER_NUM, tile_length * sizeof(T));
        pipe.InitBuffer(outQue, BUFFER_NUM, tile_length * sizeof(T));
    }

    __aicore__ inline void Process() {
        int64_t block_idx = GetBlockIdx();
        for (int64_t task_idx = block_idx; task_idx < total_tasks; task_idx += block_dim) {
            int64_t batch_idx = task_idx / num_streams;
            int64_t stream_idx = task_idx % num_streams;
            T weight = gm_weight.GetValue(stream_idx);
            int64_t in_base = batch_idx * batch_elements;
            int64_t out_base = (batch_idx * num_streams + stream_idx) * batch_elements;
            int64_t tiles = (batch_elements + tile_length - 1) / tile_length;
            for (int64_t i = 0; i < tiles; ++i) {
                int64_t off = i * tile_length;
                int64_t len = (off + tile_length > batch_elements) ? (batch_elements - off) : tile_length;
                ProcessTile(in_base + off, out_base + off, len, weight);
            }
        }
    }

private:
    __aicore__ inline void ProcessTile(int64_t in_addr, int64_t out_addr, int64_t len, T weight) {
        LocalTensor<T> local_in = inQue.AllocTensor<T>();
        uint32_t l = static_cast<uint32_t>(len);
        uint32_t aligned = (l + ALIGN - 1) / ALIGN * ALIGN;
        GlobalTensor<T> gm_in;
        gm_in.SetGlobalBuffer(gm_input + in_addr, len);
        if (l % ALIGN == 0) DataCopy(local_in, gm_in, l);
        else {
            DataCopyExtParams p{1, l * (uint32_t)sizeof(T), 0, 0, 0};
            DataCopyPadExtParams<T> pad{false, 0, 0, T(0)};
            DataCopyPad(local_in, gm_in, p, pad);
        }
        inQue.EnQue(local_in);

        LocalTensor<T> in = inQue.DeQue<T>();
        LocalTensor<T> out = outQue.AllocTensor<T>();
        Muls(out, in, weight, aligned);
        outQue.EnQue(out);
        inQue.FreeTensor(in);

        LocalTensor<T> local_out = outQue.DeQue<T>();
        GlobalTensor<T> gm_out;
        gm_out.SetGlobalBuffer(gm_output + out_addr, len);
        if (l % ALIGN == 0) DataCopy(gm_out, local_out, l);
        else {
            DataCopyExtParams p{1, l * (uint32_t)sizeof(T), 0, 0, 0};
            DataCopyPad(gm_out, local_out, p);
        }
        outQue.FreeTensor(local_out);
    }

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQue;
    GlobalTensor<T> gm_weight;
    __gm__ T* gm_input;
    __gm__ T* gm_output;
    int64_t batch, num_streams, batch_elements, tile_length, block_dim, total_tasks;
};

// ============================================================
// Strategy B: Read-once parallelism
// total_tasks = batch * tiles_per_batch
// Each task: read one tile of input[b], write N tiles to output[b*N+0..N-1]
// ============================================================
template <typename T, int32_t ALIGN>
class MhcPostReadOnce {
public:
    __aicore__ inline MhcPostReadOnce() {}

    __aicore__ inline void Init(
        GM_ADDR branch_output, GM_ADDR h_post, GM_ADDR output,
        int64_t batch, int64_t seq_len, int64_t dim, int64_t num_streams,
        int64_t tile_length, int64_t block_dim
    ) {
        gm_input = reinterpret_cast<__gm__ T*>(branch_output);
        gm_output = reinterpret_cast<__gm__ T*>(output);
        this->batch = batch;
        this->num_streams = num_streams;
        this->batch_elements = seq_len * dim;
        this->tile_length = tile_length;
        this->block_dim = block_dim;
        tiles_per_batch = (batch_elements + tile_length - 1) / tile_length;
        total_tasks = batch * tiles_per_batch;

        GlobalTensor<T> gm_h;
        gm_h.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(h_post), num_streams);
        for (int64_t s = 0; s < num_streams && s < MAX_STREAMS; ++s)
            weights[s] = gm_h.GetValue(s);

        pipe.InitBuffer(inQue, BUFFER_NUM, tile_length * sizeof(T));
        pipe.InitBuffer(outQue, BUFFER_NUM, tile_length * sizeof(T));
    }

    __aicore__ inline void Process() {
        int64_t block_idx = GetBlockIdx();
        for (int64_t task = block_idx; task < total_tasks; task += block_dim) {
            int64_t batch_idx = task / tiles_per_batch;
            int64_t tile_idx = task % tiles_per_batch;
            ProcessTileAllStreams(batch_idx, tile_idx);
        }
    }

private:
    __aicore__ inline void ProcessTileAllStreams(int64_t batch_idx, int64_t tile_idx) {
        int64_t off = tile_idx * tile_length;
        int64_t len = (off + tile_length > batch_elements) ? (batch_elements - off) : tile_length;
        int64_t in_gm = batch_idx * batch_elements + off;
        uint32_t l = static_cast<uint32_t>(len);
        uint32_t aligned = (l + ALIGN - 1) / ALIGN * ALIGN;

        // Read input ONCE
        LocalTensor<T> local_in = inQue.AllocTensor<T>();
        GlobalTensor<T> gm_in;
        gm_in.SetGlobalBuffer(gm_input + in_gm, len);
        if (l % ALIGN == 0) DataCopy(local_in, gm_in, l);
        else {
            DataCopyExtParams p{1, l * (uint32_t)sizeof(T), 0, 0, 0};
            DataCopyPadExtParams<T> pad{false, 0, 0, T(0)};
            DataCopyPad(local_in, gm_in, p, pad);
        }
        inQue.EnQue(local_in);
        LocalTensor<T> in = inQue.DeQue<T>();

        // Write N times (reusing input in UB)
        for (int64_t s = 0; s < num_streams; ++s) {
            int64_t out_gm = (batch_idx * num_streams + s) * batch_elements + off;
            LocalTensor<T> out = outQue.AllocTensor<T>();
            Muls(out, in, weights[s], aligned);
            outQue.EnQue(out);
            LocalTensor<T> local_out = outQue.DeQue<T>();
            GlobalTensor<T> gm_out;
            gm_out.SetGlobalBuffer(gm_output + out_gm, len);
            if (l % ALIGN == 0) DataCopy(gm_out, local_out, l);
            else {
                DataCopyExtParams p{1, l * (uint32_t)sizeof(T), 0, 0, 0};
                DataCopyPad(gm_out, local_out, p);
            }
            outQue.FreeTensor(local_out);
        }
        inQue.FreeTensor(in);
    }

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQue;
    T weights[MAX_STREAMS];
    __gm__ T* gm_input;
    __gm__ T* gm_output;
    int64_t batch, num_streams, batch_elements, tile_length, block_dim;
    int64_t tiles_per_batch, total_tasks;
};

// ============================================================
// BF16: Strategy A
// ============================================================
#if __CCE_AICORE__ == 220
class MhcPostPerStreamBF16 {
public:
    __aicore__ inline MhcPostPerStreamBF16() {}

    __aicore__ inline void Init(
        GM_ADDR branch_output, GM_ADDR h_post_fp32, GM_ADDR output,
        int64_t batch, int64_t seq_len, int64_t dim, int64_t num_streams,
        int64_t tile_length, int64_t block_dim
    ) {
        gm_input = reinterpret_cast<__gm__ bfloat16_t*>(branch_output);
        gm_output = reinterpret_cast<__gm__ bfloat16_t*>(output);
        gm_weight.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(h_post_fp32), num_streams);
        this->batch = batch;
        this->num_streams = num_streams;
        this->batch_elements = seq_len * dim;
        this->tile_length = tile_length;
        this->block_dim = block_dim;
        this->total_tasks = batch * num_streams;
        pipe.InitBuffer(inQue, BUFFER_NUM, tile_length * sizeof(bfloat16_t));
        pipe.InitBuffer(outQue, BUFFER_NUM, tile_length * sizeof(bfloat16_t));
        pipe.InitBuffer(tmpBuf, tile_length * sizeof(float));
    }

    __aicore__ inline void Process() {
        int64_t block_idx = GetBlockIdx();
        for (int64_t task_idx = block_idx; task_idx < total_tasks; task_idx += block_dim) {
            int64_t batch_idx = task_idx / num_streams;
            int64_t stream_idx = task_idx % num_streams;
            float weight = gm_weight.GetValue(stream_idx);
            int64_t in_base = batch_idx * batch_elements;
            int64_t out_base = (batch_idx * num_streams + stream_idx) * batch_elements;
            int64_t tiles = (batch_elements + tile_length - 1) / tile_length;
            for (int64_t i = 0; i < tiles; ++i) {
                int64_t off = i * tile_length;
                int64_t len = (off + tile_length > batch_elements) ? (batch_elements - off) : tile_length;
                ProcessTile(in_base + off, out_base + off, len, weight);
            }
        }
    }

private:
    __aicore__ inline void ProcessTile(int64_t in_addr, int64_t out_addr, int64_t len, float weight) {
        LocalTensor<bfloat16_t> local_in = inQue.AllocTensor<bfloat16_t>();
        uint32_t l = static_cast<uint32_t>(len);
        uint32_t aligned = (l + 15) / 16 * 16;
        GlobalTensor<bfloat16_t> gm_in;
        gm_in.SetGlobalBuffer(gm_input + in_addr, len);
        if (l % 16 == 0) DataCopy(local_in, gm_in, l);
        else {
            DataCopyExtParams p{1, l * 2, 0, 0, 0};
            DataCopyPadExtParams<bfloat16_t> pad{false, 0, 0, bfloat16_t(0)};
            DataCopyPad(local_in, gm_in, p, pad);
        }
        inQue.EnQue(local_in);

        LocalTensor<bfloat16_t> in = inQue.DeQue<bfloat16_t>();
        LocalTensor<float> tmp = tmpBuf.Get<float>();
        LocalTensor<bfloat16_t> out = outQue.AllocTensor<bfloat16_t>();
        Cast(tmp, in, RoundMode::CAST_NONE, aligned);
        Muls(tmp, tmp, weight, aligned);
        Cast(out, tmp, RoundMode::CAST_RINT, aligned);
        outQue.EnQue(out);
        inQue.FreeTensor(in);

        LocalTensor<bfloat16_t> local_out = outQue.DeQue<bfloat16_t>();
        GlobalTensor<bfloat16_t> gm_out;
        gm_out.SetGlobalBuffer(gm_output + out_addr, len);
        if (l % 16 == 0) DataCopy(gm_out, local_out, l);
        else {
            DataCopyExtParams p{1, l * 2, 0, 0, 0};
            DataCopyPad(gm_out, local_out, p);
        }
        outQue.FreeTensor(local_out);
    }

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQue;
    TBuf<QuePosition::VECCALC> tmpBuf;
    GlobalTensor<float> gm_weight;
    __gm__ bfloat16_t* gm_input;
    __gm__ bfloat16_t* gm_output;
    int64_t batch, num_streams, batch_elements, tile_length, block_dim, total_tasks;
};

// ============================================================
// BF16: Strategy B
// ============================================================
class MhcPostReadOnceBF16 {
public:
    __aicore__ inline MhcPostReadOnceBF16() {}

    __aicore__ inline void Init(
        GM_ADDR branch_output, GM_ADDR h_post_fp32, GM_ADDR output,
        int64_t batch, int64_t seq_len, int64_t dim, int64_t num_streams,
        int64_t tile_length, int64_t block_dim
    ) {
        gm_input = reinterpret_cast<__gm__ bfloat16_t*>(branch_output);
        gm_output = reinterpret_cast<__gm__ bfloat16_t*>(output);
        this->batch = batch;
        this->num_streams = num_streams;
        this->batch_elements = seq_len * dim;
        this->tile_length = tile_length;
        this->block_dim = block_dim;
        tiles_per_batch = (batch_elements + tile_length - 1) / tile_length;
        total_tasks = batch * tiles_per_batch;

        GlobalTensor<float> gm_h;
        gm_h.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(h_post_fp32), num_streams);
        for (int64_t s = 0; s < num_streams && s < MAX_STREAMS; ++s)
            weights[s] = gm_h.GetValue(s);

        pipe.InitBuffer(inQue, BUFFER_NUM, tile_length * sizeof(bfloat16_t));
        pipe.InitBuffer(outQue, BUFFER_NUM, tile_length * sizeof(bfloat16_t));
        pipe.InitBuffer(tmpBuf, tile_length * sizeof(float));
    }

    __aicore__ inline void Process() {
        int64_t block_idx = GetBlockIdx();
        for (int64_t task = block_idx; task < total_tasks; task += block_dim) {
            int64_t batch_idx = task / tiles_per_batch;
            int64_t tile_idx = task % tiles_per_batch;
            ProcessTileAllStreams(batch_idx, tile_idx);
        }
    }

private:
    __aicore__ inline void ProcessTileAllStreams(int64_t batch_idx, int64_t tile_idx) {
        int64_t off = tile_idx * tile_length;
        int64_t len = (off + tile_length > batch_elements) ? (batch_elements - off) : tile_length;
        int64_t in_gm = batch_idx * batch_elements + off;
        uint32_t l = static_cast<uint32_t>(len);
        uint32_t aligned = (l + 15) / 16 * 16;

        LocalTensor<bfloat16_t> local_in = inQue.AllocTensor<bfloat16_t>();
        GlobalTensor<bfloat16_t> gm_in;
        gm_in.SetGlobalBuffer(gm_input + in_gm, len);
        if (l % 16 == 0) DataCopy(local_in, gm_in, l);
        else {
            DataCopyExtParams p{1, l * 2, 0, 0, 0};
            DataCopyPadExtParams<bfloat16_t> pad{false, 0, 0, bfloat16_t(0)};
            DataCopyPad(local_in, gm_in, p, pad);
        }
        inQue.EnQue(local_in);
        LocalTensor<bfloat16_t> in = inQue.DeQue<bfloat16_t>();

        for (int64_t s = 0; s < num_streams; ++s) {
            int64_t out_gm = (batch_idx * num_streams + s) * batch_elements + off;
            LocalTensor<float> tmp = tmpBuf.Get<float>();
            LocalTensor<bfloat16_t> out = outQue.AllocTensor<bfloat16_t>();
            Cast(tmp, in, RoundMode::CAST_NONE, aligned);
            Muls(tmp, tmp, weights[s], aligned);
            Cast(out, tmp, RoundMode::CAST_RINT, aligned);
            outQue.EnQue(out);
            LocalTensor<bfloat16_t> local_out = outQue.DeQue<bfloat16_t>();
            GlobalTensor<bfloat16_t> gm_out;
            gm_out.SetGlobalBuffer(gm_output + out_gm, len);
            if (l % 16 == 0) DataCopy(gm_out, local_out, l);
            else {
                DataCopyExtParams p{1, l * 2, 0, 0, 0};
                DataCopyPad(gm_out, local_out, p);
            }
            outQue.FreeTensor(local_out);
        }
        inQue.FreeTensor(in);
    }

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQue;
    TBuf<QuePosition::VECCALC> tmpBuf;
    float weights[MAX_STREAMS];
    __gm__ bfloat16_t* gm_input;
    __gm__ bfloat16_t* gm_output;
    int64_t batch, num_streams, batch_elements, tile_length, block_dim;
    int64_t tiles_per_batch, total_tasks;
};
#endif

// ============================================================
// Tile calculation
// ============================================================
inline int64_t CalcTile(int64_t elem, int64_t sz, int64_t align) {
    int64_t t = UB_SIZE / (BUFFER_NUM * 2 * sz);
    t = (t / align) * align;
    if (t > elem) t = ((elem + align - 1) / align) * align;
    return t < align ? align : t;
}

// Threshold: when batch_elements * sizeof(T) * num_streams exceeds this,
// read-once strategy wins because re-reading input N times is too expensive.
// Empirically ~4MB total read traffic (elem >= 256K for N=4 fp32, i.e. read >= 4MB)
constexpr int64_t READONCE_THRESHOLD_BYTES = 4 * 1024 * 1024;

inline bool UseReadOnce(int64_t batch_elements, int64_t elem_size, int64_t num_streams, int64_t batch) {
    int64_t total_read = batch_elements * elem_size * num_streams;
    return total_read >= READONCE_THRESHOLD_BYTES && batch >= 16;
}

// ============================================================
// Kernel entry points - Strategy A (per-stream)
// ============================================================
extern "C" __global__ __aicore__ void mhc_post_kernel_a_fp32(
    GM_ADDR in, GM_ADDR h, GM_ADDR out,
    int64_t b, int64_t s, int64_t d, int64_t n, int64_t tile, int64_t blkDim
) {
    MhcPostPerStream<float, 8> op;
    op.Init(in, h, out, b, s, d, n, tile, blkDim);
    op.Process();
}

extern "C" __global__ __aicore__ void mhc_post_kernel_a_fp16(
    GM_ADDR in, GM_ADDR h, GM_ADDR out,
    int64_t b, int64_t s, int64_t d, int64_t n, int64_t tile, int64_t blkDim
) {
    MhcPostPerStream<half, 16> op;
    op.Init(in, h, out, b, s, d, n, tile, blkDim);
    op.Process();
}

#if __CCE_AICORE__ == 220
extern "C" __global__ __aicore__ void mhc_post_kernel_a_bf16(
    GM_ADDR in, GM_ADDR h, GM_ADDR out,
    int64_t b, int64_t s, int64_t d, int64_t n, int64_t tile, int64_t blkDim
) {
    MhcPostPerStreamBF16 op;
    op.Init(in, h, out, b, s, d, n, tile, blkDim);
    op.Process();
}
#endif

// ============================================================
// Kernel entry points - Strategy B (read-once)
// ============================================================
extern "C" __global__ __aicore__ void mhc_post_kernel_b_fp32(
    GM_ADDR in, GM_ADDR h, GM_ADDR out,
    int64_t b, int64_t s, int64_t d, int64_t n, int64_t tile, int64_t blkDim
) {
    MhcPostReadOnce<float, 8> op;
    op.Init(in, h, out, b, s, d, n, tile, blkDim);
    op.Process();
}

extern "C" __global__ __aicore__ void mhc_post_kernel_b_fp16(
    GM_ADDR in, GM_ADDR h, GM_ADDR out,
    int64_t b, int64_t s, int64_t d, int64_t n, int64_t tile, int64_t blkDim
) {
    MhcPostReadOnce<half, 16> op;
    op.Init(in, h, out, b, s, d, n, tile, blkDim);
    op.Process();
}

#if __CCE_AICORE__ == 220
extern "C" __global__ __aicore__ void mhc_post_kernel_b_bf16(
    GM_ADDR in, GM_ADDR h, GM_ADDR out,
    int64_t b, int64_t s, int64_t d, int64_t n, int64_t tile, int64_t blkDim
) {
    MhcPostReadOnceBF16 op;
    op.Init(in, h, out, b, s, d, n, tile, blkDim);
    op.Process();
}
#endif

// ============================================================
// Host dispatch: auto-select strategy
// ============================================================
extern "C" void mhc_post_do_fp32(
    uint32_t blockDim, void* stream, uint8_t* in, uint8_t* h, uint8_t* out,
    int64_t b, int64_t s, int64_t d, int64_t n
) {
    int64_t tile = CalcTile(s * d, sizeof(float), 8);
    if (UseReadOnce(s * d, sizeof(float), n, b)) {
        int64_t tiles = (s * d + tile - 1) / tile;
        uint32_t maxBlk = b * tiles;
        if (blockDim == 0 || blockDim > maxBlk) blockDim = (maxBlk > 20) ? 20 : maxBlk;
        mhc_post_kernel_b_fp32<<<blockDim, nullptr, stream>>>(in, h, out, b, s, d, n, tile, blockDim);
    } else {
        uint32_t maxBlk = b * n;
        if (blockDim == 0 || blockDim > maxBlk) blockDim = maxBlk;
        mhc_post_kernel_a_fp32<<<blockDim, nullptr, stream>>>(in, h, out, b, s, d, n, tile, blockDim);
    }
}

extern "C" void mhc_post_do_fp16(
    uint32_t blockDim, void* stream, uint8_t* in, uint8_t* h, uint8_t* out,
    int64_t b, int64_t s, int64_t d, int64_t n
) {
    int64_t tile = CalcTile(s * d, sizeof(half), 16);
    if (UseReadOnce(s * d, sizeof(half), n, b)) {
        int64_t tiles = (s * d + tile - 1) / tile;
        uint32_t maxBlk = b * tiles;
        if (blockDim == 0 || blockDim > maxBlk) blockDim = (maxBlk > 20) ? 20 : maxBlk;
        mhc_post_kernel_b_fp16<<<blockDim, nullptr, stream>>>(in, h, out, b, s, d, n, tile, blockDim);
    } else {
        uint32_t maxBlk = b * n;
        if (blockDim == 0 || blockDim > maxBlk) blockDim = maxBlk;
        mhc_post_kernel_a_fp16<<<blockDim, nullptr, stream>>>(in, h, out, b, s, d, n, tile, blockDim);
    }
}

extern "C" void mhc_post_do_bf16(
    uint32_t blockDim, void* stream, uint8_t* in, uint8_t* h, uint8_t* out,
    int64_t b, int64_t s, int64_t d, int64_t n
) {
    int64_t tile = CalcTile(s * d, 2 + 4, 16);
    if (UseReadOnce(s * d, 2, n, b)) {
        int64_t tiles = (s * d + tile - 1) / tile;
        uint32_t maxBlk = b * tiles;
        if (blockDim == 0 || blockDim > maxBlk) blockDim = (maxBlk > 20) ? 20 : maxBlk;
        mhc_post_kernel_b_bf16<<<blockDim, nullptr, stream>>>(in, h, out, b, s, d, n, tile, blockDim);
    } else {
        uint32_t maxBlk = b * n;
        if (blockDim == 0 || blockDim > maxBlk) blockDim = maxBlk;
        mhc_post_kernel_a_bf16<<<blockDim, nullptr, stream>>>(in, h, out, b, s, d, n, tile, blockDim);
    }
}

extern "C" void mhc_post_do(
    uint32_t blockDim, void* stream, uint8_t* in, uint8_t* h, uint8_t* out,
    int64_t b, int64_t s, int64_t d, int64_t n
) {
    mhc_post_do_fp32(blockDim, stream, in, h, out, b, s, d, n);
}
