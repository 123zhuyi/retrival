# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import logging 
import pytest 

import torch 
import torch_npu
import numpy as np

from src.typhoon_mla import typhoon_mla_prepare, typhoon_mla_run

from tests.test_typhoonmla import TyphoonMLA
from tests.test_torchnpu_absorb import TorchNPUPagedMLA
from tests.utils import convert_absorb_to_naive, kv_to_paged, convert_to_dense

torch.set_printoptions(sci_mode=False)
logging.basicConfig(level=logging.INFO) # Specifies the log file path, etc.

if __name__ == "__main__":
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    kv_lora_rank = 512
    v_head_dim = 128
    softmax_scale = 1 / np.sqrt(128)
    block_size = 128

    device = torch.device("npu:0")
    dtype = torch.bfloat16

    for n_heads in [64, 128]:
        for bsz in [128, 256, 512]:
            for shared_seqlen in [4 * 1024, 8 * 1024, 16 * 1024]:
                for nonshared_seqlen in [256, 1024, 4096]:
                    seqlens = [shared_seqlen] + [nonshared_seqlen] * bsz
                    
                    test_typhoon_mla = TyphoonMLA(
                        bsz, seqlens, n_heads, qk_nope_head_dim, qk_rope_head_dim, 
                        kv_lora_rank, v_head_dim, softmax_scale, 
                        device, dtype
                    )
                    
                    typhoonmla_elapsed = test_typhoon_mla.perf(warm_up=25, n_repeat=100)
                    typhoonmla_tgr = bsz / (typhoonmla_elapsed * 1e-3) * 1e-3 # ktoken/s

                    seqlens = [shared_seqlen + nonshared_seqlen] * bsz
                    
                    test_torchnpu_paged_mla = TorchNPUPagedMLA(
                        bsz, seqlens, n_heads, qk_nope_head_dim, qk_rope_head_dim, 
                        kv_lora_rank, v_head_dim, softmax_scale, 
                        device, dtype
                    )
                    torchnpu_elapsed = test_torchnpu_paged_mla.perf(warm_up=25, n_repeat=100)
                    torchnpu_tgr = bsz / (torchnpu_elapsed * 1e-3) * 1e-3 # ktoken/s

                    speedup = typhoonmla_tgr / torchnpu_tgr
                    
                    logging.info(
                        f"batch: {bsz:<5}  "
                        f"shared_seqlen: {shared_seqlen:<5}  "
                        f"nonshared_seqlen: {nonshared_seqlen:<5}  "
                        f"headnum: {n_heads:<5}  |  "
                        f"TyphoonMLA: {typhoonmla_tgr:>6.2f} ktoken/s  "
                        f"TorchNPU-Absorb: {torchnpu_tgr:>6.2f} ktoken/s  "
                        f"Speedup: {speedup:.2f}x"
                    )