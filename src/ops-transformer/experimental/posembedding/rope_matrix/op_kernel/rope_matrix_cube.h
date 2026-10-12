/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "../inc/rope_matrix_extern.h"

namespace npu_ops_transformer_ext {
namespace RopeMatrix {

using namespace matmul;

__aicore__ inline uint32_t Ceiling(uint32_t a, uint32_t b)
{
    if (b == 0) {
            return 0;
    }
    return (a + b - 1) / b;
}

template <typename A_T, typename B_T, typename C_T>
class MatmulBatchKernel {
public:
    __aicore__ inline MatmulBatchKernel(){};
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace,
                                const TCubeTiling &tiling, AscendC::TPipe *pipe);
    __aicore__ inline void Process(AscendC::TPipe *pipe, RopeMatrixTiling *ropeTiling);
    
    typedef MatmulType<AscendC::TPosition::GM, CubeFormat::ND, A_T> aType;
    typedef MatmulType<AscendC::TPosition::GM, CubeFormat::ND, B_T, false, LayoutMode::NONE, true> bType;
    typedef MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_T> cType;
    typedef MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_T> biasType;
    Matmul<aType, bType, cType, biasType> matmulObj;

    AscendC::GlobalTensor<A_T> aGlobal;
    AscendC::GlobalTensor<B_T> bGlobal;
    AscendC::GlobalTensor<C_T> cGlobal;
    TCubeTiling tiling;

private:
    __aicore__ inline void CalcGMOffset(int blockIdx, int &offsetA, int &offsetB, int &offsetC,
                                        int &tailM, int &tailN, int B, int H, int M, int N,
                                        int K, int Nindex);
};

template <typename A_T, typename B_T, typename C_T>
__aicore__ inline void MatmulBatchKernel<A_T, B_T, C_T>::Init(GM_ADDR a, GM_ADDR b, GM_ADDR c,
                                                                GM_ADDR workspace, const TCubeTiling &tiling,
                                                                AscendC::TPipe *pipe)
{
    this->tiling = tiling;
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ A_T *>(a), tiling.M * tiling.Ka);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ B_T *>(b), tiling.Ka * tiling.N);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ C_T *>(c), tiling.M * tiling.N);
}

template <typename A_T, typename B_T, typename C_T>
__aicore__ inline void MatmulBatchKernel<A_T, B_T, C_T>::Process(AscendC::TPipe *pipe, RopeMatrixTiling *ropeTiling)
{
    REGIST_MATMUL_OBJ(pipe, GetSysWorkSpacePtr(), matmulObj, &tiling);

    auto B = ropeTiling->b;
    auto H = ropeTiling->n;
    auto M = ropeTiling->s;
    auto N = ropeTiling->d;
    auto K = ropeTiling->d;

    int total_N = B * H;
    for (int Nindex = 0; Nindex < total_N; ++Nindex) {
        int offsetA = 0, offsetB = 0, offsetC = 0;
        int tailM = 0, tailN = 0;
        bool isTransA = false, isTransB = false;

        CalcGMOffset(AscendC::GetBlockIdx(), offsetA, offsetB, offsetC,
                    tailM, tailN, B, H, M, N, K, Nindex);

        matmulObj.SetTensorA(aGlobal[offsetA], isTransA);
        matmulObj.SetTensorB(bGlobal[offsetB], isTransB);
        matmulObj.SetTail(tailM, tailN);
        matmulObj.IterateAll(cGlobal[offsetC]);
    }
    matmulObj.End();
}

template <typename A_T, typename B_T, typename C_T>
__aicore__ inline void MatmulBatchKernel<A_T, B_T, C_T>::CalcGMOffset(int blockIdx,
                                                                        int &offsetA, int &offsetB, int &offsetC,
                                                                        int &tailM, int &tailN, int B, int H,
                                                                        int M, int N, int K, int Nindex)
{
    uint32_t aCoreInaS = tiling.singleCoreM;
    uint32_t aCoreMBlock = Ceiling(M, aCoreInaS);
    uint32_t mCoreIndx = blockIdx % aCoreMBlock;
    uint32_t nCoreIndx = blockIdx / aCoreMBlock;

    offsetA = mCoreIndx * tiling.Ka * aCoreInaS + Nindex * M * tiling.Ka;
    offsetB = nCoreIndx * tiling.singleCoreN;
    offsetC = mCoreIndx * tiling.N * aCoreInaS + nCoreIndx * tiling.singleCoreN + Nindex * M * tiling.N;

    tailM = M - mCoreIndx * aCoreInaS;
    tailM = tailM < aCoreInaS ? tailM : aCoreInaS;

    tailN = N - nCoreIndx * tiling.singleCoreN;
    tailN = tailN < tiling.singleCoreN ? tailN : tiling.singleCoreN;
}

} // namespace RopeMatrix
} //namespace npu_ops_transformer_ext