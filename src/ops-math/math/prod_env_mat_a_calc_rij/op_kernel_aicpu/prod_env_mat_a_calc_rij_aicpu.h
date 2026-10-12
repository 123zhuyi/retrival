/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_KERNELS_NORMALIZED_PROD_ENV_MAT_A_CALC_RIJ_H
#define AICPU_KERNELS_NORMALIZED_PROD_ENV_MAT_A_CALC_RIJ_H

#include <cfloat>
#include <cmath>
#include <vector>

#include "cpu_kernel.h"
#include "cpu_kernel_utils.h"
#include "status.h"

namespace aicpu {
class ProdEnvMatACalcRijCpuKernel : public CpuKernel {
public:
    ProdEnvMatACalcRijCpuKernel() = default;
    ~ProdEnvMatACalcRijCpuKernel() override = default;
    uint32_t Compute(CpuKernelContext& ctx) override;

private:
    // Maximum number of neighbors stored per core-region atom inside the mesh input tensor.
    static constexpr int32_t kNeighborMaxNum = 1024;

    // A view over the neighbor-list data packed inside the mesh input tensor.
    struct InputNlist {
        int32_t nlocnum;                        // number of core-region atoms
        int32_t* ilist;                         // core-region atom indices
        int32_t* numneigh;                      // neighbor count of each core-region atom
        int32_t (*firstneigh)[kNeighborMaxNum]; // neighbor indices of each core-region atom
        int32_t* nallmaptable;                  // optional index remap table (present only when mesh carries it)

        InputNlist() : nlocnum(0), ilist(nullptr), numneigh(nullptr), firstneigh(nullptr), nallmaptable(nullptr) {}
    };

    struct NeighborInfo {
        float dist;
        int32_t index;
        NeighborInfo() : dist(0), index(0) {}
        NeighborInfo(float dd, int32_t ii) : dist(dd), index(ii) {}
        bool operator<(const NeighborInfo& b) const
        {
            return ((dist < b.dist) || ((fabs(dist - b.dist) <= FLT_EPSILON) && (index < b.index)));
        }
    };

    // All batch-level pointers and configuration one sample needs, so the per-block/per-atom helpers take a single
    // argument instead of a long parameter list. Output pointers already point at this sample's slice.
    template <typename FPTYPE>
    struct RijBlock {
        FPTYPE* rij;
        int32_t* nlist;
        FPTYPE* distance;
        FPTYPE* rij_x;
        FPTYPE* rij_y;
        FPTYPE* rij_z;
        const FPTYPE* coord;
        const int32_t* type;
        InputNlist meshdata;
        const std::vector<int64_t>* sec_a;
        int32_t nnei;
        int32_t atom_types;
        bool needmap;
        float rcutsquared;
    };

    uint32_t GetInputAndCheck(const CpuKernelContext& ctx) const;
    uint32_t DoCompute(const CpuKernelContext& ctx) const;
    template <typename FPTYPE>
    uint32_t DoProdEnvMatACalcRijCompute(const CpuKernelContext& ctx) const;
    void CumSum(std::vector<int64_t>& sec, const std::vector<int64_t>& n_sel) const;
    template <typename FPTYPE>
    uint32_t ProdEnvMatARijCal(FPTYPE* rij, const CpuKernelContext& ctx, int32_t batch_index, int32_t nnei,
                               const std::vector<int64_t>& sec_a) const;
    template <typename FPTYPE>
    void ComputeRijBlock(const RijBlock<FPTYPE>& blk, int32_t start, int32_t end) const;
    template <typename FPTYPE>
    std::vector<std::vector<NeighborInfo>> CollectSortedNeighbors(const RijBlock<FPTYPE>& blk, int32_t ii,
                                                                  int32_t num_neighbor, FPTYPE i_x, FPTYPE i_y,
                                                                  FPTYPE i_z) const;
    template <typename FPTYPE>
    void WriteAtomResults(const RijBlock<FPTYPE>& blk, int32_t ii, FPTYPE i_x, FPTYPE i_y, FPTYPE i_z,
                          const std::vector<std::vector<NeighborInfo>>& sel) const;
};
} // namespace aicpu
#endif
