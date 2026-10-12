/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "prod_env_mat_a_calc_rij_aicpu.h"

#include <algorithm>
#include <vector>

#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "log.h"
#include "securec.h"
#include "status.h"
#include "utils/kernel_util.h"

namespace {
const char* const kProdEnvMatACalcRij = "ProdEnvMatACalcRij";
constexpr uint32_t kInputNum = 5;
constexpr uint32_t kOutputNum = 6;
constexpr uint32_t kCoordInputIdx = 0;
constexpr uint32_t kTypeInputIdx = 1;
constexpr uint32_t kNatomsInputIdx = 2;
constexpr uint32_t kMeshInputIdx = 4;
constexpr uint32_t kRijOutputIdx = 0;
constexpr uint32_t kNlistOutputIdx = 1;
constexpr uint32_t kDistanceOutputIdx = 2;
constexpr uint32_t kRijXOutputIdx = 3;
constexpr uint32_t kRijYOutputIdx = 4;
constexpr uint32_t kRijZOutputIdx = 5;
constexpr int32_t kCoordDimNum = 2;
constexpr int32_t kTypeDimNum = 2;
constexpr int32_t kNatomsDimNum = 1;
constexpr int32_t kMeshDimNum = 1;
constexpr int32_t kCoordinateXyzNum = 3;
constexpr int32_t kMinNatomsNum = 3;
constexpr int32_t kMeshHeaderNum = 1;           // leading slot storing nlocnum
constexpr int32_t kMeshSectionsBeforeNeigh = 2; // ilist + numneigh sections, each of length nlocnum
// Per core-region atom the mesh tensor reserves 1026 = 2 + kNeighborMaxNum int32 slots when it also carries the
// nallmaptable, so this stride is used to locate that optional table.
constexpr uint32_t kMeshDataLength = 1026;
} // namespace

namespace aicpu {
uint32_t ProdEnvMatACalcRijCpuKernel::Compute(CpuKernelContext& ctx)
{
    KERNEL_LOG_INFO("ProdEnvMatACalcRijCpuKernel::Compute start");
    KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "Check ProdEnvMatACalcRij params failed.");
    uint32_t ret = GetInputAndCheck(ctx);
    if (ret != KERNEL_STATUS_OK) {
        KERNEL_LOG_ERROR("[%s] GetInputAndCheck failed, KernelStatus is [%u].", kProdEnvMatACalcRij, ret);
        return ret;
    }
    uint32_t res = DoCompute(ctx);
    if (res != KERNEL_STATUS_OK) {
        KERNEL_LOG_ERROR("[%s] kernel compute failed, KernelStatus is [%u].", kProdEnvMatACalcRij, res);
        return res;
    }
    KERNEL_LOG_INFO("ProdEnvMatACalcRijCpuKernel::Compute end");
    return KERNEL_STATUS_OK;
}

uint32_t ProdEnvMatACalcRijCpuKernel::DoCompute(const CpuKernelContext& ctx) const
{
    DataType params_type = ctx.Input(kCoordInputIdx)->GetDataType();
    if (params_type == DT_FLOAT) {
        return DoProdEnvMatACalcRijCompute<float>(ctx);
    }
    return DoProdEnvMatACalcRijCompute<double>(ctx);
}

uint32_t ProdEnvMatACalcRijCpuKernel::GetInputAndCheck(const CpuKernelContext& ctx) const
{
    Tensor* coord_tensor = ctx.Input(kCoordInputIdx);
    Tensor* type_tensor = ctx.Input(kTypeInputIdx);
    Tensor* natoms_tensor = ctx.Input(kNatomsInputIdx);
    Tensor* mesh_tensor = ctx.Input(kMeshInputIdx);

    // check input dims
    int32_t coord_dims = coord_tensor->GetTensorShape()->GetDims();
    int32_t type_dims = type_tensor->GetTensorShape()->GetDims();
    int32_t natoms_dims = natoms_tensor->GetTensorShape()->GetDims();
    int32_t mesh_dims = mesh_tensor->GetTensorShape()->GetDims();

    KERNEL_CHECK_FALSE((coord_dims == kCoordDimNum), KERNEL_STATUS_PARAM_INVALID,
                       "Dim of coord should be 2, but is [%d].", coord_dims)
    KERNEL_CHECK_FALSE((type_dims == kTypeDimNum), KERNEL_STATUS_PARAM_INVALID, "Dim of type should be 2, but is [%d].",
                       type_dims)
    KERNEL_CHECK_FALSE((natoms_dims == kNatomsDimNum), KERNEL_STATUS_PARAM_INVALID,
                       "Dim of natoms should be 1, but is [%d].", natoms_dims)
    KERNEL_CHECK_FALSE((mesh_dims == kMeshDimNum), KERNEL_STATUS_PARAM_INVALID, "Dim of mesh should be 1, but is [%d].",
                       mesh_dims)
    KERNEL_CHECK_FALSE((natoms_tensor->GetTensorShape()->GetDimSize(0) >= kMinNatomsNum), KERNEL_STATUS_PARAM_INVALID,
                       "Number of atoms should be larger than (or equal to) 3.")

    // check input type
    DataType coord_type = coord_tensor->GetDataType();
    DataType type_type = type_tensor->GetDataType();
    DataType mesh_type = mesh_tensor->GetDataType();
    KERNEL_CHECK_FALSE((coord_type == DT_FLOAT || coord_type == DT_DOUBLE), KERNEL_STATUS_PARAM_INVALID,
                       "Type of coord should be DT_FLOAT or DT_DOUBLE, but is [%s].", DTypeStr(coord_type).c_str())
    KERNEL_CHECK_FALSE((type_type == DT_INT32), KERNEL_STATUS_PARAM_INVALID,
                       "Type of type should be DT_INT32, but is [%s].", DTypeStr(type_type).c_str())
    KERNEL_CHECK_FALSE((mesh_type == DT_INT32), KERNEL_STATUS_PARAM_INVALID,
                       "Type of mesh should be DT_INT32, but is [%s].", DTypeStr(mesh_type).c_str())

    auto natoms = static_cast<int32_t*>(natoms_tensor->GetData());
    KERNEL_CHECK_NULLPTR(natoms, KERNEL_STATUS_PARAM_INVALID, "Get input natoms data failed.")
    int32_t nall = natoms[1];
    int32_t nsamples = coord_tensor->GetTensorShape()->GetDimSize(0);
    KERNEL_CHECK_FALSE((nsamples == type_tensor->GetTensorShape()->GetDimSize(0)), KERNEL_STATUS_PARAM_INVALID,
                       "Number of samples should match between coord and type.")
    KERNEL_CHECK_FALSE((nall * kCoordinateXyzNum == coord_tensor->GetTensorShape()->GetDimSize(1)),
                       KERNEL_STATUS_PARAM_INVALID, "Number of coord should match.")
    KERNEL_CHECK_FALSE((nall == type_tensor->GetTensorShape()->GetDimSize(1)), KERNEL_STATUS_PARAM_INVALID,
                       "Number of type should match.")
    return KERNEL_STATUS_OK;
}

void ProdEnvMatACalcRijCpuKernel::CumSum(std::vector<int64_t>& sec, const std::vector<int64_t>& n_sel) const
{
    sec.resize(n_sel.size() + 1);
    sec[0] = 0;
    for (uint32_t ii = 1; ii < sec.size(); ii++) {
        sec[ii] = sec[ii - 1] + n_sel[ii - 1];
    }
}

template <typename FPTYPE>
uint32_t ProdEnvMatACalcRijCpuKernel::DoProdEnvMatACalcRijCompute(const CpuKernelContext& ctx) const
{
    Tensor* coord_tensor = ctx.Input(kCoordInputIdx);
    Tensor* natoms_tensor = ctx.Input(kNatomsInputIdx);
    Tensor* rij_tensor = ctx.Output(kRijOutputIdx);

    AttrValue* sel_a_attr = ctx.GetAttr("sel_a");
    KERNEL_CHECK_NULLPTR(sel_a_attr, KERNEL_STATUS_PARAM_INVALID, "Get attr [sel_a] failed.")
    std::vector<int64_t> sel_a = sel_a_attr->GetListInt();
    std::vector<int64_t> sec_a;
    CumSum(sec_a, sel_a);
    int32_t nnei = static_cast<int32_t>(sec_a.back());

    auto natoms = static_cast<int32_t*>(natoms_tensor->GetData());
    int32_t nloc = natoms[0];
    int32_t nsamples = coord_tensor->GetTensorShape()->GetDimSize(0);

    FPTYPE* p_rij = static_cast<FPTYPE*>(rij_tensor->GetData());
    int32_t batch_output_datalen = nloc * nnei * kCoordinateXyzNum;
    for (int32_t ff = 0; ff < nsamples; ff++) {
        FPTYPE* rij = p_rij + ff * batch_output_datalen;
        KERNEL_HANDLE_ERROR(ProdEnvMatARijCal<FPTYPE>(rij, ctx, ff, nnei, sec_a), "[%s] parallel compute failed.",
                            kProdEnvMatACalcRij);
    }
    return KERNEL_STATUS_OK;
}

template <typename FPTYPE>
uint32_t ProdEnvMatACalcRijCpuKernel::ProdEnvMatARijCal(FPTYPE* rij, const CpuKernelContext& ctx, int32_t batch_index,
                                                        int32_t nnei, const std::vector<int64_t>& sec_a) const
{
    auto natoms = static_cast<int32_t*>(ctx.Input(kNatomsInputIdx)->GetData());
    int32_t nloc = natoms[0];
    Tensor* coord_tensor = ctx.Input(kCoordInputIdx);
    int32_t nall = coord_tensor->GetTensorShape()->GetDimSize(1) / kCoordinateXyzNum;
    AttrValue* rcut_r_attr = ctx.GetAttr("rcut_r");
    KERNEL_CHECK_NULLPTR(rcut_r_attr, KERNEL_STATUS_PARAM_INVALID, "Get attr [rcut_r] failed.")
    float rcut = rcut_r_attr->GetFloat();

    // Gather everything one sample's helpers need; output pointers already advanced to this sample's slice.
    RijBlock<FPTYPE> blk;
    blk.rij = rij;
    blk.nlist = static_cast<int32_t*>(ctx.Output(kNlistOutputIdx)->GetData()) + batch_index * nloc * nnei;
    blk.distance = static_cast<FPTYPE*>(ctx.Output(kDistanceOutputIdx)->GetData()) + batch_index * nloc * nnei;
    blk.rij_x = static_cast<FPTYPE*>(ctx.Output(kRijXOutputIdx)->GetData()) + batch_index * nloc * nnei;
    blk.rij_y = static_cast<FPTYPE*>(ctx.Output(kRijYOutputIdx)->GetData()) + batch_index * nloc * nnei;
    blk.rij_z = static_cast<FPTYPE*>(ctx.Output(kRijZOutputIdx)->GetData()) + batch_index * nloc * nnei;
    blk.coord = static_cast<const FPTYPE*>(coord_tensor->GetData()) + batch_index * nall * kCoordinateXyzNum;
    blk.type = static_cast<const int32_t*>(ctx.Input(kTypeInputIdx)->GetData()) + batch_index * nall;
    blk.sec_a = &sec_a;
    blk.nnei = nnei;
    blk.atom_types = static_cast<int32_t>(sec_a.size()) - 1;
    blk.rcutsquared = rcut * rcut;

    // Reinterpret the flat mesh buffer as the neighbor-list view (see kMeshDataLength for the layout).
    auto p_mesh = static_cast<int32_t*>(ctx.Input(kMeshInputIdx)->GetData());
    blk.meshdata.nlocnum = p_mesh[0];
    blk.meshdata.ilist = &p_mesh[kMeshHeaderNum];
    blk.meshdata.numneigh = &p_mesh[kMeshHeaderNum + blk.meshdata.nlocnum];
    blk.meshdata.firstneigh = reinterpret_cast<int32_t(*)[kNeighborMaxNum]>(
        &p_mesh[kMeshHeaderNum + kMeshSectionsBeforeNeigh * blk.meshdata.nlocnum]);
    blk.needmap = ctx.Input(kMeshInputIdx)->GetTensorShape()->GetDimSize(0) >
                  (kMeshHeaderNum + static_cast<int64_t>(kMeshDataLength) * blk.meshdata.nlocnum);
    if (blk.needmap) {
        blk.meshdata.nallmaptable = &p_mesh[kMeshHeaderNum + kMeshDataLength * blk.meshdata.nlocnum];
    }

    return CpuKernelUtils::ParallelFor(ctx, nloc, 1,
                                       [this, &blk](int32_t start, int32_t end) { ComputeRijBlock(blk, start, end); });
}

template <typename FPTYPE>
void ProdEnvMatACalcRijCpuKernel::ComputeRijBlock(const RijBlock<FPTYPE>& blk, int32_t start, int32_t end) const
{
    // Reset this block's output region: rij/rij_x/y/z to 0, nlist to -1, distance to an unreachable rcut^2 + 1.
    int32_t rijaxeslen = static_cast<int32_t>((end - start) * blk.nnei * static_cast<int32_t>(sizeof(FPTYPE)));
    (void)memset_s(blk.rij + start * blk.nnei * kCoordinateXyzNum, rijaxeslen * kCoordinateXyzNum, 0x00,
                   rijaxeslen * kCoordinateXyzNum);
    std::fill(blk.nlist + start * blk.nnei, blk.nlist + end * blk.nnei, -1);
    std::fill(blk.distance + start * blk.nnei, blk.distance + end * blk.nnei, blk.rcutsquared + 1.0);
    (void)memset_s(blk.rij_x + start * blk.nnei, rijaxeslen, 0x00, rijaxeslen);
    (void)memset_s(blk.rij_y + start * blk.nnei, rijaxeslen, 0x00, rijaxeslen);
    (void)memset_s(blk.rij_z + start * blk.nnei, rijaxeslen, 0x00, rijaxeslen);
    for (int32_t ii = start; ii < end; ii++) {
        int32_t num_neighbor = blk.meshdata.numneigh[ii];
        if (num_neighbor == -1) {
            continue;
        }
        // firstneigh reserves only kNeighborMaxNum slots per atom; clamp so a malformed numneigh cannot read past them.
        num_neighbor = std::min(num_neighbor, kNeighborMaxNum);
        int32_t core_natom_index = blk.meshdata.ilist[ii];
        FPTYPE i_x = blk.coord[core_natom_index * kCoordinateXyzNum];
        FPTYPE i_y = blk.coord[core_natom_index * kCoordinateXyzNum + 1];
        FPTYPE i_z = blk.coord[core_natom_index * kCoordinateXyzNum + 2];
        std::vector<std::vector<NeighborInfo>> sel = CollectSortedNeighbors(blk, ii, num_neighbor, i_x, i_y, i_z);
        WriteAtomResults(blk, ii, i_x, i_y, i_z, sel);
    }
}

template <typename FPTYPE>
std::vector<std::vector<ProdEnvMatACalcRijCpuKernel::NeighborInfo>> ProdEnvMatACalcRijCpuKernel::CollectSortedNeighbors(
    const RijBlock<FPTYPE>& blk, int32_t ii, int32_t num_neighbor, FPTYPE i_x, FPTYPE i_y, FPTYPE i_z) const
{
    const int32_t* neighbors = blk.meshdata.firstneigh[ii];
    std::vector<std::vector<NeighborInfo>> sel(blk.atom_types, std::vector<NeighborInfo>());
    for (auto& one_type : sel) {
        one_type.reserve(static_cast<size_t>(num_neighbor));
    }
    for (int32_t j = 0; j < num_neighbor; j++) {
        int32_t j_idx = neighbors[j];
        FPTYPE dx = blk.coord[j_idx * kCoordinateXyzNum] - i_x;
        FPTYPE dy = blk.coord[j_idx * kCoordinateXyzNum + 1] - i_y;
        FPTYPE dz = blk.coord[j_idx * kCoordinateXyzNum + 2] - i_z;
        FPTYPE rr = dx * dx + dy * dy + dz * dz;
        if (rr < blk.rcutsquared) {
            // Guard the type-indexed bucket: a type value outside [0, atom_types) — including every value when sel_a
            // is empty and atom_types is 0 — would otherwise index sel out of range.
            int32_t j_type = blk.type[j_idx];
            if (j_type >= 0 && j_type < blk.atom_types) {
                sel[j_type].emplace_back(static_cast<float>(rr), j_idx);
            }
        }
    }
    for (auto& one_type : sel) {
        std::sort(one_type.begin(), one_type.end());
    }
    return sel;
}

template <typename FPTYPE>
void ProdEnvMatACalcRijCpuKernel::WriteAtomResults(const RijBlock<FPTYPE>& blk, int32_t ii, FPTYPE i_x, FPTYPE i_y,
                                                   FPTYPE i_z, const std::vector<std::vector<NeighborInfo>>& sel) const
{
    FPTYPE* cur_rij = blk.rij + ii * blk.nnei * kCoordinateXyzNum;
    int32_t* cur_nlist = blk.nlist + ii * blk.nnei;
    FPTYPE* cur_dist = blk.distance + ii * blk.nnei;
    FPTYPE* cur_rij_x = blk.rij_x + ii * blk.nnei;
    FPTYPE* cur_rij_y = blk.rij_y + ii * blk.nnei;
    FPTYPE* cur_rij_z = blk.rij_z + ii * blk.nnei;
    const std::vector<int64_t>& sec_a = *blk.sec_a;
    for (uint32_t m_k = 0; m_k < sel.size(); m_k++) {
        uint32_t sorted_idx = 0;
        int64_t cnt_size = std::min(sec_a[m_k + 1], sec_a[m_k] + static_cast<int64_t>(sel[m_k].size()));
        for (int64_t res_idx = sec_a[m_k]; res_idx < cnt_size; res_idx++) {
            int32_t cur_atom_idx = sel[m_k][sorted_idx].index;
            cur_nlist[res_idx] = blk.needmap ? blk.meshdata.nallmaptable[cur_atom_idx] : cur_atom_idx;
            cur_dist[res_idx] = sel[m_k][sorted_idx].dist;
            FPTYPE dx = blk.coord[cur_atom_idx * kCoordinateXyzNum] - i_x;
            FPTYPE dy = blk.coord[cur_atom_idx * kCoordinateXyzNum + 1] - i_y;
            FPTYPE dz = blk.coord[cur_atom_idx * kCoordinateXyzNum + 2] - i_z;
            cur_rij[res_idx * kCoordinateXyzNum] = dx;
            cur_rij[res_idx * kCoordinateXyzNum + 1] = dy;
            cur_rij[res_idx * kCoordinateXyzNum + 2] = dz;
            cur_rij_x[res_idx] = dx;
            cur_rij_y[res_idx] = dy;
            cur_rij_z[res_idx] = dz;
            sorted_idx++;
        }
    }
}

REGISTER_CPU_KERNEL(kProdEnvMatACalcRij, ProdEnvMatACalcRijCpuKernel);
} // namespace aicpu
