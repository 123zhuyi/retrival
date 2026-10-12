/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file squared_difference_tiling.cpp
 * \brief SquaredDifference 算子 Tiling 实现（ops-math 结构）
 *        OneDim / BRC 双分支，5 dtype x 2 mode 共 10 个 tiling key
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/squared_difference_tiling_data.h"
#include "../op_kernel/squared_difference_tiling_key.h"
#include <algorithm>
#include <cstring>
#include <limits>

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t kUbBlockSize = SD_UB_BLOCK_SIZE;

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "coreNum <= 0"), return ge::GRAPH_FAILED);
    ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize <= 0, OP_LOGE(context, "ubSize <= 0"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ubSize < kUbBlockSize || ubSize % kUbBlockSize != 0, OP_LOGE(context, "invalid ubSize: %lu", ubSize),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

// ---------- 补维 + 广播 + 合轴 ----------
struct Collapsed {
    int32_t len = 0;
    int64_t outDims[SD_MAX_DIM];
    int64_t s1[SD_MAX_DIM];
    int64_t s2[SD_MAX_DIM];
    int64_t so[SD_MAX_DIM];
    int64_t total = 1;
};

static bool CollapseDims(const gert::Shape& a0, const gert::Shape& b0, Collapsed& c)
{
    // 合轴前工作数组用更大上限（原始 rank 可远超合轴后维度）；
    // 只对合轴后的维度数 m 施加 SD_MAX_DIM 约束。
    constexpr int SD_MAX_RANK = 16;
    int na = static_cast<int>(a0.GetDimNum());
    int nb = static_cast<int>(b0.GetDimNum());
    int n = std::max(na, nb);
    if (n == 0)
        n = 1;
    if (n > SD_MAX_RANK)
        return false;

    int64_t a[SD_MAX_RANK], b[SD_MAX_RANK], o[SD_MAX_RANK] = {};
    std::fill(a, a + SD_MAX_RANK, 1);
    std::fill(b, b + SD_MAX_RANK, 1);
    for (int i = 0; i < na; i++)
        a[n - na + i] = a0.GetDim(i);
    for (int i = 0; i < nb; i++)
        b[n - nb + i] = b0.GetDim(i);
    for (int i = 0; i < n; i++) {
        if (a[i] != b[i] && a[i] != 1 && b[i] != 1)
            return false;
        o[i] = (a[i] == b[i]) ? a[i] : ((a[i] == 1) ? b[i] : a[i]);
    }

    int flag[SD_MAX_RANK] = {};
    for (int i = 0; i < n; i++) {
        if (a[i] == 1 && o[i] != 1)
            flag[i] |= 1;
        if (b[i] == 1 && o[i] != 1)
            flag[i] |= 2;
    }

    int64_t ma[SD_MAX_RANK], mb[SD_MAX_RANK], mo[SD_MAX_RANK];
    int mflag[SD_MAX_RANK];
    int m = 0;
    for (int i = 0; i < n; i++) {
        if (m > 0 && flag[i] == mflag[m - 1]) {
            ma[m - 1] *= a[i];
            mb[m - 1] *= b[i];
            mo[m - 1] *= o[i];
        } else {
            ma[m] = a[i];
            mb[m] = b[i];
            mo[m] = o[i];
            mflag[m] = flag[i];
            m++;
        }
    }
    // 合轴后维度数才是 kernel 的真正约束
    if (m > SD_MAX_DIM)
        return false;
    c.len = m;
    int64_t acca = 1, accb = 1, acco = 1;
    for (int i = m - 1; i >= 0; i--) {
        c.outDims[i] = mo[i];
        c.so[i] = acco;
        acco *= mo[i];
        c.s1[i] = (ma[i] == 1 && mo[i] != 1) ? 0 : acca;
        acca *= ma[i];
        c.s2[i] = (mb[i] == 1 && mo[i] != 1) ? 0 : accb;
        accb *= mb[i];
    }
    c.total = acco;
    return true;
}

// ---------- Tiling 计算 ----------
static int64_t Align32(int64_t bytes) { return CeilDiv(bytes, kUbBlockSize) * kUbBlockSize; }

static bool CheckedMul(int64_t lhs, int64_t rhs, int64_t& result)
{
    if (lhs < 0 || rhs < 0)
        return false;
    if (lhs == 0 || rhs == 0) {
        result = 0;
        return true;
    }
    if (lhs > std::numeric_limits<int64_t>::max() / rhs)
        return false;
    result = lhs * rhs;
    return true;
}

static bool PlanBlocks(SquaredDifferenceTilingData& t, int64_t coreNum, int32_t dtypeSize)
{
    (void)dtypeSize;
    const int64_t fused = t.fusedProduct;
    if (fused <= 0 || coreNum <= 0)
        return false;

    const int64_t legacyBlocks = CeilDiv(fused, CeilDiv(fused, coreNum));
    const int64_t targetBlocks = std::min(coreNum, fused);
    const bool promote = fused >= coreNum;

    t.blockPolicy = promote ? 1 : 0;
    t.blockNum = promote ? targetBlocks : legacyBlocks;
    if (t.blockNum < 1 || t.blockNum > targetBlocks)
        return false;
    t.blockBase = fused / t.blockNum;
    t.blockRemainder = fused % t.blockNum;
    return t.blockBase >= 1 && t.blockRemainder >= 0 && t.blockRemainder < t.blockNum;
}

static int64_t ActualUbBytes(int32_t mode, int32_t dtypeKey, int32_t dtypeSize, int64_t allocElem)
{
    int32_t qDepth = 2;
    int64_t dataBytes = Align32(allocElem * dtypeSize);
    int64_t total = 3 * qDepth * dataBytes;
    if (mode == SD_MODE_BRC) {
        if (dtypeKey != SD_DT_INT64) {
            total += 2 * dataBytes;
            total += dataBytes;
        }
    }
    if (dtypeKey == SD_DT_FP16 || dtypeKey == SD_DT_BF16) {
        total += 2 * Align32(allocElem * static_cast<int64_t>(sizeof(float)));
    }
    return total;
}

// 快速路径 UB 预算：x1/y 队列(qDepth) + x2 小缓冲 + work1_ + 广播 tmp + cast（不分配 work2_/brcTmp_）
static int64_t ActualUbBytesFast(int32_t dtypeKey, int32_t dtypeSize, int64_t allocElem, int64_t srcTileElems,
                                 int64_t brcTmpBytes)
{
    int32_t qDepth = 2;
    int64_t dataBytes = Align32(allocElem * dtypeSize);
    int64_t total = (2 * qDepth + 1) * dataBytes;        // stgQ1_ + outQB_ + work1_
    total += qDepth * Align32(srcTileElems * dtypeSize); // stgQ2_（广播源，很小）
    if (dtypeKey == SD_DT_FP16 || dtypeKey == SD_DT_BF16) {
        total += 2 * Align32(allocElem * static_cast<int64_t>(sizeof(float)));
    }
    total += brcTmpBytes;
    return total;
}

// BRC 快速路径（单轴广播）：对齐 TBE 的「inner 合并广播轴 + 紧凑展开 + 连续搬移」。
// 当前仅覆盖尾维广播（kind=2，b == len-1）；中间维广播（kind=1）暂走旧路径。
// int64 走旧路径（标量计算）。
static bool ComputeTilingFastBrc(const Collapsed& c, int32_t dtypeKey, int32_t dtypeSize, int64_t coreNum,
                                 int64_t ubBytes, int32_t brcAxis, SquaredDifferenceTilingData& t)
{
    const int32_t len = c.len;
    const int32_t b = brcAxis;

    int64_t inner = 1;
    for (int i = b; i < len; i++) {
        if (!CheckedMul(inner, c.outDims[i], inner))
            return false;
    }
    int64_t broadcastLen = c.outDims[b];
    int64_t innerSrc = inner / broadcastLen;
    int64_t outerTotal = 1;
    for (int i = 0; i < b; i++) {
        if (!CheckedMul(outerTotal, c.outDims[i], outerTotal))
            return false;
    }
    if (inner <= 0 || innerSrc <= 0 || outerTotal <= 0)
        return false;

    bool x1Brc = (c.s1[b] == 0);
    bool x2Brc = (c.s2[b] == 0);
    if (x1Brc == x2Brc)
        return false; // 同真(标量)或同假，走旧路径
    t.brcWhich = x1Brc ? 1 : 2;

    t.mode = SD_MODE_BRC;
    t.ubSplitAxis = b;
    t.innerDim = inner;
    t.broadcastLen = broadcastLen;
    t.innerSrc = innerSrc;
    t.brcKind = (b == len - 1) ? 2 : 1;
    t.outerTotal = outerTotal;

    int64_t ubUsable = ubBytes - ubBytes / 16;
    if (ubUsable <= 0)
        return false;

    // 展开 tmp 缓冲大小（元素）与 x2 源大小
    int64_t elemPer32B = kUbBlockSize / dtypeSize;
    int64_t qDepth = 2;
    int64_t perElemApprox = (2 * qDepth + 1) * dtypeSize +
                            ((dtypeKey == SD_DT_FP16 || dtypeKey == SD_DT_BF16) ? 2 * 4 : 0);
    int64_t alignInnerSrc = ((innerSrc + elemPer32B - 1) / elemPer32B) * elemPer32B;
    int64_t srcPerOuter = (b == len - 1) ? innerSrc : alignInnerSrc; // kind=2 紧凑，kind=1 补 pad
    int64_t tmpPerOuter = 0; // 每外层 tmp 字节（kind=1/kind=2 都用 per-outer 计算）
    int64_t tmpBase = 0;     // kind=2：单次 repeat(16 行) 的 tmp 字节
    if (b == len - 1) {
        int64_t numBlocksAlign = ((broadcastLen + elemPer32B - 1) / elemPer32B) * elemPer32B;
        tmpBase = (elemPer32B * elemPer32B + elemPer32B * numBlocksAlign) * dtypeSize;
    } else {
        tmpPerOuter = broadcastLen * alignInnerSrc * dtypeSize;
    }
    auto TmpBytes = [&](int64_t ot) {
        if (b == len - 1) {
            int64_t repeats = (ot + elemPer32B - 1) / elemPer32B;
            return repeats * tmpBase;
        }
        return ot * tmpPerOuter;
    };

    int64_t outerTile = (ubUsable / perElemApprox) / inner;
    if (outerTile < 1)
        outerTile = 1;
    if (outerTile > outerTotal)
        outerTile = outerTotal;
    while (outerTile > 1) {
        int64_t mt = outerTile * inner;
        int64_t src = outerTile * srcPerOuter;
        int64_t tmp = TmpBytes(outerTile);
        if (ActualUbBytesFast(dtypeKey, dtypeSize, mt, src, tmp) <= ubUsable)
            break;
        outerTile--;
    }
    {
        int64_t mt = outerTile * inner;
        int64_t src = outerTile * srcPerOuter;
        int64_t tmp = TmpBytes(outerTile);
        if (ActualUbBytesFast(dtypeKey, dtypeSize, mt, src, tmp) > ubUsable) {
            return false; // inner 单外层都放不下，回退旧路径
        }
    }

    if (outerTile < 1)
        return false; // inner 放不下 UB，回退旧路径
    if (outerTile > outerTotal)
        outerTile = outerTotal;

    t.ubFormer = outerTile;
    t.ubOuter = CeilDiv(outerTotal, outerTile);
    t.ubTail = outerTotal - (t.ubOuter - 1) * outerTile;
    t.maxTileElem = outerTile * inner;
    t.srcTileElems = Align32(outerTile * srcPerOuter * dtypeSize) / dtypeSize;
    t.fusedProduct = t.ubOuter;
    t.alignInner = inner;
    t.nFormer = inner;
    t.nOuter = 1;
    t.nTail = inner;
    if (!PlanBlocks(t, coreNum, dtypeSize))
        return false;
    return true;
}

// int64 单轴广播：整广播轴一个 tile（M=广播轴长），内轴 N 切到 ~coreNum 块。
// kernel 读广播源一次、标量跨 M 广播，消除外层广播造成的冗余搬移与冗余 tile。
static bool ComputeTilingInt64Bcast(const Collapsed& c, int32_t dtypeKey, int32_t dtypeSize, int64_t coreNum,
                                    int64_t ubBytes, int32_t brcAxis, SquaredDifferenceTilingData& t)
{
    (void)dtypeKey;
    const int32_t len = c.len;
    int64_t broadcastLen = c.outDims[brcAxis];
    int64_t inner = 1;
    for (int i = brcAxis + 1; i < len; i++) {
        if (!CheckedMul(inner, c.outDims[i], inner))
            return false;
    }
    int64_t outerTotal = 1;
    for (int i = 0; i < brcAxis; i++) {
        if (!CheckedMul(outerTotal, c.outDims[i], outerTotal))
            return false;
    }
    if (broadcastLen <= 0 || inner <= 0)
        return false;

    bool x1Brc = (c.s1[brcAxis] == 0);
    bool x2Brc = (c.s2[brcAxis] == 0);
    if (x1Brc == x2Brc)
        return false; // 同真(标量)或同假，走旧路径
    t.brcWhich = x1Brc ? 1 : 2;

    int64_t ubUsable = ubBytes - ubBytes / 16;
    int64_t elemPer32B = kUbBlockSize / dtypeSize;

    // N 切到 ~coreNum 块保证满核
    int64_t nFormer = CeilDiv(inner, coreNum);
    nFormer = CeilDiv(nFormer, elemPer32B) * elemPer32B;
    if (nFormer < elemPer32B)
        nFormer = elemPer32B;
    if (nFormer > inner)
        nFormer = inner;
    while (nFormer > elemPer32B) {
        int64_t bytes = (2 * broadcastLen + 1) * Align32(nFormer * dtypeSize);
        if (bytes <= ubUsable)
            break;
        nFormer -= elemPer32B;
    }
    if ((2 * broadcastLen + 1) * Align32(nFormer * dtypeSize) > ubUsable)
        return false;

    int64_t nOuter = CeilDiv(inner, nFormer);
    int64_t nTail = inner - (nOuter - 1) * nFormer;
    int64_t fused = outerTotal * nOuter;
    if (fused <= 0)
        return false;

    t.mode = SD_MODE_BRC;
    t.ubSplitAxis = brcAxis;
    t.innerDim = inner;
    t.broadcastLen = broadcastLen;
    t.outerTotal = outerTotal;
    t.ubFormer = broadcastLen;
    t.ubOuter = 1;
    t.ubTail = broadcastLen;
    t.nFormer = nFormer;
    t.nOuter = nOuter;
    t.nTail = nTail;
    t.alignInner = nFormer;
    t.maxTileElem = broadcastLen * nFormer;
    t.srcTileElems = Align32(nFormer * dtypeSize) / dtypeSize;
    t.fusedProduct = fused;
    t.bcastOnM = 1;
    if (!PlanBlocks(t, coreNum, dtypeSize))
        return false;
    return true;
}

static bool ComputeTiling(const Collapsed& c, int32_t dtypeKey, int32_t dtypeSize, int64_t coreNum, int64_t ubBytes,
                          SquaredDifferenceTilingData& t)
{
    memset(&t, 0, sizeof(t));
    t.dtypeKey = dtypeKey;
    t.shapeLen = c.len;
    t.totalLength = c.total;
    for (int i = 0; i < c.len; i++) {
        t.outDims[i] = c.outDims[i];
        t.x1Strides[i] = c.s1[i];
        t.x2Strides[i] = c.s2[i];
        t.outStrides[i] = c.so[i];
    }

    if (c.total == 0) {
        t.mode = (c.len == 1) ? SD_MODE_ONEDIM : SD_MODE_BRC;
        t.blockNum = 1;
        t.blockBase = 0;
        t.blockRemainder = 0;
        return true;
    }

    int64_t ubUsable = ubBytes - ubBytes / 16;
    if (ubUsable <= 0)
        return false;

    bool needCast = (dtypeKey == SD_DT_FP16 || dtypeKey == SD_DT_BF16);
    bool needBrcTmp = (dtypeKey != SD_DT_INT64);
    int64_t qDepth = 2;
    int64_t perElemOne = 3 * dtypeSize * qDepth + (needCast ? 2 * 4 : 0);
    int64_t perElemBrc = 3 * dtypeSize * qDepth + (needBrcTmp ? 3 * dtypeSize : 0) + (needCast ? 2 * 4 : 0);
    constexpr int64_t tileSearchStep = 128;
    int64_t maxElemOne = FloorAlign(ubUsable / perElemOne, tileSearchStep);
    int64_t maxElemBrc = FloorAlign(ubUsable / perElemBrc, tileSearchStep);
    while (maxElemOne >= tileSearchStep && ActualUbBytes(SD_MODE_ONEDIM, dtypeKey, dtypeSize, maxElemOne) > ubUsable) {
        maxElemOne -= tileSearchStep;
    }
    while (maxElemBrc >= tileSearchStep && ActualUbBytes(SD_MODE_BRC, dtypeKey, dtypeSize, maxElemBrc) > ubUsable) {
        maxElemBrc -= tileSearchStep;
    }
    if (maxElemOne < tileSearchStep || maxElemBrc < tileSearchStep)
        return false;
    int64_t elemPer32B = kUbBlockSize / dtypeSize;

    if (c.len == 1) {
        t.mode = SD_MODE_ONEDIM;
        t.x1Scalar = (c.s1[0] == 0) ? 1 : 0;
        t.x2Scalar = (c.s2[0] == 0) ? 1 : 0;
        int64_t dimLen = c.outDims[0];
        int64_t ubFormer = std::min(maxElemOne, dimLen);
        if (dtypeKey == SD_DT_INT64 && dimLen > coreNum) {
            int64_t parallelTile = CeilDiv(dimLen, coreNum);
            parallelTile = CeilDiv(parallelTile, elemPer32B) * elemPer32B;
            if (parallelTile < ubFormer)
                ubFormer = parallelTile;
        }
        if (ubFormer < 1)
            ubFormer = 1;
        int64_t ubOuter = CeilDiv(dimLen, ubFormer);
        int64_t ubTail = dimLen - (ubOuter - 1) * ubFormer;
        t.ubFormer = ubFormer;
        t.ubOuter = ubOuter;
        t.ubTail = ubTail;
        t.fusedProduct = ubOuter;
        t.innerDim = 1;
        t.alignInner = 1;
        t.maxTileElem = ubFormer;
        if (!PlanBlocks(t, coreNum, dtypeSize))
            return false;
    } else {
        // 单轴广播快速路径（尾维广播 kind=2，非 int64）
        int32_t brcAxis = -1;
        int32_t brcCnt = 0;
        for (int i = 0; i < c.len; i++) {
            if (c.s1[i] == 0 || c.s2[i] == 0) {
                if (brcAxis < 0)
                    brcAxis = i;
                brcCnt++;
            }
        }
        if (brcCnt == 1 && dtypeKey != SD_DT_INT64 &&
            ComputeTilingFastBrc(c, dtypeKey, dtypeSize, coreNum, ubBytes, brcAxis, t)) {
            return true;
        }
        // 快速路径失败回退旧路径：清掉 fast path 已写入的 brcKind，避免 kernel 误判
        t.brcKind = 0;
        // int64 单轴广播：广播轴=M、内轴 N 切分，kernel 读广播源一次、标量跨 M 广播
        if (brcCnt == 1 && dtypeKey == SD_DT_INT64 &&
            ComputeTilingInt64Bcast(c, dtypeKey, dtypeSize, coreNum, ubBytes, brcAxis, t)) {
            return true;
        }

        t.mode = SD_MODE_BRC;
        int32_t sa = c.len - 2;
        t.ubSplitAxis = sa;
        int64_t N = c.outDims[c.len - 1];
        t.innerDim = N;
        int64_t Mdim = c.outDims[sa];
        int64_t outerProd = 1;
        for (int i = 0; i < sa; i++) {
            if (!CheckedMul(outerProd, c.outDims[i], outerProd))
                return false;
        }

        int64_t alignNfull = ((N + elemPer32B - 1) / elemPer32B) * elemPer32B;
        int64_t alignN, ubFormer, nFormer, nOuter, nTail;
        int64_t rowBlocks = alignNfull * dtypeSize / kUbBlockSize;
        if (alignNfull <= maxElemBrc && rowBlocks <= SD_DATACOPY_MAX_BLOCK_COUNT) {
            // 整行放得下：保持上次提交行为，切 M 轴，N 不切分（nOuter=1）
            alignN = alignNfull;
            nFormer = N;
            nOuter = 1;
            nTail = N;
            ubFormer = maxElemBrc / (alignN > 0 ? alignN : 1);
            if (ubFormer > Mdim)
                ubFormer = Mdim;
            if (ubFormer > SD_DATACOPY_MAX_BLOCK_COUNT)
                ubFormer = SD_DATACOPY_MAX_BLOCK_COUNT;
            while (ubFormer > 0 && ActualUbBytes(SD_MODE_BRC, dtypeKey, dtypeSize, ubFormer * alignN) > ubUsable) {
                --ubFormer;
            }
            if (ubFormer < 1)
                return false;
        } else {
            // 单行超预算：M-tile=1，切 N 轴。走 kernel 独立 N 切分路径。
            ubFormer = 1;
            int64_t maxNByCopy = SD_DATACOPY_MAX_BLOCK_COUNT * elemPer32B;
            nFormer = FloorAlign(std::min(maxElemBrc, maxNByCopy), elemPer32B);
            while (nFormer >= elemPer32B && ActualUbBytes(SD_MODE_BRC, dtypeKey, dtypeSize, nFormer) > ubUsable) {
                nFormer -= elemPer32B;
            }
            if (nFormer < elemPer32B)
                return false;
            nOuter = CeilDiv(N, nFormer);
            nTail = N - (nOuter - 1) * nFormer;
            alignN = nFormer; // 已 32B 对齐
        }
        if (dtypeKey == SD_DT_INT64) {
            int64_t groups = outerProd * nOuter;
            if (groups < coreNum) {
                int64_t targetTiles = CeilDiv(coreNum, groups);
                int64_t parallelRows = CeilDiv(Mdim, targetTiles);
                if (parallelRows < ubFormer)
                    ubFormer = parallelRows;
                if (ubFormer < 1)
                    ubFormer = 1;
            }
        }
        t.alignInner = alignN;
        t.nFormer = nFormer;
        t.nOuter = nOuter;
        t.nTail = nTail;

        int64_t ubOuter = CeilDiv(Mdim, ubFormer);
        int64_t ubTail = Mdim - (ubOuter - 1) * ubFormer;
        if (nOuter > 1 && (ubFormer != 1 || ubOuter != Mdim || ubTail != 1))
            return false;
        int64_t fused = 0;
        if (!CheckedMul(outerProd, ubOuter, fused) || !CheckedMul(fused, nOuter, fused))
            return false;
        t.ubFormer = ubFormer;
        t.ubOuter = ubOuter;
        t.ubTail = ubTail;
        t.fusedProduct = fused;
        t.maxTileElem = ubFormer * alignN;
        if (!PlanBlocks(t, coreNum, dtypeSize))
            return false;
    }
    return true;
}

// ---------- Tiling 入口 ----------
static ge::graphStatus SquaredDifferenceTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
                return ge::GRAPH_FAILED);

    auto* x1Desc = context->GetInputDesc(0);
    auto* x2Desc = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Desc);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Desc);

    // dtype
    ge::DataType dt = x1Desc->GetDataType();
    int32_t dtypeKey, dtypeSize;
    if (dt == ge::DT_FLOAT) {
        dtypeKey = SD_DT_FP32;
        dtypeSize = 4;
    } else if (dt == ge::DT_FLOAT16) {
        dtypeKey = SD_DT_FP16;
        dtypeSize = 2;
    } else if (dt == ge::DT_BF16) {
        dtypeKey = SD_DT_BF16;
        dtypeSize = 2;
    } else if (dt == ge::DT_INT32) {
        dtypeKey = SD_DT_INT32;
        dtypeSize = 4;
    } else if (dt == ge::DT_INT64) {
        dtypeKey = SD_DT_INT64;
        dtypeSize = 8;
    } else {
        OP_LOGE(context, "unsupported dtype");
        return ge::GRAPH_FAILED;
    }

    // 合轴 — null check before deref
    auto* s1raw = context->GetInputShape(0);
    auto* s2raw = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, s1raw);
    OP_CHECK_NULL_WITH_CONTEXT(context, s2raw);
    const gert::Shape& s1ref = s1raw->GetOriginShape();
    const gert::Shape& s2ref = s2raw->GetOriginShape();

    Collapsed col;
    if (!CollapseDims(s1ref, s2ref, col)) {
        OP_LOGE(context, "CollapseDims failed (shapes not broadcastable)");
        return ge::GRAPH_FAILED;
    }

    SquaredDifferenceTilingData* tiling = context->GetTilingData<SquaredDifferenceTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    OP_CHECK_IF(!ComputeTiling(col, dtypeKey, dtypeSize, coreNum, static_cast<int64_t>(ubSize), *tiling),
                OP_LOGE(context, "insufficient UB for squared_difference tiling"), return ge::GRAPH_FAILED);

    context->SetBlockDim(static_cast<uint32_t>(tiling->blockNum));

    // tilingKey = dtypeKey*2 + mode（10 keys = 5 dtype x 2 mode）
    // if constexpr 确保每个 kernel 实例只编译一种 dtype 的 buffer，防止 UB 溢出
    int keyIdx = dtypeKey * 2 + (tiling->mode == SD_MODE_BRC ? 1 : 0);
    uint64_t tilingKey;
    switch (keyIdx) {
        case 0:
            tilingKey = GET_TPL_TILING_KEY(SD_KEY_FP32_ONEDIM);
            break;
        case 1:
            tilingKey = GET_TPL_TILING_KEY(SD_KEY_FP32_BRC);
            break;
        case 2:
            tilingKey = GET_TPL_TILING_KEY(SD_KEY_FP16_ONEDIM);
            break;
        case 3:
            tilingKey = GET_TPL_TILING_KEY(SD_KEY_FP16_BRC);
            break;
        case 4:
            tilingKey = GET_TPL_TILING_KEY(SD_KEY_BF16_ONEDIM);
            break;
        case 5:
            tilingKey = GET_TPL_TILING_KEY(SD_KEY_BF16_BRC);
            break;
        case 6:
            tilingKey = GET_TPL_TILING_KEY(SD_KEY_INT32_ONEDIM);
            break;
        case 7:
            tilingKey = GET_TPL_TILING_KEY(SD_KEY_INT32_BRC);
            break;
        case 8:
            tilingKey = GET_TPL_TILING_KEY(SD_KEY_INT64_ONEDIM);
            break;
        default:
            tilingKey = GET_TPL_TILING_KEY(SD_KEY_INT64_BRC);
            break;
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSquaredDifference([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct SquaredDifferenceCompileInfo {};

IMPL_OP_OPTILING(SquaredDifference)
    .Tiling(SquaredDifferenceTilingFunc)
    .TilingParse<SquaredDifferenceCompileInfo>(TilingParseForSquaredDifference);

} // namespace optiling
