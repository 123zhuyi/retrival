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
 * \file prompt_flash_attention_comm.h
 * \brief
 */
#ifndef PROMPT_FLASH_ATTENTION_COMM_H
#define PROMPT_FLASH_ATTENTION_COMM_H
#include "lib/matmul_intf.h"
#include "kernel_tiling/kernel_tiling.h"
// PFATODO ConstPolicySelector放到matmul_modules后，可以不include这些头文件
#include "../matmul_modules/pfa_matmul_policy.h"

constexpr int32_t PFA_PARAMS_QUEUE_CAPBABILITY = 4;
constexpr uint32_t ATTENTION_MASK_MAX_SIZE = 2048;
constexpr uint32_t SOFTMAX_COLUMN_SIZE = 1;

constexpr uint32_t ANTIQUANT_SOUTER_BASE_SIZE = 32;
constexpr uint32_t ANTIQUANT_SINNER_BASE_SIZE = 128;
constexpr uint32_t ANTIQUANT_SOFTMAX_COLUMN_SIZE = 8;

constexpr uint32_t NO_CONSTANT = 0;
constexpr uint32_t SOUTER_CONST_32 = 32;
constexpr uint32_t SOUTER_CONST_64 = 64;
constexpr uint32_t SOUTER_CONST_96 = 96;
constexpr uint32_t SOUTER_CONST_128 = 128;
constexpr uint32_t SINNER_CONST_64 = 64;
constexpr uint32_t SINNER_CONST_128 = 128;
constexpr uint32_t SINNER_CONST_256 = 256;
constexpr uint32_t DSIZE_CONST_64 = 64;
constexpr uint32_t DSIZE_CONST_128 = 128;
constexpr uint32_t DSIZE_CONST_192 = 192;
constexpr uint32_t DSIZE_CONST_256 = 256;
constexpr uint32_t DSIZE_CONST_512 = 512;
constexpr uint32_t DSIZE_CONST_576 = 576;

constexpr static uint32_t PFA_NEGATIVE_MIN_VAULE_FP32 = 0xFF7FFFFF;
constexpr static uint32_t PFA_NEGATIVE_MIN_VAULE_FP16 = 0xC77FE000;
constexpr static uint32_t NEGATIVE_MAX_VAULE_FP32 = 0x7F7FFFFF;

constexpr static int64_t SPARSE_MODE_INT_MAX = 2147483647;

__aicore__ inline constexpr MatmulConfig GetPFANormalConfig() {
    MatmulShapeParams shapeParams = {0, 0, 0, 0, 0, 0};
    auto mmCfg = GetMMConfig<MatmulConfigMode::CONFIG_NORM>(shapeParams);
    mmCfg.enableSetDefineData = true;
    return mmCfg;
};

__aicore__ inline constexpr MatmulConfig GetPFACustomConfig(bool isCO1Shared = true,
    uint32_t sharedCO1BufferSize = 64 * 1024, bool isConcat = true, // 64 and 1024 are default values for sharedCO1BufferSize
    uint32_t singleM = 0, uint32_t singleN = 0,  uint32_t singleK = 0, uint32_t baseM = 0,
    uint32_t baseN = 0, uint32_t baseK = 0, bool isA2B2Shared = true, bool enableSetTail = true) {
    MatmulShapeParams shapeParams = {singleM, singleN, singleK, baseM, baseN, baseK};
    auto mmCfg = GetMMConfig<MatmulConfigMode::CONFIG_NORM>(shapeParams);
    mmCfg.intrinsicsCheck = false;
    mmCfg.enUnitFlag = false;
    mmCfg.enableInit = false;
    mmCfg.enableSetBias = false;
    mmCfg.enableQuantVector = false;
    mmCfg.isBiasBatch = false;
    mmCfg.enableSetOrgShape = true;
    mmCfg.enableSetTail = enableSetTail;
    mmCfg.enableSetDefineData = true;
    mmCfg.isA2B2Shared = isA2B2Shared;
    mmCfg.isCO1Shared = isCO1Shared;
    mmCfg.iterateMode = IterateMode::ITERATE_MODE_DEFAULT;
    mmCfg.sharedCO1BufferSize = sharedCO1BufferSize;
    mmCfg.doNorm = isConcat;
    mmCfg.doIBShareNorm = !isConcat;
    return mmCfg;
}

constexpr MatmulConfig CFG_PFA_NORM = GetPFANormalConfig();

constexpr MatmulConfig CFG_SAMEAB_S1_64_S2_256_D64 = GetPFACustomConfig(
    true,  // isCO1Shared
    (64 * 1024),  // sharedCO1BufferSize
    true,  // doNorm
    64,    // singleM
    256,   // singleN
    64,    // singleK
    64,    // baseM
    256,   // baseN
    64,    // baseK
    true   // isA2B2Shared, L0ab double buffer
);

constexpr MatmulConfig CFG_SAMEAB_S1_128_S2_128_D64 = GetPFACustomConfig(
    true,  // isCO1Shared
    (64 * 1024),  // sharedCO1BufferSize
    true,  // doNorm
    128,   // singleM
    128,   // singleN
    64,    // singleK
    128,   // baseM
    128,   // baseN
    64,    // baseK
    true   // isA2B2Shared, L0ab double buffer
);

constexpr MatmulConfig CFG_SAMEAB_S1_128_S2_256_D64 = GetPFACustomConfig(
    true,  // isCO1Shared
    (128 * 1024),  // sharedCO1BufferSize
    true,  // doNorm
    256,   // singleM
    128,   // singleN
    64,    // singleK
    256,   // baseM
    128,   // baseN
    64,    // baseK
    false  // isA2B2Shared, disable L0ab double buffer
);

constexpr MatmulConfig CFG_SAMEB_S1_64_S2_256_D64 = GetPFACustomConfig(
    true,  // isCO1Shared
    (64 * 1024), // sharedCO1BufferSize
    true,  // isConcat
    64,    // singleM
    64,    // singleN
    256,   // singleK
    64,    // baseM
    64,    // baseN
    256,   // baseK
    true   // isA2B2Shared, L0ab double buffer
);

constexpr MatmulConfig CFG_SAMEB_S1_128_S2_128_D64 = GetPFACustomConfig(
    true,  // isCO1Shared
    (64 * 1024), // sharedCO1BufferSize
    true,  // isConcat
    128,   // singleM
    64,    // singleN
    128,   // singleK
    128,   // baseM
    64,    // baseN
    128,   // baseK
    true   // isA2B2Shared, L0ab double buffer
);

constexpr MatmulConfig CFG_SAMEB_S1_128_S2_256_D64 = GetPFACustomConfig(
    true,  // isCO1Shared
    (128 * 1024),  // sharedCO1BufferSize
    true,  // doNorm
    128,   // singleM
    64,    // singleN
    256,   // singleK
    128,   // baseM
    64,    // baseN
    128,   // baseK
    false  // isA2B2Shared, disable L0ab double buffer
);

constexpr MatmulConfig CFG_SAMEAB_S1_64_S2_256_D128 = GetPFACustomConfig(
    true,  // isCO1Shared
    (64 * 1024),  // sharedCO1BufferSize
    true,  // doNorm
    64,    // singleM
    256,   // singleN
    128,   // singleK
    64,    // baseM
    256,   // baseN
    128,   // baseK
    false  // isA2B2Shared, L0ab double buffer
);

constexpr MatmulConfig CFG_SAMEAB_S1_128_S2_128_D128 = GetPFACustomConfig(
    true,  // isCO1Shared
    (64 * 1024),  // sharedCO1BufferSize
    true,  // doNorm
    128,   // singleM
    128,   // singleN
    128,   // singleK
    128,   // baseM
    128,   // baseN
    128,   // baseK
    true   // isA2B2Shared, L0ab double buffer
);

constexpr MatmulConfig CFG_SAMEB_S1_64_S2_256_D128 = GetPFACustomConfig(
    true,  // isCO1Shared
    (64 * 1024),  // sharedCO1BufferSize
    true,  // doNorm
    64,    // singleM
    128,   // singleN
    256,   // singleK
    64,    // baseM
    128,   // baseN
    256,   // baseK
    false  // isA2B2Shared, L0ab double buffer
);

constexpr MatmulConfig CFG_SAMEB_S1_128_S2_128_D128 = GetPFACustomConfig(
    true,  // isCO1Shared
    (64 * 1024),  // sharedCO1BufferSize
    true,  // doNorm
    128,   // singleM
    128,   // singleN
    128,   // singleK
    128,   // baseM
    128,   // baseN
    128,   // baseK
    true   // isA2B2Shared, L0ab double buffer
);

// IFA PA mm1 D = 256 s2 = 128
constexpr MatmulConfig CFG_SAMEAB_S1_64_S2_128_D256 = GetPFACustomConfig(
    true,           // isCO1Shared
    (64 * 1024),    // sharedCO1BufferSize
    true,           // doNorm
    64,             // singleM
    128,            // singleN
    256,            // singleK
    64,             // baseM
    128,            // baseN
    256,            // baseK
    false           // isA2B2Shared, L0ab double buffer
);

// IFA PA mm2 D = 256 s2 = 128
constexpr MatmulConfig CFG_SAMEB_S1_64_S2_128_D256 = GetPFACustomConfig(
    true,           // isCO1Shared
    (64 * 1024),    // sharedCO1BufferSize
    true,           // doNorm
    64,             // singleM
    256,            // singleN
    128,            // singleK
    64,             // baseM
    256,            // baseN
    128,            // baseK
    false           // isA2B2Shared, L0ab double buffer
);

// IFA PA mm1 D = 512 s2 = 128 splitD
constexpr MatmulConfig CFG_SAMEAB_S1_64_S2_128_D512 = GetPFACustomConfig(
    true,           // isCO1Shared
    (64 * 1024),    // sharedCO1BufferSize
    true,           // doNorm
    64,             // singleM
    128,            // singleN
    512,            // singleK
    64,             // baseM
    128,            // baseN
    256,            // baseK
    false           // isA2B2Shared, L0ab double buffer
);

// IFA PA mm2 D = 512 s2 = 128 splitD
constexpr MatmulConfig CFG_SAMEB_S1_64_S2_128_D512 = GetPFACustomConfig(
    true,           // isCO1Shared
    (64 * 1024),    // sharedCO1BufferSize
    true,           // doNorm
    64,             // singleM
    512,            // singleN
    128,            // singleK
    64,             // baseM
    256,            // baseN
    128,            // baseK
    false           // isA2B2Shared, L0ab double buffer
);

constexpr MatmulConfig CFG_SAMEAB_S1_128_S2_128_D256 = GetPFACustomConfig(
    true,  // isCO1Shared
    (64 * 1024), // sharedCO1BufferSize
    true,  // isConcat
    128,   // singleM
    128,   // singleN
    256,   // singleK
    128,   // baseM
    128,   // baseN
    256,   // baseK
    false  // isA2B2Shared, disable L0ab double buffer
);

constexpr MatmulConfig CFG_SAMEB_S1_128_S2_128_D128_PFAMLA = GetPFACustomConfig(
    true,  // isCO1Shared
    (64 * 1024),  // sharedCO1BufferSize
    true,  // doNorm
    128,   // singleM
    128,   // singleN
    128,   // singleK
    128,   // baseM
    128,   // baseN
    128,   // baseK
    false   // isA2B2Shared, disable L0ab double buffer
);

constexpr MatmulConfig CFG_SAMEAB_S1_64_S2_64_D256 = GetPFACustomConfig(
    true,  // isCO1Shared
    (64 * 1024), // sharedCO1BufferSize
    true,  // isConcat
    64,    // singleM
    64,    // singleN
    256,   // singleK
    64,    // baseM
    64,    // baseN
    256,   // baseK
    false  // isA2B2Shared, disable L0ab double buffer
);

constexpr MatmulConfig CFG_SAMEB_S1_64_S2_64_D256 = GetPFACustomConfig(
    true,  // isCO1Shared
    (64 * 1024), // sharedCO1BufferSize
    true,  // isConcat
    64,    // singleM
    256,   // singleN
    64,    // singleK
    64,    // baseM
    256,   // baseN
    64,    // baseK
    false  // isA2B2Shared, disable L0ab double buffer
);

constexpr MatmulConfig CFG_SAMEAB_S1_64_S2_64_D512 = GetPFACustomConfig(
    true,  // isCO1Shared
    (64 * 1024), // sharedCO1BufferSize
    true,  // isConcat
    64,    // singleM
    64,   // singleN
    512,   // singleK
    64,    // baseM
    64,   // baseN
    256,   // baseK
    false  // isA2B2Shared, disable L0ab double buffer
);

constexpr MatmulConfig CFG_SAMEB_S1_64_S2_64_D512 = GetPFACustomConfig(
    true,  // isCO1Shared
    (64 * 1024), // sharedCO1BufferSize
    true,  // isConcat
    64,    // singleM
    512,   // singleN
    64,   // singleK
    64,    // baseM
    256,   // baseN
    64,   // baseK
    false  // isA2B2Shared, disable L0ab double buffer
);

constexpr MatmulConfig CFG_SAMEAB_G_64_S2_128_D576 = GetPFACustomConfig(
    true,  // isCO1Shared
    (64 * 1024), // sharedCO1BufferSize
    true,  // isConcat
    64,    // singleM
    128,   // singleN
    576,   // singleK
    64,    // baseM
    128,   // baseN
    256,   // baseK
    false  // isA2B2Shared, disable L0ab double buffer
);

constexpr MatmulConfig CFG_SAMEB_G_64_S2_128_D512 = GetPFACustomConfig(
    true,  // isCO1Shared
    (64 * 1024), // sharedCO1BufferSize
    true,  // isConcat
    64,    // singleM
    512,   // singleN
    128,   // singleK
    64,    // baseM
    256,   // baseN
    128,   // baseK
    false  // isA2B2Shared, disable L0ab double buffer
);

struct ConstParam {
    int64_t tmpBlockIdx;
    uint32_t subBlockIdx; // 单个numBlocks中vectorCore编号: 0或者1

    int32_t sIdStart;        // 分核后，单个核batch的开始idx
    int32_t sIdEnd;          // 分核后，单个核batch的结束idx
    int32_t nLoopStart;      // 分核后，单个核N的开始idx
    int32_t nLoopEnd;        // 分核后，单个核N的结束idx
    int32_t outerLoopStart;  // 分核后，单个核S1方向的开始idx
    int32_t outerLoopEnd;    // 分核后，单个核S1方向的结束idx

    int64_t preTokens;       // 用户输入的preToken
    int64_t nextTokens;      // 用户输入的nextTokens

    bool isRowInvalid;       // 是否使能行无效

    uint32_t actualSeqLenSize;    // 用户输入的actualseq的长度
    uint32_t actualSeqLenKVSize;  // 用户输入的actualseq_kv的长度
    bool isActualLenDimsNull;     // 判断是否有actualseq
    bool isActualLenDimsKVNull;   // 判断是否有actualseq_kv
    uint32_t isKvContinuous;      // 是否为tensorlist

    uint32_t multiHeadQ;          // n * qkD
    uint32_t multiHeadK;          // n * qkD / headNumRatio
    uint32_t multiHeadV;          // n * vD / headNumRatio
    uint32_t multiHeadQRope;      // n * ropeD
    uint32_t multiHeadKRope;      // n * ropeD / headNumRatio
    uint32_t multiHeadOut;        // n * vD

    int32_t attentionMaskType;    // sparse 类型
    uint32_t attenMaskBatch;      // attenMask 是否为多 batch
    uint32_t maskQsSize;          // mask S1
    uint32_t maskTypeByteNum;     // mask 数据类型的ByteNum

    uint32_t pseShiftBatch;       // PSE 是否为多 batch
    uint32_t pseShiftS1Size;      // PSE S1
    uint32_t pseShiftS2Size;      // PSE S2
    uint32_t pseShiftTypeByteNum; // PSE 数据类型的ByteNum
    uint32_t pseShiftStride;      // PSE Stride 

    // service mm1
    uint32_t bmm1TilingDataRectM;
    uint32_t bmm1TilingDataRectN;
    uint32_t bmm1TilingDataRectKa;
    uint32_t bmm1TilingDataRectKb;

    // service mm2
    uint32_t bmm2TilingDataRectN;
    uint32_t bmm2TilingDataRectKb;

    // service mm1 mm2 pageAttention
    uint32_t blockTableDim2;
    uint32_t blockSize;
    uint32_t paLayoutType;
    uint32_t paBlockNumSum;

    // service vector1
    float scaleValue;
    uint32_t typeByteNum;         // BYTE_BLOCK(32) / datatypesize
    uint32_t softmaxFlashTilingDataSrcM;
    uint32_t softmaxFlashTilingDataSrcK;
    uint32_t softmaxFlashTilingDataSrcSize;

    // service vector2
    uint32_t softmaxTypeByteNum;  // BYTE_BLOCK(32) / softmaxDataTypeSize;
    uint32_t outputTypeByteNum;   // BYTE_BLOCK(32) / outputDataTypeSize;
    uint32_t isBSNDOut;           // LAYOUT是否为BNSD_BSND

    // B N1 N2 G S1 S2长度
    int32_t sNum;                 // B，后续更名为bSize
    uint32_t headNumSize;         // N1，Q N size，后续更名为n1Size
    uint32_t kvHeadNumSize;       // N2，K V N size，后续更名为n2Size
    uint32_t n2Size;              // 新增：N2，KV N size，N1 = N2 * G，同时也表示将Q的N拆分为N2组，每组有G个S*D。MHA场景下N1=N2，G=1
    uint32_t headNumRatio;        // G，GQA场景，每组下面有多少个S*D，MHA场景下为1，后续更名为gSize
    uint32_t gOfMla;              // 在mla场景下headNumRatio为1，gOfMla为真实g
    int64_t seqSize;              // S1，Q sequence长度，后续更名为s1Size
    int64_t seqInnerSize;         // S2，KV sequence长度，后续更名为s2Size
    uint32_t headSize;            // D，D size，后续更名为dSize
    uint32_t ropeHeadSize;        // ropeDsize
    uint32_t qkHeadSize;          // qkDsize
    uint32_t vHeadSize;           // vDsize

    // q k v attenMask不同轴的stride
    int64_t qBStride;             // 新增：Q B轴的stride，BNSD格式下为N1*S1*D，BSH格式下为S1*N1*D
    int64_t qN2Stride;            // 新增：Q N2轴的stride，GQA场景下表示组与组之间的偏移，BNSD格式下为G*S1*D，BSH格式下为G*D
    int64_t qGStride;             // 新增：Q G轴的stride，BNSD格式下为S1*D，BSH格式下为D
    int32_t qS1Stride;            // 新增：Q S1轴的stride，BNSD格式下为D，BSH格式下为N1*D
    int64_t kvBStride;            // 新增：KV B轴的stride，BNSD格式下为N2*S2*D，BSH格式下为S2*N2*D
    int64_t kvN2Stride;           // 新增：KV N2轴的stride，BNSD格式下为S2*D，BSH格式下为D
    int32_t kvS2Stirde;           // 新增：KV S2轴的stride，BNSD格式下为D，BSH格式下为N2*D
    int64_t maskBStride;          // 新增：Mask B轴的stride，[B,N1,S1,S2]格式下为N1*S1*S2，[B,1,S1,S2]格式下为S1*S2，[S1,S2]格式下为0
    int64_t maskN2Stride;         // 新增：Mask N2轴的stride，当前不支持N2维度的偏移，默认为0
    int64_t maskGStride;          // 新增：Mask G轴的stride，当前不支持G维度的偏移，默认为0
    uint32_t attentionMaskStride; // mask S2  maskKVsSize，表示Mask S1轴的stride，后续更名为maskS1Stride
    uint64_t rStride;             // ropeD or n * ropeD

    // 基本块切分
    uint32_t singleProcessSInnerSize;           // sinner (128)，后续更名为s2SplitSize
    uint32_t singleProcessSOuterSizeWhole;      // souter（64），后续更名为s1SplitSize
    uint32_t singleProcessCubeSOuterSizeWhole;  // souter（128） cube分核。后续更名为???
    uint32_t gSplitNum;                         // 新增：G轴切块的个数
    uint32_t gSplitSize;                        // 新增：G轴切分单块的大小

    // 左padding
    bool isQHasLeftPadding;
    bool isKVHasLeftPadding;
    int64_t queryRightPaddingSize;
    int64_t kvRightPaddingSize;

    // 伪量化相关
    int32_t antiKVRowStepSize;                  // 新增：伪量化，KV反量化时，单次循环搬入行数。例如buffer size=16KB，antiKVRowStepSize=16K/sizeof(KV_T)/D
    bool isAntiquantSymmetric;

    // lse输出
    bool isSoftmaxLseEnable;
    int64_t totalSoftmaxLseOutputSize;

    // IFA PA
    bool isIFA;
};

struct RunParam {  // 分核与切块需要使用到参数
    uint32_t actualSeqLengthsIdx = 0;

    // N循环生产的数据
    bool isLast = false;       // 用于判断是否为最后一个N
    int64_t batchNOffset;      // loopNIdx
    int32_t tmpNLoopEnd;       // batch方向的循环控制信息 结束idx

    // NB循环生产的数据
    int64_t s2InCurrentBatch;                 // Tensorlist场景，不同batch的S2长度，后续用计算KvStride
    int64_t actualSeqLengthPerBatch = 0;      // Q的actualSeqLength
    int64_t actualSeqLengthOfMlaPerBatch = 0; // 在mla场景下Q的actualSeqLength
    int64_t actualSeqLengthKVPerBatch = 0;    // KV的actualSeqLength
    uint32_t singleProcessSOuterSizeTail;     // souter的尾块，B循环时更新
    uint32_t singleProcessCubeSOuterSizeTail; // cube分核时，cube的souter的尾块，B循环时更新 
    uint32_t unalignSInner;               // s2方向尾块长度，非对齐  B循环时更新
    uint32_t singleProcessSInnerSizeTail; // s2方向尾块长度，按64对齐
    // 优化 uint32_t maxInnerLoopTimes;           // S2的切块个数，用于判断是否为尾块
    uint32_t maskInnerTailAlign;          // s2方向尾块长度，按mask数据类型对齐(maskTypeByteNum)
    uint32_t padSize;                     // maskInnerTailAlign - unalignSInner
    uint32_t pseShiftInnerTailAlign;      // s2方向尾块长度，按pse数据类型对齐(pseShiftTypeByteNum)
    uint32_t pseShiftPadSize;             // pseShiftInnerTailAlign - unalignSInner
    int64_t preTokensPerBatch = 0;         // 左上顶点的pretoken
    int64_t nextTokensPerBatch = 0;        // 左上顶点的nexttoken
    int64_t nextTokensOfMlaPerBatch = 0;   // 在mla场景下左上顶点的nexttoken，用于计算BNSD的行无效
    // 优化 int64_t multiSeqOffset;               // sIdx * seqSize * multiHeadQ
    // 优化 int32_t sOuterBlockNum;               // 不同batch souter的块数
    int32_t tmpOuterLoopEnd;              // S1方向的循环控制信息 结束idx
    int32_t taskBatch;                    // sIdx PageAttention时mm需要使用

    // NBS1循环生产的数据
    int64_t sOuterOffset;               // 单个S内 souter的 souterIdx * singleProcessSOuterSize
    int64_t cubeSOuterOffset;           // 单个S内 souter的 souterIdx * singleProcessSOuterSize
    int64_t keyCoreOffset;              // Souter方向上，不同souter的Key的offset
    int64_t kRopeNBGOffset;             // G方向上，不同g的KeyRope的offset
    int64_t valueCoreOffset;            // Souter方向上，不同souter的value的offset
    uint64_t attenMaskCoreOffset;       // Souter方向上，不同souter的attenMask的offset
    uint64_t pseShiftCoreOffset;        // Souter方向上，不同souter的pseShift的offset
    int64_t tensorAOffset;              // Query 的 offset
    int64_t qRopeNBGOffset;             // QueryRope 的 offset
    int64_t attentionOutOffset;         // attentionOut的 offset
    uint32_t singleProcessSOuterSize;   // souter大小，区分尾块和非尾块
    uint32_t cubeSOuterSize;            // cube视角的sOuter，在SAMEAB场景中cubeSOuterSize为两倍的 singleProcessSOuterSize

    int32_t startIndex;            // S2方向的循环控制信息 
    int32_t endIndex;              // S2方向的循环控制信息

    // q k v attenMask不同轴的offset
    // B轴offset 
    int64_t multiSeqOffset;             // bIdx * seqSize * multiHeadQ，后续更名为qBOffset
    int64_t qRopeBOffset;
    int64_t kvBOffset;                  // 新增：KV B=bIdx时，B轴的偏移。bIdx * kvBStride
    int64_t maskBOffset;                // 新增：Mask B=bIdx时，B轴的偏移。bIdx * maskBStride
    int64_t attenOutBOffset;            // 新增：attenOut B=bIdx时，B轴的偏移。当前等于qBOffset

    // Batch轴确定后，可以确定的信息
    int32_t s1FirstToken;               // 新增：S1方向循环的首Token（需要考虑actualSeqQ、actualSeqKV、preToken、nextToken）
    int32_t s1LastToken;                // 新增：S1方向循环的尾Token（需要考虑actualSeqQ、actualSeqKV、preToken、nextToken）
    int32_t sOuterBlockNum;             // S1方向的切块个数，后续更名为s1SplitNum
    int64_t s2FirstToken;               // S2方向循环的首Token
    int64_t s2LastToken;                // S2方向循环的尾Token
    uint32_t maxInnerLoopTimes;         // S2方向的切块个数，用于判断是否为尾块，后续更名为s2SplitNum

    // N2轴offset
    int64_t qN2Offset;                  // 新增：Q N2=n2Idx时，N2轴的偏移。n2Idx * qN2Stride
    int64_t kvN2Offset;                 // 新增：KV N2=n2Idx时，N2轴的偏移。n2Idx * kvN2Stride
    int64_t maskN2Offset;               // 新增：Mask N2=n2Idx时，N2轴的偏移。n2Idx * maskN2Stride
    int64_t attenOutN2Offset;           // 新增：attenOut N2=n2Idx时，N2轴的偏移。当前等于qN2Offset

    // G轴offset
    int64_t qGOffset;                   // 新增：Q G轴第gIdx块，G轴的偏移。gIdx * gSplitSize * qGStride
    int64_t attenMaskGOffset;           // 新增：Mask G轴第gIdx块，G轴的偏移。gIdx * gSplitSize * maskGStride
    int64_t attenOutGOffset;            // 新增：attenOut G轴第gIdx块，G轴的偏移。当前等于qGOffset

    // S1轴offset
    int64_t qS1Offset;                  // 新增：Q S1轴第s1Idx块，S1轴的偏移。s1Idx * s1SplitSize * qS1Stride
    int64_t maskS1Offset;               // 新增：Mask S1轴第s1Idx块，S1轴的偏移。s1Idx * s1SplitSize * maskS1Stride
    int64_t attenOutS1Offset;           // 新增：attenOut S1轴第s1Idx块，S1轴的偏移。当前等于qS1Offset

    // S2轴offset
    int64_t kvSInnerOffset;             // 新增：KV S2轴第s2Idx块，S2轴的偏移。s2Idx * s2SplitSize * kvS2Stride
    int64_t attenMaskSInnerOffset;      // 新增：Mask S2轴第s2Idx块，S2轴的偏移。s2Idx * s2SplitSize * maskS2Stride

    // 左padding
    int64_t queryLeftPaddingSize;
    int64_t kvLeftPaddingSize;

    // lse 输出offset
    int64_t softmaxLseOffset;
};

struct TaskParam {
    bool isValid = false;                         // 新增：当前索引的任务是不是有效的

    // 这4个变量，后续可以改成枚举，变成1个变量
    bool isInnerTail;              // 是否为s2方向的尾块
    bool isFirstInnerIter;         // s2方向的控制信息，是否为首轮
    bool isSecondInnerIter;        // s2方向的控制信息，是否为第二轮
    bool isLastInnerIter;          // s2方向的控制信息，是否为最后一轮，并非尾块 sparse

    bool splitPingPong;          // 用于两轮sinner之间的ping pong
    bool taskPingPong;           // 用于sinner内每个task之间的ping pong

    int64_t sInnerOffsetDataSize;         // mm1 PageAttention 记录s2轴的偏移
    int32_t taskBatch;                    // mm1 PageAttention sIdx
    int64_t batchNOffset;                 // mm1 PageAttention nIdx
    int64_t tensorAOffset;                // mm1 拷贝自RunParam中的tensorAOffset，后续更名为qFinalOffset
    int64_t tensorBOffset;                // mm1 Key 的 offset，后续更名为kFinalOffset
    int64_t qRopeOffset;
    int64_t kRopeOffset;

    uint32_t cubeSOuterSize;              // mm2 cube视角的sOuter 区分尾块
    uint32_t singleProcessSOuterSize;     // mm2 vector1 vector2 souter大小，区分尾块和非尾块
    uint32_t mm1SingleCoreN;              // mm2 s2方向非尾块时等于singleProcessSInnerSize 区分尾块  // 重复 考虑删除 singleProcessSInnerSizeNow
    uint32_t singleProcessSInnerBmmTail;  // mm2 vector1 s2方向非尾块时等于singleProcessSInnerSize，尾块时等于unalignSInner
    int64_t valueOffset;                  // mm2 Value 的 offset  复用tensorBOffset，后续更名为vFinalOffset

    uint32_t singleProcessSInnerSizeNow;  // vector1 s2方向切块大小 区分尾块

    uint32_t maskCopyInCol;               // vector1 s2方向非尾块时等于singleProcessSInnerSize，尾块时等于maskInnerTailAlign
    uint32_t padSize;                     // vector1 maskInnerTailAlign - unalignSInner
    uint64_t attenMaskOffset;             // vector1 mask的 offset，后续更名为???
    uint64_t attenMaskOffsetPre;          // vector1 band场景 第二个mask的 offset，后续更名为???

    uint32_t pseShiftCopyInCol;           // vector1 s2方向非尾块时等于singleProcessSInnerSize，尾块时等于pseShiftInnerTailAlign
    uint32_t pseShiftPadSize;             // vector1 pseShiftInnerTailAlign - unalignSInner
    uint64_t pseShiftOffset;              // vector1 pse 的 offset

    int64_t preTokensPerBatch;            // vector2 左上顶点的pretoken
    int64_t nextTokensPerBatch;           // vector2 左上顶点的nexttoken
    int64_t nextTokensOfMlaPerBatch;      // vector2 在mla场景下左上顶点的nexttoken，用于计算BNSD的行无效
    int64_t actualSeqLengthPerBatch;      // vector2 Q的actualSeqLength
    int64_t actualSeqLengthOfMlaPerBatch; // vector2 在mla场景下Q的actualSeqLength
    int64_t actualSeqLengthKVPerBatch;    // vector2 KV的actualSeqLength
    int64_t sOuterOffset;                 // vector2 单个S内 souter的 souterIdx * singleProcessSOuterSize
    int64_t attentionOutOffset;           // vector2 attenOut最终的offset，当前等于qFinalOffset  后续更名为attenOutOffset

    // 伪量化相关
    int32_t antiKVRowLoopTimes;           // 新增：antiquantKV循环搬入次数
    int32_t antiKVRowTailSize;            // 新增：antiquantKV尾块搬入行数

    // lse 输出offset
    int64_t softmaxLseOffset;
};

template<uint32_t TASK_CACHE>
class TaskManager {
    public:
        __aicore__ inline TaskParam& GetTaskRef(uint64_t taskIdx) {
            return taskParamArr[taskIdx % static_cast<uint64_t>(TASK_CACHE)];
        };

    private:
        TaskParam taskParamArr[TASK_CACHE];
};

enum PFAMask {          // PFATODO 添加class
    DISABLE_MASK = 0,
    ENABLE_MASK_NO_BAND = 1,
    ENABLE_MASK_BAND = 2,
};

enum PFAPse {          // PFATODO 添加class
    DISABLE_PSE = 0,
    ENABLE_PSE = 1,
};

enum RunMode {
    HighPrecision,
    HighPerformance
};

enum SplitCoreMode {
    SPLIT_NBS_VECTOR = 0,
    SPLIT_NBS_CUBE,
    BALANCE_VECTOR,
    BALANCE_CUBE,
};

enum class PFALayout {
    BSH = 0,
    BNSD,
    TND
};

enum class PFAMatMulType {
    MM_PFA = 0,
    MM_PA,
    MM_IFA_MLA,
    MM_IFA_MLA_PA,
    MM_PA_D512,
    MM_DN,
};

enum class PFADTemplateType {
    PFA_PA_BMM1 = 0,
    PFA_PA_BMM2
};

template<typename T, RunMode M = RunMode::HighPerformance>
struct PromptFlashAttentionTypeTraits
{
    using mmInputType = T;
    using mmBiasType = T;
    using mmOutputType = T;
    using softmaxType = T;
    using pseShiftType = T;
    using pseShiftCastType = half;
};

template<>
struct PromptFlashAttentionTypeTraits<half, RunMode::HighPerformance>
{
    using mmInputType = half;
    using mmBiasType = float;
    using mmOutputType = half;
    using softmaxType = half;
    using pseShiftType = half;
    using pseShiftCastType = half;
};

#if (__CCE_AICORE__ > 200)

template<>
struct PromptFlashAttentionTypeTraits<half, RunMode::HighPrecision>
{
    using mmInputType = half;
    using mmBiasType = float;
    using mmOutputType = float;
    using softmaxType = float;
    using pseShiftType = half;
    using pseShiftCastType = float;  // pseShiftCastType只有在高精度和bf16的情况下为fp32
};

template<>
struct PromptFlashAttentionTypeTraits<bfloat16_t>
{
    using mmInputType = bfloat16_t;
    using mmBiasType = float;
    using mmOutputType = float;
    using softmaxType = float;
    using pseShiftType = bfloat16_t;
    using pseShiftCastType = float;  // pseShiftCastType只有在高精度和bf16的情况下为fp32
};

template<>
struct PromptFlashAttentionTypeTraits<bfloat16_t, RunMode::HighPrecision>
{
    using mmInputType = bfloat16_t;
    using mmBiasType = float;
    using mmOutputType = float;
    using softmaxType = float;
    using pseShiftType = bfloat16_t;
    using pseShiftCastType = float;  // pseShiftCastType只有在高精度和bf16的情况下为fp32
};

template<>
struct PromptFlashAttentionTypeTraits<int8_t, RunMode::HighPrecision>
{
    using mmInputType = int8_t;
    using mmBiasType = int32_t;
    using mmOutputType = int32_t;
    using softmaxType = float;
    using pseShiftType = half;
    using pseShiftCastType = float;  // pseShiftCastType只有在高精度和bf16的情况下为fp32
};

template<>
struct PromptFlashAttentionTypeTraits<fp8_e4m3fn_t, RunMode::HighPrecision>
{
    using mmInputType = fp8_e4m3fn_t;
    using mmBiasType = float;
    using mmOutputType = float;
    using softmaxType = float;
    using pseShiftType = half;
    using pseShiftCastType = float;  // pseShiftCastType只有在高精度和bf16的情况下为fp32
};

template<>
struct PromptFlashAttentionTypeTraits<hifloat8_t, RunMode::HighPrecision>
{
    using mmInputType = hifloat8_t;
    using mmBiasType = float;
    using mmOutputType = float;
    using softmaxType = float;
    using pseShiftType = half;
    using pseShiftCastType = float;  // pseShiftCastType只有在高精度和bf16的情况下为fp32
};
#endif

template<>
struct PromptFlashAttentionTypeTraits<int8_t>
{
    using mmInputType = int8_t;
    using mmBiasType = int32_t;
    using mmOutputType = half;
    using softmaxType = half;
    using pseShiftType = half;
    using pseShiftCastType = half;
};

// PFATODO 这部分定义是不是放到matmul_modules文件夹里面？
// 这一部分训练是单独放了个文件夹 我们可以参考一下 暂时可以放这里
template<PFAMatMulType MM_TYPE>
struct ConstPolicySelector {
    template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : AscendC::Impl::Detail::MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>{};
};

template<PFAMatMulType MM_TYPE, bool isSInner256>
struct PFABmm2ConstPolicySelector {
    template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : AscendC::Impl::Detail::MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>{};
};

template<>
struct ConstPolicySelector<PFAMatMulType::MM_PFA> {
    template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : AscendC::Impl::Detail::PFANormBmm1Policy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>{};
};

template<>
struct PFABmm2ConstPolicySelector<PFAMatMulType::MM_PFA, false> {
    template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : AscendC::Impl::Detail::PFANormBmm2Policy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>{};
};

template<>
struct ConstPolicySelector<PFAMatMulType::MM_IFA_MLA> {
    template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : AscendC::Impl::Detail::IFAMLABmm1Policy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>{};
};

template<>
struct PFABmm2ConstPolicySelector<PFAMatMulType::MM_IFA_MLA, false> {
    template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : AscendC::Impl::Detail::IFAMLABmm2Policy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>{};
};

template<>
struct ConstPolicySelector<PFAMatMulType::MM_PA> {
    template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : AscendC::Impl::Detail::FAPaBmm1Policy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>{};
};

template<>
struct PFABmm2ConstPolicySelector<PFAMatMulType::MM_PA, false> {
    template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : AscendC::Impl::Detail::FAPaBmm2Policy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>{};
};

template<>
struct ConstPolicySelector<PFAMatMulType::MM_PA_D512> {
    template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : AscendC::Impl::Detail::FAPaBmm1Policy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>{};
};

template<>
struct PFABmm2ConstPolicySelector<PFAMatMulType::MM_PA_D512, false> {
    template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : AscendC::Impl::Detail::FAPaBmm2Policy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>{};
};

template<>
struct ConstPolicySelector<PFAMatMulType::MM_IFA_MLA_PA> {
    template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : AscendC::Impl::Detail::IFAMLAPaBmm1Policy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>{};
};

template<>
struct PFABmm2ConstPolicySelector<PFAMatMulType::MM_IFA_MLA_PA, false> {
    template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : AscendC::Impl::Detail::IFAMLAPaBmm2Policy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>{};
};

template<>
struct ConstPolicySelector<PFAMatMulType::MM_DN> {
    template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : AscendC::Impl::Detail::PFABmm1PolicyDN<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>{};
};

template<>
struct PFABmm2ConstPolicySelector<PFAMatMulType::MM_DN, true> {
    template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : AscendC::Impl::Detail::PFABmm2PolicyDN<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>{};
};

template<>
struct PFABmm2ConstPolicySelector<PFAMatMulType::MM_DN, false> {
    template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : AscendC::Impl::Detail::PFANormBmm2Policy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>{};
};

template <PFALayout L, typename T, typename U, typename O = T, typename KV_T = T,
          PFAMask Mask = PFAMask::DISABLE_MASK,
          PFAPse Pse = PFAPse::DISABLE_PSE,
          RunMode M = RunMode::HighPerformance,
          SplitCoreMode SCM = SplitCoreMode::SPLIT_NBS_VECTOR,
          uint32_t SOUTER = NO_CONSTANT,
          uint32_t SINNER = NO_CONSTANT,
          uint32_t QKDSIZE = NO_CONSTANT,
          uint32_t VDSIZE = QKDSIZE,
          const PFAMatMulType MM_TYPE_TMP = PFAMatMulType::MM_PFA,
          typename...Args>
struct PFAType {
    __aicore__ const static constexpr bool IsSplitCoreByCube() {
        if constexpr (SCM == SplitCoreMode::SPLIT_NBS_CUBE || SCM == SplitCoreMode::BALANCE_CUBE) {
            return true;
        } else {
            return false;
        }
    }

    __aicore__ const static constexpr bool IsUseNormMatmul() {
        // 当非Cube分核且非PA场景时，使用原生matmul，不使用matmulpolicy
        if constexpr (!IsSplitCoreByCube() && MM_TYPE == PFAMatMulType::MM_PFA) {
            return true;
        } else {
            return false;
        }
    }

    __aicore__ const static constexpr MatmulConfig &GetMM1Config() {
        if constexpr (!IsSplitCoreByCube()) {
            return CFG_PFA_NORM;
        }
        if constexpr (QKDSIZE == DSIZE_CONST_64 && SINNER == SINNER_CONST_128) {
            return CFG_SAMEAB_S1_128_S2_128_D64;
        } else if constexpr (QKDSIZE == DSIZE_CONST_64 && SOUTER == SOUTER_CONST_128 && SINNER == SINNER_CONST_256) {
            return CFG_SAMEAB_S1_128_S2_256_D64;
        } else if constexpr (QKDSIZE == DSIZE_CONST_64 && SOUTER == SOUTER_CONST_64 && SINNER == SINNER_CONST_256) {
            return CFG_SAMEAB_S1_64_S2_256_D64;
        } else if constexpr (QKDSIZE == DSIZE_CONST_128 && SOUTER == SOUTER_CONST_64 && SINNER == SINNER_CONST_256) {
            return CFG_SAMEAB_S1_64_S2_256_D128;
        } else if constexpr (QKDSIZE == DSIZE_CONST_128) {
            return CFG_SAMEAB_S1_128_S2_128_D128;
        } else if constexpr ((QKDSIZE != VDSIZE) && QKDSIZE == DSIZE_CONST_256) { // PFA MLA QKD=192,VD=128
            return CFG_SAMEAB_S1_128_S2_128_D256;
        } else if constexpr ((QKDSIZE == VDSIZE) && QKDSIZE == DSIZE_CONST_256 && SINNER == SINNER_CONST_64) { // PFA D=256,s2=64
            return CFG_SAMEAB_S1_64_S2_64_D256;
        } else if constexpr ((QKDSIZE == VDSIZE) && QKDSIZE == DSIZE_CONST_256 && SINNER == SINNER_CONST_128) { // IFA D=256,s2=128
            return CFG_SAMEAB_S1_64_S2_128_D256;
        } else if constexpr (QKDSIZE == DSIZE_CONST_512 && SINNER == SINNER_CONST_64) { // PFA D=512,s2=64
            return CFG_SAMEAB_S1_64_S2_64_D512;
        } else if constexpr (QKDSIZE == DSIZE_CONST_512 && SINNER == SINNER_CONST_128) { // IFA D=512,s2=128
            return CFG_SAMEAB_S1_64_S2_128_D512;
        } else if constexpr (QKDSIZE == DSIZE_CONST_576) {
            return CFG_SAMEAB_G_64_S2_128_D576;
        } else {
            return CFG_PFA_NORM;
        }
    }

    __aicore__ const static constexpr MatmulConfig &GetMM2Config() {
        if constexpr (!IsSplitCoreByCube()) {
            return CFG_PFA_NORM;
        }
        if constexpr (VDSIZE == DSIZE_CONST_64 && SINNER == SINNER_CONST_128) {
            return CFG_SAMEB_S1_128_S2_128_D64;
        } else if constexpr (VDSIZE == DSIZE_CONST_64 && SOUTER == SOUTER_CONST_128 && SINNER == SINNER_CONST_256) {
            return CFG_SAMEB_S1_128_S2_256_D64;
        } else if constexpr (VDSIZE == DSIZE_CONST_64 && SOUTER == SOUTER_CONST_64 && SINNER == SINNER_CONST_256) {
            return CFG_SAMEB_S1_64_S2_256_D64;
        } else if constexpr (VDSIZE == DSIZE_CONST_128 && SOUTER == SOUTER_CONST_64 && SINNER == SINNER_CONST_256) {
            return CFG_SAMEB_S1_64_S2_256_D128;
        } else if constexpr ((QKDSIZE == VDSIZE) && (VDSIZE == DSIZE_CONST_128)) {
            return CFG_SAMEB_S1_128_S2_128_D128;
        } else if constexpr ((QKDSIZE != VDSIZE) && (VDSIZE == DSIZE_CONST_128)) {
            return CFG_SAMEB_S1_128_S2_128_D128_PFAMLA;
        } else if constexpr ((VDSIZE == DSIZE_CONST_256) && (SINNER == SINNER_CONST_64)) { // PFA VD=256,S2=64
            return CFG_SAMEB_S1_64_S2_64_D256;
        }  else if constexpr ((VDSIZE == DSIZE_CONST_256) && (SINNER == SINNER_CONST_128)) { // IFA VD=256,S2=128
            return CFG_SAMEB_S1_64_S2_128_D256;
        } else if constexpr ((QKDSIZE == VDSIZE) && (VDSIZE == DSIZE_CONST_512) && (SINNER == SINNER_CONST_64)) { // PFA VD=512,S2=64
            return CFG_SAMEB_S1_64_S2_64_D512;
        } else if constexpr ((QKDSIZE == VDSIZE) && (VDSIZE == DSIZE_CONST_512) && (SINNER == SINNER_CONST_128)) { // IFA VD=512,S2=128
            return CFG_SAMEB_S1_64_S2_128_D512;
        } else if constexpr ((QKDSIZE != VDSIZE) && (VDSIZE == DSIZE_CONST_512)) {
            return CFG_SAMEB_G_64_S2_128_D512;
        } else {
            return CFG_PFA_NORM;
        }
    }

    __aicore__ const static constexpr bool IsBmm2Concat() {
        if constexpr (!IsSplitCoreByCube()) {
            return false;
        }
        return true;
    }

    using inputType = T;
    using maskType = U;
    using outputType = O;
    using kvInputType = KV_T;
    static constexpr PFALayout layout = L;
    static constexpr RunMode calcMode = M;
    static constexpr PFAMatMulType MM_TYPE = MM_TYPE_TMP;
    static constexpr bool isSplitCoreByCube = IsSplitCoreByCube();
    static constexpr bool isBmm2Concat = IsBmm2Concat();
    static constexpr bool isUseNormMatmul = IsUseNormMatmul();
    static constexpr bool isHasAtten = (Mask == PFAMask::ENABLE_MASK_NO_BAND ||
                                        Mask == PFAMask::ENABLE_MASK_BAND);
    static constexpr bool isBand = (Mask == PFAMask::ENABLE_MASK_BAND);
    static constexpr bool isHasPse = (Pse == PFAPse::ENABLE_PSE);
    static constexpr uint32_t sOuter = SOUTER;

    static constexpr uint32_t vsOuter = SOUTER / 2;  // vector视角的sOuter

    static constexpr uint32_t sInner = SINNER;
    static constexpr uint32_t qkDSize = QKDSIZE;
    static constexpr uint32_t vDSize = VDSIZE;
    static constexpr bool IFA_MLA = (QKDSIZE != VDSIZE) && (MM_TYPE_TMP == PFAMatMulType::MM_IFA_MLA ||
        MM_TYPE_TMP == PFAMatMulType::MM_IFA_MLA_PA);
    static constexpr bool PFA_MLA = (QKDSIZE != VDSIZE) && (MM_TYPE_TMP != PFAMatMulType::MM_IFA_MLA &&
        MM_TYPE_TMP != PFAMatMulType::MM_IFA_MLA_PA);
    static constexpr bool useDN = (MM_TYPE_TMP == PFAMatMulType::MM_DN);
};

#endif  // PROMPT_FLASH_ATTENTION_COMM_H