/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "utils/aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#undef private
#undef protected
#include "Eigen/Core"

using namespace std;
using namespace aicpu;

class TEST_INVGRAD_UT : public testing::Test {};

const Eigen::half input1_fp16_data[] = {
    static_cast<Eigen::half>(6.266f),   static_cast<Eigen::half>(-1.418f),   static_cast<Eigen::half>(3.463f),
    static_cast<Eigen::half>(-5.395f),  static_cast<Eigen::half>(-2.828f),   static_cast<Eigen::half>(-5.523f),
    static_cast<Eigen::half>(-7.945f),  static_cast<Eigen::half>(2.383f),    static_cast<Eigen::half>(6.64f),
    static_cast<Eigen::half>(-0.0829f), static_cast<Eigen::half>(3.744f),    static_cast<Eigen::half>(3.486f),
    static_cast<Eigen::half>(-0.727f),  static_cast<Eigen::half>(0.876f),    static_cast<Eigen::half>(9.734f),
    static_cast<Eigen::half>(4.145f),   static_cast<Eigen::half>(8.516f),    static_cast<Eigen::half>(8.51f),
    static_cast<Eigen::half>(9.945f),   static_cast<Eigen::half>(9.7f),      static_cast<Eigen::half>(-0.623f),
    static_cast<Eigen::half>(-7.016f),  static_cast<Eigen::half>(5.684f),    static_cast<Eigen::half>(-0.7114f),
    static_cast<Eigen::half>(8.914f),   static_cast<Eigen::half>(-2.75f),    static_cast<Eigen::half>(0.6665f),
    static_cast<Eigen::half>(0.573f),   static_cast<Eigen::half>(-8.08f),    static_cast<Eigen::half>(-8.68f),
    static_cast<Eigen::half>(-0.739f),  static_cast<Eigen::half>(-7.094f),   static_cast<Eigen::half>(-5.543f),
    static_cast<Eigen::half>(-0.9414f), static_cast<Eigen::half>(8.63f),     static_cast<Eigen::half>(7.344f),
    static_cast<Eigen::half>(5.79f),    static_cast<Eigen::half>(5.836f),    static_cast<Eigen::half>(7.8f),
    static_cast<Eigen::half>(-7.55f),   static_cast<Eigen::half>(-8.04f),    static_cast<Eigen::half>(-5.76f),
    static_cast<Eigen::half>(4.99f),    static_cast<Eigen::half>(2.963f),    static_cast<Eigen::half>(0.0829f),
    static_cast<Eigen::half>(0.0405f),  static_cast<Eigen::half>(-1.688f),   static_cast<Eigen::half>(4.23f),
    static_cast<Eigen::half>(3.766f),   static_cast<Eigen::half>(4.23f),     static_cast<Eigen::half>(6.97f),
    static_cast<Eigen::half>(8.266f),   static_cast<Eigen::half>(-3.887f),   static_cast<Eigen::half>(3.389f),
    static_cast<Eigen::half>(-8.26f),   static_cast<Eigen::half>(-0.02687f), static_cast<Eigen::half>(0.12f),
    static_cast<Eigen::half>(-0.2181f), static_cast<Eigen::half>(2.041f),    static_cast<Eigen::half>(2.635f),
    static_cast<Eigen::half>(5.445f),   static_cast<Eigen::half>(-5.17f),    static_cast<Eigen::half>(-6.246f),
    static_cast<Eigen::half>(1.952f),   static_cast<Eigen::half>(5.43f),     static_cast<Eigen::half>(-1.001f),
    static_cast<Eigen::half>(2.336f),   static_cast<Eigen::half>(3.592f),    static_cast<Eigen::half>(6.754f),
    static_cast<Eigen::half>(6.69f),    static_cast<Eigen::half>(6.96f),     static_cast<Eigen::half>(7.09f),
    static_cast<Eigen::half>(-2.09f),   static_cast<Eigen::half>(5.57f),     static_cast<Eigen::half>(7.19f),
    static_cast<Eigen::half>(5.57f),    static_cast<Eigen::half>(-2.316f),   static_cast<Eigen::half>(-9.42f),
    static_cast<Eigen::half>(-2.96f),   static_cast<Eigen::half>(-0.572f),   static_cast<Eigen::half>(-8.69f),
    static_cast<Eigen::half>(-2.691f),  static_cast<Eigen::half>(1.134f),    static_cast<Eigen::half>(-8.65f),
    static_cast<Eigen::half>(9.625f),   static_cast<Eigen::half>(4.996f),    static_cast<Eigen::half>(-3.996f),
    static_cast<Eigen::half>(-3.154f),  static_cast<Eigen::half>(8.8f),      static_cast<Eigen::half>(-7.51f),
    static_cast<Eigen::half>(3.896f),   static_cast<Eigen::half>(1.448f),    static_cast<Eigen::half>(-2.174f),
    static_cast<Eigen::half>(4.02f),    static_cast<Eigen::half>(-8.29f),    static_cast<Eigen::half>(-0.3406f),
    static_cast<Eigen::half>(9.61f),    static_cast<Eigen::half>(-2.648f),   static_cast<Eigen::half>(-5.465f),
    static_cast<Eigen::half>(-3.209f),  static_cast<Eigen::half>(0.6055f),   static_cast<Eigen::half>(-3.705f),
    static_cast<Eigen::half>(0.04593f), static_cast<Eigen::half>(-2.72f),    static_cast<Eigen::half>(8.36f),
    static_cast<Eigen::half>(-6.85f),   static_cast<Eigen::half>(-2.254f),   static_cast<Eigen::half>(-1.7705f),
    static_cast<Eigen::half>(0.696f),   static_cast<Eigen::half>(1.007f),    static_cast<Eigen::half>(-5.004f),
    static_cast<Eigen::half>(0.6143f),  static_cast<Eigen::half>(-9.195f),   static_cast<Eigen::half>(-6.133f),
    static_cast<Eigen::half>(-9.42f),   static_cast<Eigen::half>(2.055f),    static_cast<Eigen::half>(-7.96f),
    static_cast<Eigen::half>(0.7646f),  static_cast<Eigen::half>(-4.316f),   static_cast<Eigen::half>(-5.633f),
    static_cast<Eigen::half>(-2.945f),  static_cast<Eigen::half>(7.023f),    static_cast<Eigen::half>(-5.926f),
    static_cast<Eigen::half>(0.8994f),  static_cast<Eigen::half>(9.82f),     static_cast<Eigen::half>(-4.746f),
    static_cast<Eigen::half>(1.721f),   static_cast<Eigen::half>(9.9f),      static_cast<Eigen::half>(1.723f),
    static_cast<Eigen::half>(-3.887f),  static_cast<Eigen::half>(-5.88f),    static_cast<Eigen::half>(-8.74f),
    static_cast<Eigen::half>(9.64f),    static_cast<Eigen::half>(-2.754f),   static_cast<Eigen::half>(7.887f),
    static_cast<Eigen::half>(-3.941f),  static_cast<Eigen::half>(-6.883f),   static_cast<Eigen::half>(-0.368f),
    static_cast<Eigen::half>(-5.24f),   static_cast<Eigen::half>(-8.05f),    static_cast<Eigen::half>(-7.887f),
    static_cast<Eigen::half>(-0.505f),  static_cast<Eigen::half>(7.934f),    static_cast<Eigen::half>(-7.46f),
    static_cast<Eigen::half>(6.76f),    static_cast<Eigen::half>(-7.73f),    static_cast<Eigen::half>(-1.055f),
    static_cast<Eigen::half>(4.316f),   static_cast<Eigen::half>(-6.727f),   static_cast<Eigen::half>(6.156f),
    static_cast<Eigen::half>(-9.836f),  static_cast<Eigen::half>(4.613f),    static_cast<Eigen::half>(-2.346f),
    static_cast<Eigen::half>(9.01f),    static_cast<Eigen::half>(-1.775f),   static_cast<Eigen::half>(3.344f),
    static_cast<Eigen::half>(6.81f),    static_cast<Eigen::half>(3.662f),    static_cast<Eigen::half>(-9.06f),
    static_cast<Eigen::half>(6.535f),   static_cast<Eigen::half>(2.352f),    static_cast<Eigen::half>(-8.5f),
    static_cast<Eigen::half>(0.8657f),  static_cast<Eigen::half>(7.89f),     static_cast<Eigen::half>(8.68f),
    static_cast<Eigen::half>(5.684f),   static_cast<Eigen::half>(-7.34f),    static_cast<Eigen::half>(-5.73f),
    static_cast<Eigen::half>(9.44f),    static_cast<Eigen::half>(-1.649f),   static_cast<Eigen::half>(-9.0f),
    static_cast<Eigen::half>(-8.94f),   static_cast<Eigen::half>(-6.293f),   static_cast<Eigen::half>(-0.71f),
    static_cast<Eigen::half>(-3.861f),  static_cast<Eigen::half>(-0.7197f),  static_cast<Eigen::half>(9.09f),
    static_cast<Eigen::half>(-3.11f),   static_cast<Eigen::half>(-0.4783f),  static_cast<Eigen::half>(6.12f),
    static_cast<Eigen::half>(2.318f),   static_cast<Eigen::half>(8.97f),     static_cast<Eigen::half>(-8.305f),
    static_cast<Eigen::half>(2.562f),   static_cast<Eigen::half>(-7.457f),   static_cast<Eigen::half>(5.016f),
    static_cast<Eigen::half>(-3.328f),  static_cast<Eigen::half>(5.906f),    static_cast<Eigen::half>(4.77f),
    static_cast<Eigen::half>(-3.959f),  static_cast<Eigen::half>(6.04f),     static_cast<Eigen::half>(-6.152f)};

const Eigen::half input2_fp16_data[] = {
    static_cast<Eigen::half>(-3.693f),  static_cast<Eigen::half>(6.85f),    static_cast<Eigen::half>(-3.316f),
    static_cast<Eigen::half>(-5.54f),   static_cast<Eigen::half>(9.45f),    static_cast<Eigen::half>(4.516f),
    static_cast<Eigen::half>(-8.73f),   static_cast<Eigen::half>(-5.246f),  static_cast<Eigen::half>(2.822f),
    static_cast<Eigen::half>(5.812f),   static_cast<Eigen::half>(4.305f),   static_cast<Eigen::half>(-5.094f),
    static_cast<Eigen::half>(-6.48f),   static_cast<Eigen::half>(4.8f),     static_cast<Eigen::half>(3.748f),
    static_cast<Eigen::half>(7.965f),   static_cast<Eigen::half>(-3.758f),  static_cast<Eigen::half>(8.57f),
    static_cast<Eigen::half>(-1.497f),  static_cast<Eigen::half>(6.977f),   static_cast<Eigen::half>(-8.4f),
    static_cast<Eigen::half>(0.384f),   static_cast<Eigen::half>(5.254f),   static_cast<Eigen::half>(9.26f),
    static_cast<Eigen::half>(7.125f),   static_cast<Eigen::half>(0.6836f),  static_cast<Eigen::half>(5.832f),
    static_cast<Eigen::half>(-9.72f),   static_cast<Eigen::half>(-3.484f),  static_cast<Eigen::half>(0.9097f),
    static_cast<Eigen::half>(-7.707f),  static_cast<Eigen::half>(4.363f),   static_cast<Eigen::half>(8.38f),
    static_cast<Eigen::half>(-7.816f),  static_cast<Eigen::half>(7.55f),    static_cast<Eigen::half>(9.32f),
    static_cast<Eigen::half>(-0.8794f), static_cast<Eigen::half>(-9.11f),   static_cast<Eigen::half>(-0.6934f),
    static_cast<Eigen::half>(-8.71f),   static_cast<Eigen::half>(-1.237f),  static_cast<Eigen::half>(-1.571f),
    static_cast<Eigen::half>(6.047f),   static_cast<Eigen::half>(5.418f),   static_cast<Eigen::half>(7.53f),
    static_cast<Eigen::half>(-8.08f),   static_cast<Eigen::half>(-9.23f),   static_cast<Eigen::half>(-6.266f),
    static_cast<Eigen::half>(-9.3f),    static_cast<Eigen::half>(0.2169f),  static_cast<Eigen::half>(-0.5146f),
    static_cast<Eigen::half>(9.5f),     static_cast<Eigen::half>(-0.3955f), static_cast<Eigen::half>(-8.516f),
    static_cast<Eigen::half>(2.463f),   static_cast<Eigen::half>(-4.363f),  static_cast<Eigen::half>(-1.986f),
    static_cast<Eigen::half>(-2.453f),  static_cast<Eigen::half>(2.254f),   static_cast<Eigen::half>(8.91f),
    static_cast<Eigen::half>(2.352f),   static_cast<Eigen::half>(1.51f),    static_cast<Eigen::half>(-8.14f),
    static_cast<Eigen::half>(-8.52f),   static_cast<Eigen::half>(2.025f),   static_cast<Eigen::half>(-5.105f),
    static_cast<Eigen::half>(-1.41f),   static_cast<Eigen::half>(-1.699f),  static_cast<Eigen::half>(5.332f),
    static_cast<Eigen::half>(4.203f),   static_cast<Eigen::half>(-8.055f),  static_cast<Eigen::half>(3.432f),
    static_cast<Eigen::half>(0.337f),   static_cast<Eigen::half>(-4.78f),   static_cast<Eigen::half>(-4.44f),
    static_cast<Eigen::half>(-1.367f),  static_cast<Eigen::half>(-2.195f),  static_cast<Eigen::half>(-1.422f),
    static_cast<Eigen::half>(3.904f),   static_cast<Eigen::half>(-0.1322f), static_cast<Eigen::half>(-0.2764f),
    static_cast<Eigen::half>(-2.707f),  static_cast<Eigen::half>(-2.03f),   static_cast<Eigen::half>(-8.7f),
    static_cast<Eigen::half>(-9.625f),  static_cast<Eigen::half>(2.654f),   static_cast<Eigen::half>(-2.553f),
    static_cast<Eigen::half>(-1.226f),  static_cast<Eigen::half>(-6.02f),   static_cast<Eigen::half>(-0.3855f),
    static_cast<Eigen::half>(-3.377f),  static_cast<Eigen::half>(4.023f),   static_cast<Eigen::half>(-3.592f),
    static_cast<Eigen::half>(0.4963f),  static_cast<Eigen::half>(-6.45f),   static_cast<Eigen::half>(5.305f),
    static_cast<Eigen::half>(1.763f),   static_cast<Eigen::half>(-8.02f),   static_cast<Eigen::half>(-1.178f),
    static_cast<Eigen::half>(-1.07f),   static_cast<Eigen::half>(1.602f),   static_cast<Eigen::half>(-0.2089f),
    static_cast<Eigen::half>(-2.152f),  static_cast<Eigen::half>(-9.55f),   static_cast<Eigen::half>(2.596f),
    static_cast<Eigen::half>(7.887f),   static_cast<Eigen::half>(3.396f),   static_cast<Eigen::half>(0.0712f),
    static_cast<Eigen::half>(-8.32f),   static_cast<Eigen::half>(1.708f),   static_cast<Eigen::half>(4.773f),
    static_cast<Eigen::half>(-0.2668f), static_cast<Eigen::half>(9.3f),     static_cast<Eigen::half>(-0.9326f),
    static_cast<Eigen::half>(8.195f),   static_cast<Eigen::half>(-9.63f),   static_cast<Eigen::half>(1.619f),
    static_cast<Eigen::half>(9.586f),   static_cast<Eigen::half>(-4.055f),  static_cast<Eigen::half>(4.57f),
    static_cast<Eigen::half>(5.164f),   static_cast<Eigen::half>(-0.394f),  static_cast<Eigen::half>(-1.35f),
    static_cast<Eigen::half>(8.94f),    static_cast<Eigen::half>(5.164f),   static_cast<Eigen::half>(-2.34f),
    static_cast<Eigen::half>(7.438f),   static_cast<Eigen::half>(9.09f),    static_cast<Eigen::half>(3.705f),
    static_cast<Eigen::half>(-4.11f),   static_cast<Eigen::half>(0.4211f),  static_cast<Eigen::half>(3.654f),
    static_cast<Eigen::half>(8.97f),    static_cast<Eigen::half>(-2.756f),  static_cast<Eigen::half>(0.666f),
    static_cast<Eigen::half>(-3.047f),  static_cast<Eigen::half>(-8.98f),   static_cast<Eigen::half>(7.99f),
    static_cast<Eigen::half>(-6.188f),  static_cast<Eigen::half>(5.75f),    static_cast<Eigen::half>(-4.258f),
    static_cast<Eigen::half>(-3.025f),  static_cast<Eigen::half>(-9.95f),   static_cast<Eigen::half>(-4.57f),
    static_cast<Eigen::half>(-4.41f),   static_cast<Eigen::half>(5.305f),   static_cast<Eigen::half>(8.74f),
    static_cast<Eigen::half>(-8.1f),    static_cast<Eigen::half>(-9.83f),   static_cast<Eigen::half>(9.89f),
    static_cast<Eigen::half>(-9.76f),   static_cast<Eigen::half>(4.523f),   static_cast<Eigen::half>(-8.83f),
    static_cast<Eigen::half>(7.34f),    static_cast<Eigen::half>(6.21f),    static_cast<Eigen::half>(1.349f),
    static_cast<Eigen::half>(-6.23f),   static_cast<Eigen::half>(-8.17f),   static_cast<Eigen::half>(2.13f),
    static_cast<Eigen::half>(-3.705f),  static_cast<Eigen::half>(8.32f),    static_cast<Eigen::half>(-7.066f),
    static_cast<Eigen::half>(-2.705f),  static_cast<Eigen::half>(-0.7275f), static_cast<Eigen::half>(1.442f),
    static_cast<Eigen::half>(-5.008f),  static_cast<Eigen::half>(-2.824f),  static_cast<Eigen::half>(5.227f),
    static_cast<Eigen::half>(-9.25f),   static_cast<Eigen::half>(-2.936f),  static_cast<Eigen::half>(9.55f),
    static_cast<Eigen::half>(-8.3f),    static_cast<Eigen::half>(6.734f),   static_cast<Eigen::half>(6.52f),
    static_cast<Eigen::half>(-1.474f),  static_cast<Eigen::half>(9.195f),   static_cast<Eigen::half>(-5.85f),
    static_cast<Eigen::half>(4.625f),   static_cast<Eigen::half>(2.533f),   static_cast<Eigen::half>(-0.1328f),
    static_cast<Eigen::half>(7.395f),   static_cast<Eigen::half>(6.777f),   static_cast<Eigen::half>(1.975f),
    static_cast<Eigen::half>(-6.414f),  static_cast<Eigen::half>(8.27f),    static_cast<Eigen::half>(-7.37f),
    static_cast<Eigen::half>(-3.096f),  static_cast<Eigen::half>(-1.073f),  static_cast<Eigen::half>(7.74f),
    static_cast<Eigen::half>(-5.734f),  static_cast<Eigen::half>(-3.107f),  static_cast<Eigen::half>(2.28f)};
const Eigen::half output_exp_fp16_data[] = {
    static_cast<Eigen::half>(145.0f),    static_cast<Eigen::half>(-13.77f),  static_cast<Eigen::half>(39.78f),
    static_cast<Eigen::half>(161.1f),    static_cast<Eigen::half>(-75.6f),   static_cast<Eigen::half>(-137.8f),
    static_cast<Eigen::half>(551.0f),    static_cast<Eigen::half>(29.8f),    static_cast<Eigen::half>(-124.44f),
    static_cast<Eigen::half>(-0.03995f), static_cast<Eigen::half>(-60.34f),  static_cast<Eigen::half>(61.9f),
    static_cast<Eigen::half>(3.428f),    static_cast<Eigen::half>(-3.686f),  static_cast<Eigen::half>(-355.2f),
    static_cast<Eigen::half>(-136.8f),   static_cast<Eigen::half>(272.5f),   static_cast<Eigen::half>(-620.5f),
    static_cast<Eigen::half>(148.1f),    static_cast<Eigen::half>(-656.5f),  static_cast<Eigen::half>(3.26f),
    static_cast<Eigen::half>(-18.9f),    static_cast<Eigen::half>(-169.8f),  static_cast<Eigen::half>(-4.688f),
    static_cast<Eigen::half>(-566.0f),   static_cast<Eigen::half>(-5.168f),  static_cast<Eigen::half>(-2.592f),
    static_cast<Eigen::half>(3.193f),    static_cast<Eigen::half>(227.4f),   static_cast<Eigen::half>(-68.5f),
    static_cast<Eigen::half>(4.207f),    static_cast<Eigen::half>(-219.5f),  static_cast<Eigen::half>(-257.5f),
    static_cast<Eigen::half>(6.926f),    static_cast<Eigen::half>(-562.5f),  static_cast<Eigen::half>(-502.8f),
    static_cast<Eigen::half>(29.45f),    static_cast<Eigen::half>(310.2f),   static_cast<Eigen::half>(42.2f),
    static_cast<Eigen::half>(496.5f),    static_cast<Eigen::half>(79.94f),   static_cast<Eigen::half>(52.16f),
    static_cast<Eigen::half>(-150.5f),   static_cast<Eigen::half>(-47.56f),  static_cast<Eigen::half>(-0.05173f),
    static_cast<Eigen::half>(0.01325f),  static_cast<Eigen::half>(26.31f),   static_cast<Eigen::half>(112.1f),
    static_cast<Eigen::half>(131.9f),    static_cast<Eigen::half>(-3.88f),   static_cast<Eigen::half>(25.0f),
    static_cast<Eigen::half>(-649.0f),   static_cast<Eigen::half>(5.977f),   static_cast<Eigen::half>(97.8f),
    static_cast<Eigen::half>(-168.0f),   static_cast<Eigen::half>(0.00315f), static_cast<Eigen::half>(0.0286f),
    static_cast<Eigen::half>(0.1167f),   static_cast<Eigen::half>(-9.38f),   static_cast<Eigen::half>(-61.8f),
    static_cast<Eigen::half>(-69.75f),   static_cast<Eigen::half>(-40.38f),  static_cast<Eigen::half>(317.5f),
    static_cast<Eigen::half>(32.47f),    static_cast<Eigen::half>(-59.72f),  static_cast<Eigen::half>(5.117f),
    static_cast<Eigen::half>(7.695f),    static_cast<Eigen::half>(21.92f),   static_cast<Eigen::half>(-243.2f),
    static_cast<Eigen::half>(-188.2f),   static_cast<Eigen::half>(390.5f),   static_cast<Eigen::half>(-172.5f),
    static_cast<Eigen::half>(-1.472f),   static_cast<Eigen::half>(148.4f),   static_cast<Eigen::half>(229.8f),
    static_cast<Eigen::half>(42.44f),    static_cast<Eigen::half>(11.78f),   static_cast<Eigen::half>(126.2f),
    static_cast<Eigen::half>(-34.22f),   static_cast<Eigen::half>(0.0432f),  static_cast<Eigen::half>(20.86f),
    static_cast<Eigen::half>(19.61f),    static_cast<Eigen::half>(2.607f),   static_cast<Eigen::half>(651.0f),
    static_cast<Eigen::half>(891.5f),    static_cast<Eigen::half>(-66.25f),  static_cast<Eigen::half>(40.75f),
    static_cast<Eigen::half>(12.195f),   static_cast<Eigen::half>(465.8f),   static_cast<Eigen::half>(21.75f),
    static_cast<Eigen::half>(51.25f),    static_cast<Eigen::half>(-8.44f),   static_cast<Eigen::half>(16.98f),
    static_cast<Eigen::half>(-8.016f),   static_cast<Eigen::half>(443.0f),   static_cast<Eigen::half>(-0.615f),
    static_cast<Eigen::half>(-162.8f),   static_cast<Eigen::half>(56.28f),   static_cast<Eigen::half>(35.16f),
    static_cast<Eigen::half>(11.02f),    static_cast<Eigen::half>(-0.5874f), static_cast<Eigen::half>(2.867f),
    static_cast<Eigen::half>(0.00454f),  static_cast<Eigen::half>(70.7f),    static_cast<Eigen::half>(-181.4f),
    static_cast<Eigen::half>(-370.2f),   static_cast<Eigen::half>(-17.27f),  static_cast<Eigen::half>(-0.2233f),
    static_cast<Eigen::half>(4.027f),    static_cast<Eigen::half>(-1.731f),  static_cast<Eigen::half>(-119.56f),
    static_cast<Eigen::half>(0.10065f),  static_cast<Eigen::half>(-786.0f),  static_cast<Eigen::half>(35.1f),
    static_cast<Eigen::half>(-727.5f),   static_cast<Eigen::half>(40.7f),    static_cast<Eigen::half>(-102.6f),
    static_cast<Eigen::half>(-5.6f),     static_cast<Eigen::half>(75.5f),    static_cast<Eigen::half>(-145.0f),
    static_cast<Eigen::half>(-44.78f),   static_cast<Eigen::half>(19.44f),   static_cast<Eigen::half>(47.4f),
    static_cast<Eigen::half>(-7.23f),    static_cast<Eigen::half>(-498.0f),  static_cast<Eigen::half>(52.72f),
    static_cast<Eigen::half>(-22.02f),   static_cast<Eigen::half>(-891.0f),  static_cast<Eigen::half>(-10.99f),
    static_cast<Eigen::half>(62.1f),     static_cast<Eigen::half>(-14.555f), static_cast<Eigen::half>(-279.2f),
    static_cast<Eigen::half>(-833.5f),   static_cast<Eigen::half>(20.9f),    static_cast<Eigen::half>(-41.4f),
    static_cast<Eigen::half>(47.3f),     static_cast<Eigen::half>(425.2f),   static_cast<Eigen::half>(-1.081f),
    static_cast<Eigen::half>(169.8f),    static_cast<Eigen::half>(-372.2f),  static_cast<Eigen::half>(264.8f),
    static_cast<Eigen::half>(0.771f),    static_cast<Eigen::half>(626.5f),   static_cast<Eigen::half>(254.4f),
    static_cast<Eigen::half>(201.6f),    static_cast<Eigen::half>(-317.0f),  static_cast<Eigen::half>(-9.73f),
    static_cast<Eigen::half>(150.9f),    static_cast<Eigen::half>(444.8f),   static_cast<Eigen::half>(-375.0f),
    static_cast<Eigen::half>(944.0f),    static_cast<Eigen::half>(-96.25f),  static_cast<Eigen::half>(48.6f),
    static_cast<Eigen::half>(-595.5f),   static_cast<Eigen::half>(-19.58f),  static_cast<Eigen::half>(-15.08f),
    static_cast<Eigen::half>(288.8f),    static_cast<Eigen::half>(109.6f),   static_cast<Eigen::half>(-175.0f),
    static_cast<Eigen::half>(158.2f),    static_cast<Eigen::half>(-46.03f),  static_cast<Eigen::half>(510.5f),
    static_cast<Eigen::half>(2.027f),    static_cast<Eigen::half>(45.28f),   static_cast<Eigen::half>(-108.6f),
    static_cast<Eigen::half>(161.9f),    static_cast<Eigen::half>(152.1f),   static_cast<Eigen::half>(-171.6f),
    static_cast<Eigen::half>(824.0f),    static_cast<Eigen::half>(7.99f),    static_cast<Eigen::half>(-773.5f),
    static_cast<Eigen::half>(662.5f),    static_cast<Eigen::half>(-266.8f),  static_cast<Eigen::half>(-3.285f),
    static_cast<Eigen::half>(21.97f),    static_cast<Eigen::half>(-4.766f),  static_cast<Eigen::half>(483.8f),
    static_cast<Eigen::half>(-44.72f),   static_cast<Eigen::half>(-0.5796f), static_cast<Eigen::half>(4.977f),
    static_cast<Eigen::half>(-39.75f),   static_cast<Eigen::half>(-545.0f),  static_cast<Eigen::half>(-136.1f),
    static_cast<Eigen::half>(42.12f),    static_cast<Eigen::half>(-460.0f),  static_cast<Eigen::half>(185.4f),
    static_cast<Eigen::half>(34.28f),    static_cast<Eigen::half>(37.44f),   static_cast<Eigen::half>(-176.0f),
    static_cast<Eigen::half>(89.9f),     static_cast<Eigen::half>(113.3f),   static_cast<Eigen::half>(-86.25f)};

#define CREATE_NODEDEF(shapes, data_types, datas)                    \
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
    NodeDefBuilder(node_def.get(), "InvGrad", "InvGrad")             \
        .Input({"x1", data_types[0], shapes[0], datas[0]})           \
        .Input({"x2", data_types[1], shapes[1], datas[1]})           \
        .Output({"y", data_types[2], shapes[2], datas[2]})

template <typename T1, typename T2, typename T3>
void RunInvGradKernel(
    vector<DataType> data_types, vector<vector<int64_t>>& shapes, const T1* input1_data, const T2* input2_data,
    const T3* output_exp_data)
{
    uint64_t input1_size = CalTotalElements(shapes, 0);
    T1* input1 = new T1[input1_size];

    uint64_t input2_size = CalTotalElements(shapes, 1);
    T2* input2 = new T2[input2_size];

    for (uint64_t i = 0; i < input1_size; ++i) {
        input1[i] = input1_data[i];
    }
    for (uint64_t i = 0; i < input2_size; ++i) {
        input2[i] = input2_data[i];
    }

    uint64_t output_size = CalTotalElements(shapes, 2);
    T3* output = new T3[output_size];
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    T3* output_exp = new T3[output_size];
    for (uint64_t i = 0; i < output_size; ++i) {
        output_exp[i] = output_exp_data[i];
    }

    bool compare = CompareResult(output, output_exp, output_size);
    EXPECT_EQ(compare, true);
    delete[] input1;
    delete[] input2;
    delete[] output;
    delete[] output_exp;
}

TEST_F(TEST_INVGRAD_UT, DATA_TYPE_FLOAT_SUCC_1D)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{15}, {15}, {15}};
    const float input1_data[] = {81.3377f,   42.911854f, 67.3148f,   23.028294f, 35.85503f,
                                 22.392572f, 10.274563f, 61.913395f, 83.20328f,  49.58565f,
                                 68.719315f, 67.43358f,  46.36467f,  54.379158f, 98.67426f};
    const float input2_data[] = {70.72506f,  92.57594f,  92.54831f,  99.74494f,  98.53133f,
                                 46.88357f,  14.914657f, 78.41768f,  46.443245f, 94.559235f,
                                 36.245827f, 53.333603f, 52.865215f, 9.609693f,  6.6154857f};
    const float output_exp_data[] = {-467904.34f, -170471.86f, -419362.47f, -52894.973f, -126670.21f,
                                     -23508.703f, -1574.4902f, -300596.03f, -321516.62f, -232496.27f,
                                     -171165.27f, -242523.23f, -113643.44f, -28416.752f, -64412.406f};

    RunInvGradKernel<float, float, float>(data_types, shapes, input1_data, input2_data, output_exp_data);
}

TEST_F(TEST_INVGRAD_UT, DATA_TYPE_FLOAT16_SUCC_1D)
{
    vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
    vector<vector<int64_t>> shapes = {{16, 12}, {16, 12}, {16, 12}};

    RunInvGradKernel<Eigen::half, Eigen::half, Eigen::half>(
        data_types, shapes, input1_fp16_data, input2_fp16_data, output_exp_fp16_data);
}

// exception instance
TEST_F(TEST_INVGRAD_UT, INPUT_NUMBER_EXCEPTION)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{2, 2, 4}, {2, 2, 3}, {2, 2, 4}};
    float input1[12] = {(float)1};
    float input2[16] = {(float)0};
    float output[16] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_INVGRAD_UT, INPUT_DTYPE_EXCEPTION)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT16, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}, {2, 11}};
    float input1[22] = {(float)1};
    Eigen::half input2[22] = {(Eigen::half)0};
    float output[22] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_INVGRAD_UT, INPUT_NULL_EXCEPTION)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}, {2, 11}};
    float output[22] = {0};
    vector<void*> datas = {(void*)nullptr, (void*)nullptr, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_INVGRAD_UT, INPUT_BOOL_UNSUPPORT)
{
    vector<DataType> data_types = {DT_BOOL, DT_BOOL, DT_BOOL};
    vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}, {2, 11}};
    bool input1[22] = {(bool)1};
    bool input2[22] = {(bool)0};
    bool output[22] = {(bool)0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_INVGRAD_UT, DATA_TYPE_DOUBLE_SUCC_1D)
{
    vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
    vector<vector<int64_t>> shapes = {{8}, {8}, {8}};
    const double input1_data[] = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8};
    const double input2_data[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    const double output_exp_data[] = {-0.121, -0.968, -3.267, -7.744, -15.125, -26.136, -41.503, -61.952};

    RunInvGradKernel<double, double, double>(data_types, shapes, input1_data, input2_data, output_exp_data);
}

TEST_F(TEST_INVGRAD_UT, DATA_TYPE_COMPLEX64_SUCC_1D)
{
    vector<DataType> data_types = {DT_COMPLEX64, DT_COMPLEX64, DT_COMPLEX64};
    vector<vector<int64_t>> shapes = {{4}, {4}, {4}};
    const std::complex<float> input1_data[] = {{1.0f, 0.0f}, {2.0f, 0.0f}, {3.0f, 0.0f}, {4.0f, 0.0f}};
    const std::complex<float> input2_data[] = {{0.1f, 0.0f}, {0.2f, 0.0f}, {0.3f, 0.0f}, {0.4f, 0.0f}};
    const std::complex<float> output_exp_data[] = {{-0.1f, 0.0f}, {-0.8f, 0.0f}, {-2.7f, 0.0f}, {-6.4f, 0.0f}};

    RunInvGradKernel<std::complex<float>, std::complex<float>, std::complex<float>>(data_types, shapes, input1_data, input2_data, output_exp_data);
}

TEST_F(TEST_INVGRAD_UT, DATA_TYPE_COMPLEX128_SUCC_1D)
{
    vector<DataType> data_types = {DT_COMPLEX128, DT_COMPLEX128, DT_COMPLEX128};
    vector<vector<int64_t>> shapes = {{3}, {3}, {3}};
    const std::complex<double> input1_data[] = {{1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}};
    const std::complex<double> input2_data[] = {{0.1, 0.0}, {0.2, 0.0}, {0.3, 0.0}};
    const std::complex<double> output_exp_data[] = {{-0.1, 0.0}, {-0.8, 0.0}, {-2.7, 0.0}};

    RunInvGradKernel<std::complex<double>, std::complex<double>, std::complex<double>>(data_types, shapes, input1_data, input2_data, output_exp_data);
}

TEST_F(TEST_INVGRAD_UT, DATA_TYPE_INT32_UNSUPPORT)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{4}, {4}, {4}};
    int32_t input1[4] = {1, 2, 3, 4};
    int32_t input2[4] = {1, 1, 1, 1};
    int32_t output[4] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}
