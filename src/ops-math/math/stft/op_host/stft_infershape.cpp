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
 * \file stft_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {

static constexpr size_t STFT_IDX_IN_NUM = 0;
static constexpr size_t STFT_IDX_IN_THREE = 2;
static constexpr size_t STFT_IDX_OUT_NUM = 0;
static constexpr size_t COMPLEX_IDX = 4;
static constexpr size_t ONE_SIDED_IDX = 3;
static constexpr size_t N_FFT_IDX = 5;
static constexpr size_t HOP_LENGTH_IDX = 0;
static constexpr size_t WIN_LENGTH_IDX = 1;
static constexpr size_t SUPPORTED_DIM_TWO = 2;
static constexpr size_t SUPPORTED_DIM_THREE = 3;
static constexpr size_t SUPPORTED_DIM_FOUR = 4;
static constexpr size_t DIM_NUM_ZERO = 0;
static constexpr size_t DIM_NUM_ONE = 1;
static constexpr size_t DIM_NUM_TWO = 2;
static constexpr size_t DIM_NUM_THREE = 3;

static ge::graphStatus InferShape4STFT(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "----------- enter InferShape4STFT. -----------");
    auto x_desc = context->GetInputTensor(STFT_IDX_IN_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_desc);

    auto x_shape = context->GetInputShape(STFT_IDX_IN_NUM);
    int64_t dims = x_shape->GetDimNum();
    if (dims != DIM_NUM_ONE && dims != DIM_NUM_TWO) {
        OP_LOGE(context->GetNodeName(), "The input 'x' expected a 1D or 2D tensor.");
        return ge::GRAPH_FAILED;
    }
    int64_t len = (dims == 1) ? x_shape->GetDim(0) : x_shape->GetDim(1);
    int64_t batch = (dims == 1) ? 1 : x_shape->GetDim(0);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const bool* complex_attr = attrs->GetAttrPointer<bool>(COMPLEX_IDX);
    bool complex = true;
    if (complex_attr != nullptr && *complex_attr == false) {
        complex = false;
    }

    const bool* one_sided_attr = attrs->GetAttrPointer<bool>(ONE_SIDED_IDX);
    bool one_sided = true;
    if (one_sided_attr != nullptr && *one_sided_attr == false) {
        one_sided = false;
    }

    const int64_t* n_fft_attr = attrs->GetAttrPointer<int64_t>(N_FFT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, n_fft_attr);
    int n_fft = *n_fft_attr;

    const int64_t* hop_length_attr = attrs->GetAttrPointer<int64_t>(HOP_LENGTH_IDX);
    int hop_length = (n_fft >> 2);
    if (hop_length_attr != nullptr && *hop_length_attr != 0) {
        hop_length = *hop_length_attr;
    }

    const int64_t* win_length_attr = attrs->GetAttrPointer<int64_t>(WIN_LENGTH_IDX);
    int win_length = n_fft;
    if (win_length_attr != nullptr && *win_length_attr != 0) {
        win_length = *win_length_attr;
    }

    auto window = context->GetInputDesc(STFT_IDX_IN_THREE);
    if (window != nullptr) {
        auto win_shape = context->GetInputShape(STFT_IDX_IN_THREE);
        int64_t win_dim = win_shape->GetDimNum();
        int64_t window_length = win_shape->GetDim(0);
        if (win_dim != 1 || window_length != win_length) {
            OP_LOGE(context->GetNodeName(), "Expected a 1D window tensor of size equal to win_length.");
            return GRAPH_FAILED;
        }
    }

    int64_t n_frames = 1 + (len - n_fft) / hop_length;
    auto out_shape = context->GetOutputShape(STFT_IDX_OUT_NUM);
    int32_t n_dim = (!one_sided) ? n_fft : (n_fft / 2 + 1);
    if (batch == 1 && dims == 1 && complex) {
        out_shape->SetDimNum(SUPPORTED_DIM_TWO);
        out_shape->SetDim(0, n_dim);
        out_shape->SetDim(1, n_frames);
    } else if (batch == 1 && dims == 1 && !complex) {
        out_shape->SetDimNum(SUPPORTED_DIM_THREE);
        out_shape->SetDim(0, n_dim);
        out_shape->SetDim(1, n_frames);
        out_shape->SetDim(DIM_NUM_TWO, DIM_NUM_TWO);
    } else if (complex) {
        out_shape->SetDimNum(SUPPORTED_DIM_THREE);
        out_shape->SetDim(0, batch);
        out_shape->SetDim(1, n_dim);
        out_shape->SetDim(DIM_NUM_TWO, n_frames);
    } else {
        out_shape->SetDimNum(SUPPORTED_DIM_FOUR);
        out_shape->SetDim(0, batch);
        out_shape->SetDim(1, n_dim);
        out_shape->SetDim(DIM_NUM_TWO, n_frames);
        out_shape->SetDim(DIM_NUM_THREE, DIM_NUM_TWO);
    }
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4STFT(gert::InferDataTypeContext* context)
{
    auto x_type = context->GetInputDataType(STFT_IDX_IN_NUM);
    if (!(x_type == DT_COMPLEX64 || x_type == DT_COMPLEX128 || x_type == DT_DOUBLE || x_type == DT_FLOAT)) {
        OP_LOGE(context->GetNodeName(), "The input 'x' should have dtype float, double, complex64 or complex128.");
        return ge::GRAPH_FAILED;
    }

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    const bool* complex_attr = attrs->GetAttrPointer<bool>(COMPLEX_IDX);
    bool complex = true;
    if (complex_attr != nullptr && *complex_attr == false) {
        complex = false;
    }

    auto window = context->GetInputDesc(STFT_IDX_IN_THREE);
    DataType type = DT_FLOAT;
    if (window != nullptr) {
        auto window_type = context->GetInputDataType(STFT_IDX_IN_THREE);
        if (!(window_type == DT_COMPLEX64 || window_type == DT_COMPLEX128 || window_type == DT_DOUBLE ||
              window_type == DT_FLOAT)) {
            OP_LOGE(context->GetNodeName(), "Window should have dtype float, double, complex64 or complex128.");
            return GRAPH_FAILED;
        }

        if (x_type == DT_DOUBLE || x_type == DT_COMPLEX128) {
            type = complex ? DT_COMPLEX128 : DT_DOUBLE;
        } else if (x_type == DT_FLOAT || x_type == DT_COMPLEX64) {
            if (window_type == DT_FLOAT || window_type == DT_COMPLEX64) {
                type = complex ? DT_COMPLEX64 : DT_FLOAT;
            } else if (window_type == DT_DOUBLE || window_type == DT_COMPLEX128) {
                type = complex ? DT_COMPLEX128 : DT_DOUBLE;
            }
        }
    } else {
        if (x_type == DT_FLOAT || x_type == DT_COMPLEX64) {
            type = complex ? DT_COMPLEX64 : DT_FLOAT;
        } else if (x_type == DT_DOUBLE || x_type == DT_COMPLEX128) {
            type = complex ? DT_COMPLEX128 : DT_DOUBLE;
        }
    }
    context->SetOutputDataType(STFT_IDX_OUT_NUM, type);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(STFT).InferShape(InferShape4STFT).InferDataType(InferDataType4STFT);

} // namespace ops