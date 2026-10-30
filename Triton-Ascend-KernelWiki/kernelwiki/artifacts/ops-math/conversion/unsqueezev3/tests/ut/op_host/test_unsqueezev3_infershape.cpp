/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "register/op_impl_registry.h"
#include "infershape_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace {
template <class T>
gert::ContextHolder<gert::InferShapeContext> GetKernelRunContextHolder(std::unique_ptr<uint8_t[]>& input_tensor_holder,
                                                       std::vector<T> axes, gert::StorageShape &input_shape,
                                                       gert::StorageShape &output_shape,
                                                       const gert::StorageShape &axis_storage_shape,
                                                       const gert::StorageFormat &axis_storage_fmt, ge::DataType dt) {
  auto axes_shape_size = axes.size();
  auto input_tensor = reinterpret_cast<gert::Tensor*>(input_tensor_holder.get());
  auto tensor_data = reinterpret_cast<T*>(input_tensor + 1);
  new (input_tensor) gert::Tensor(axis_storage_shape, axis_storage_fmt, gert::kFollowing, dt, tensor_data);
  for (size_t idx = 0; idx < axes_shape_size; ++idx) {
    tensor_data[idx] = axes[idx];
  }
  auto holder = gert::InferShapeContextFaker()
                    .SetOpType("UnsqueezeV3")
                    .NodeIoNum(2, 1)
                    .InputTensors({(gert::Tensor *)&input_shape, input_tensor})
                    .OutputShapes({&output_shape})
                    .Build();
  return holder;
}
}  // namespace

class UNSQUEEZEV3_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "UNSQUEEZEV3_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "UNSQUEEZEV3_UT TearDown" << std::endl;
  }
};

TEST_F(UNSQUEEZEV3_UT, rt2_unsqueezev3_no_axis_input) {
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto infer_shape_func = space_registry->GetOpImpl("UnsqueezeV3")->infer_shape;

  gert::StorageShape input_shape = {{8, 3, 224, 224}, {8, 3, 224, 224}};
  gert::StorageShape output_shape = {{}, {}};
  auto holder1 = gert::InferShapeContextFaker()
                      .SetOpType("UnsqueezeV3")
                      .NodeIoNum(2, 1)
                      .InputTensors({(gert::Tensor *)&input_shape})
                      .OutputShapes({&output_shape})
                      .Build();
  EXPECT_EQ(infer_shape_func(holder1.GetContext()), ge::GRAPH_FAILED);

  auto holder2 = gert::InferShapeContextFaker()
                      .SetOpType("UnsqueezeV3")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor *)&input_shape})
                      .OutputShapes({&output_shape})
                      .Build();
  EXPECT_EQ(infer_shape_func(holder2.GetContext()), ge::GRAPH_FAILED);
}

TEST_F(UNSQUEEZEV3_UT, rt2_unsqueezev3_unknown_rank_axis_OK) {
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto infer_shape_func = space_registry->GetOpImpl("UnsqueezeV3")->infer_shape;

  std::vector<int64_t> axes = {-5, 3};
  gert::StorageShape input_shape = {{-2}, {-2}};
  gert::StorageShape output_shape = {{}, {}};
  gert::StorageShape axis_storage_shape = {{2}, {2}};
  gert::StorageFormat axis_storage_fmt = {ge::FORMAT_ND, ge::FORMAT_ND, {}};
  ge::DataType axis_dt = ge::DT_INT64;
  auto input_tensor_holder =
      std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * axes.size()]);
  ASSERT_NE(input_tensor_holder, nullptr);
  auto input_tensor = reinterpret_cast<gert::Tensor*>(input_tensor_holder.get());
  auto tensor_data = reinterpret_cast<int64_t*>(input_tensor + 1);
  new (input_tensor) gert::Tensor(axis_storage_shape, axis_storage_fmt, gert::kFollowing, ge::DT_INT64, nullptr);
  tensor_data[0] = axes[0];
  tensor_data[1] = axes[1];
  auto holder = gert::InferShapeContextFaker()
                      .SetOpType("UnsqueezeV3")
                      .NodeIoNum(2, 1)
                      .InputTensors({(gert::Tensor *)&input_shape, input_tensor})
                      .OutputShapes({&output_shape})
                      .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 1);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), -2);
}

TEST_F(UNSQUEEZEV3_UT, rt2_unsqueezev3_no_axis_input_buf) {
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto infer_shape_func = space_registry->GetOpImpl("UnsqueezeV3")->infer_shape;

  gert::StorageShape input_shape = {{1, 3, 224, 224}, {1, 3, 224, 224}};
  gert::StorageShape output_shape = {{}, {}};
  gert::StorageShape axis_storage_shape = {{1}, {1}};
  gert::StorageFormat axis_storage_fmt = {ge::FORMAT_ND, ge::FORMAT_ND, {}};
  auto input_tensor_holder = std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * 0UL]);
  ASSERT_NE(input_tensor_holder, nullptr);
  auto input_tensor = reinterpret_cast<gert::Tensor*>(input_tensor_holder.get());
  new (input_tensor) gert::Tensor(axis_storage_shape, axis_storage_fmt, gert::kFollowing, ge::DT_INT64, nullptr);

  auto holder = gert::InferShapeContextFaker()
                      .SetOpType("UnsqueezeV3")
                      .NodeIoNum(2, 1)
                      .InputTensors({(gert::Tensor *)&input_shape, input_tensor})
                      .OutputShapes({&output_shape})
                      .Build();
  EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 1);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), -2);
}

TEST_F(UNSQUEEZEV3_UT, rt2_unsqueezev3_empty_axis_OK) {
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto infer_shape_func = space_registry->GetOpImpl("UnsqueezeV3")->infer_shape;

  std::vector<int64_t> axes = {};
  gert::StorageShape input_shape = {{3, 2, -1, 5}, {3, 2, -1, 5}};
  gert::StorageShape output_shape = {{}, {}};
  gert::StorageShape axis_storage_shape = {{0}, {0}};
  gert::StorageFormat axis_storage_fmt = {ge::FORMAT_ND, ge::FORMAT_ND, {}};
  ge::DataType axis_dt = ge::DT_INT64;
  auto input_tensor_holder =
      std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * axes.size()]);
  auto holder = GetKernelRunContextHolder<int64_t>(input_tensor_holder, axes, input_shape, output_shape,
                                                   axis_storage_shape, axis_storage_fmt, axis_dt);
  EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 4);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 3);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 2);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), -1);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(3), 5);
}

TEST_F(UNSQUEEZEV3_UT, rt2_unsqueezev3_axis_out_of_upper_bound) {
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto infer_shape_func = space_registry->GetOpImpl("UnsqueezeV3")->infer_shape;

  std::vector<int64_t> axes = {5};
  gert::StorageShape input_shape = {{1, 3, 2, 5}, {1, 3, 2, 5}};
  gert::StorageShape output_shape = {{}, {}};
  gert::StorageShape axis_storage_shape = {{1}, {1}};
  gert::StorageFormat axis_storage_fmt = {ge::FORMAT_ND, ge::FORMAT_ND, {}};
  ge::DataType axis_dt = ge::DT_INT64;
  auto input_tensor_holder =
      std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * axes.size()]);
  auto holder = GetKernelRunContextHolder<int64_t>(input_tensor_holder, axes, input_shape, output_shape,
                                                   axis_storage_shape, axis_storage_fmt, axis_dt);
  EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_FAILED);
}

TEST_F(UNSQUEEZEV3_UT, rt2_unsqueezev3_axis_out_of_lower_bound) {
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto infer_shape_func = space_registry->GetOpImpl("UnsqueezeV3")->infer_shape;

  std::vector<int64_t> axes = {-6};
  gert::StorageShape input_shape = {{1, 3, 2, 5}, {1, 3, 2, 5}};
  gert::StorageShape output_shape = {{}, {}};
  gert::StorageShape axis_storage_shape = {{1}, {1}};
  gert::StorageFormat axis_storage_fmt = {ge::FORMAT_ND, ge::FORMAT_ND, {}};
  ge::DataType axis_dt = ge::DT_INT64;
  auto input_tensor_holder =
      std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * axes.size()]);
  auto holder = GetKernelRunContextHolder<int64_t>(input_tensor_holder, axes, input_shape, output_shape,
                                                   axis_storage_shape, axis_storage_fmt, axis_dt);
  EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_FAILED);
}

TEST_F(UNSQUEEZEV3_UT, rt2_unsqueezev3_repeated_axis) {
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto infer_shape_func = space_registry->GetOpImpl("UnsqueezeV3")->infer_shape;

  std::vector<int64_t> axes = {2, 2};
  gert::StorageShape input_shape = {{1, 3, 2, 5}, {1, 3, 2, 5}};
  gert::StorageShape output_shape = {{}, {}};
  gert::StorageShape axis_storage_shape = {{2}, {2}};
  gert::StorageFormat axis_storage_fmt = {ge::FORMAT_ND, ge::FORMAT_ND, {}};
  ge::DataType axis_dt = ge::DT_INT64;
  auto input_tensor_holder =
      std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * axes.size()]);
  auto holder = GetKernelRunContextHolder<int64_t>(input_tensor_holder, axes, input_shape, output_shape,
                                                   axis_storage_shape, axis_storage_fmt, axis_dt);
  EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_FAILED);
}

TEST_F(UNSQUEEZEV3_UT, rt2_unsqueezev3_normal_one_axis_OK) {
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto infer_shape_func = space_registry->GetOpImpl("UnsqueezeV3")->infer_shape;

  std::vector<int64_t> axes = {4};
  gert::StorageShape input_shape = {{1, 3, 2, 5}, {1, 3, 2, 5}};
  gert::StorageShape output_shape = {{}, {}};
  gert::StorageShape axis_storage_shape = {{1}, {1}};
  gert::StorageFormat axis_storage_fmt = {ge::FORMAT_ND, ge::FORMAT_ND, {}};
  ge::DataType axis_dt = ge::DT_INT64;
  auto input_tensor_holder =
      std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * axes.size()]);
  auto holder = GetKernelRunContextHolder<int64_t>(input_tensor_holder, axes, input_shape, output_shape,
                                                   axis_storage_shape, axis_storage_fmt, axis_dt);
  EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 5);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 1);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 3);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), 2);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(3), 5);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(4), 1);
}

TEST_F(UNSQUEEZEV3_UT, rt2_unsqueezev3_normal_negative_axis_OK) {
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto infer_shape_func = space_registry->GetOpImpl("UnsqueezeV3")->infer_shape;

  std::vector<int64_t> axes = {-5};
  gert::StorageShape input_shape = {{1, 3, 2, 5}, {1, 3, 2, 5}};
  gert::StorageShape output_shape = {{}, {}};
  gert::StorageShape axis_storage_shape = {{1}, {1}};
  gert::StorageFormat axis_storage_fmt = {ge::FORMAT_ND, ge::FORMAT_ND, {}};
  ge::DataType axis_dt = ge::DT_INT64;
  auto input_tensor_holder =
      std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * axes.size()]);
  auto holder = GetKernelRunContextHolder<int64_t>(input_tensor_holder, axes, input_shape, output_shape,
                                                   axis_storage_shape, axis_storage_fmt, axis_dt);
  EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 5);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 1);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 1);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), 3);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(3), 2);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(4), 5);

  auto axes_input_tensor = reinterpret_cast<gert::Tensor *>(input_tensor_holder.get());
  auto axes_tensor_data = reinterpret_cast<int64_t *>(axes_input_tensor + 1);
  ASSERT_EQ(axes_input_tensor->GetShapeSize(), axes.size());
  // Unsqueezev3算子的输入节点axes可能会作为其他节点的输入，因此不能修改axes的tensor值
  EXPECT_EQ(axes_tensor_data[0], axes[0]);
}

TEST_F(UNSQUEEZEV3_UT, rt2_unsqueezev3_unsorted_axis_OK) {
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto infer_shape_func = space_registry->GetOpImpl("UnsqueezeV3")->infer_shape;

  std::vector<int64_t> axes = {3, -6, 5};
  gert::StorageShape input_shape = {{-1, -1, 5}, {-1, -1, 5}};
  gert::StorageShape output_shape = {{}, {}};
  gert::StorageShape axis_storage_shape = {{3}, {3}};
  gert::StorageFormat axis_storage_fmt = {ge::FORMAT_ND, ge::FORMAT_ND, {}};
  ge::DataType axis_dt = ge::DT_INT64;
  auto input_tensor_holder =
      std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * axes.size()]);
  auto holder = GetKernelRunContextHolder<int64_t>(input_tensor_holder, axes, input_shape, output_shape,
                                                   axis_storage_shape, axis_storage_fmt, axis_dt);
  EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 6);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 1);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), -1);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), -1);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(3), 1);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(4), 5);
  EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(5), 1);
}