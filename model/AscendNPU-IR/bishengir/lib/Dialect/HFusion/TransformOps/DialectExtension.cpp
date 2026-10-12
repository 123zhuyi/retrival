//===- DialectExtension.cpp - HFusion transform dialect extension ---------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/TransformOps/HFusionTransformOps.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/MathExt/IR/MathExt.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

namespace {
class HFusionTransformDialectExtension
    : public transform::TransformDialectExtension<
          HFusionTransformDialectExtension> {
public:
  using Base::Base;

  void init() {
    declareDependentDialect<hfusion::HFusionDialect>();
    declareDependentDialect<linalg::LinalgDialect>();
    declareDependentDialect<func::FuncDialect>();

    declareGeneratedDialect<affine::AffineDialect>();
    declareGeneratedDialect<annotation::AnnotationDialect>();
    declareGeneratedDialect<arith::ArithDialect>();
    declareGeneratedDialect<index::IndexDialect>();
    declareGeneratedDialect<linalg::LinalgDialect>();
    declareGeneratedDialect<tensor::TensorDialect>();
    declareDependentDialect<hfusion::HFusionDialect>();
    declareDependentDialect<hivm::HIVMDialect>();

    registerTransformOps<
#define GET_OP_LIST
#include "bishengir/Dialect/HFusion/TransformOps/HFusionTransformOps.cpp.inc"
        >();
  }
};
} // namespace

void mlir::hfusion::registerTransformDialectExtension(
    DialectRegistry &registry) {
  registry.addExtensions<HFusionTransformDialectExtension>();
}
