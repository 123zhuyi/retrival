//===-----DecomposeTensorConcat.cpp -----------------------------*- C++ -*-===//
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
//===---------------------------------------------------------------------===//

#include "bishengir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_DECOMPOSETENSORCONCAT
#include "bishengir/Dialect/Tensor/Transforms/Passes.h.inc"
} // namespace mlir

namespace {
using namespace mlir;

struct DecomposeTensorConcatPass
    : public impl::DecomposeTensorConcatBase<DecomposeTensorConcatPass> {
  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    tensor::populateDecomposeTensorConcatPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::tensor::createDecomposeTensorConcatPass() {
  return std::make_unique<DecomposeTensorConcatPass>();
}
