//===------ NarrowTensorEmpty.cpp -  narrow liveness of tensor empty-------===//
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
//
// This file implements a pass to narrow liveness of tensor empty.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_NARROWTENSOROP
#include "bishengir/Dialect/Tensor/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct NarrowTensorOp : public impl::NarrowTensorOpBase<NarrowTensorOp> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

template <typename OP>
struct NarrowTensorOpPattern : public OpRewritePattern<OP> {
public:
  using OpRewritePattern<OP>::OpRewritePattern;
  LogicalResult
  matchAndRewrite(OP op, mlir::PatternRewriter &rewriter) const override {
    auto candidateLoop = getCandidateLoopOfUsers(op);
    if (!candidateLoop ||
        candidateLoop == op->template getParentOfType<LoopLikeOpInterface>()) {
      return failure();
    }
    assert(!candidateLoop.getLoopRegions().empty());
    // clone tensor.empty and replace
    rewriter.setInsertionPointToStart(
        &(candidateLoop.getLoopRegions()[0]->getBlocks().front()));
    auto clonedOp = rewriter.clone(*op.getOperation());
    rewriter.replaceAllUsesExcept(op->getResult(0), clonedOp->getResult(0), op);
    return success();
  }

private:
  LoopLikeOpInterface getCandidateLoopOfUsers(Operation *op) const {
    LoopLikeOpInterface candidateLoop;
    for (const auto &user : op->getUsers()) {
      auto parentLoopOpOfUser = user->getParentOfType<LoopLikeOpInterface>();
      if (!parentLoopOpOfUser) {
        return parentLoopOpOfUser;
      }

      if (!candidateLoop ||
          parentLoopOpOfUser->isProperAncestor(candidateLoop)) {
        candidateLoop = parentLoopOpOfUser;
      }
    }

    return candidateLoop;
  }
};

void NarrowTensorOp::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  patterns.insert<NarrowTensorOpPattern<tensor::EmptyOp>,
                  NarrowTensorOpPattern<tensor::ExtractSliceOp>>(
      patterns.getContext());
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<Pass> mlir::tensor::createNarrowTensorOpPass() {
  return std::make_unique<NarrowTensorOp>();
}
