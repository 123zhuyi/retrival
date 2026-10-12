//===--------- FoldAllocReshape.cpp -  fold alloc and reshape ops ---------===//
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
// This file implements a pass to fold alloc and reshape ops.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_FOLDALLOCRESHAPEOP
#include "bishengir/Dialect/MemRef/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct FoldAllocReshapeOp
    : public impl::FoldAllocReshapeOpBase<FoldAllocReshapeOp> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

struct FoldAllocReshapeOpPattern
    : public OpRewritePattern<memref::ExpandShapeOp> {
public:
  using OpRewritePattern<memref::ExpandShapeOp>::OpRewritePattern;
  LogicalResult
  matchAndRewrite(memref::ExpandShapeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto definingOp = op.getSrc().getDefiningOp<memref::AllocOp>();
    if (!definingOp) {
      return failure();
    }

    if (!definingOp->hasOneUse()) {
      return failure();
    }

    rewriter.setInsertionPointAfter(op);
    auto expandedAllocOp = rewriter.create<memref::AllocOp>(
        op->getLoc(), op.getResultType(), op.getOutputShape());
    rewriter.replaceOp(op, expandedAllocOp);

    return success();
  }
};

void FoldAllocReshapeOp::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  patterns.insert<FoldAllocReshapeOpPattern>(patterns.getContext());
  memref::DimOp::getCanonicalizationPatterns(patterns, patterns.getContext());
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<Pass> memref::createFoldAllocReshapePass() {
  return std::make_unique<FoldAllocReshapeOp>();
}
