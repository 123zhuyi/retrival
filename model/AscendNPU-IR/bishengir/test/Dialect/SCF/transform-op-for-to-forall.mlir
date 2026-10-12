// RUN: bishengir-opt -transform-interpreter -split-input-file -allow-unregistered-dialect -verify-diagnostics %s | FileCheck %s

module attributes { transform.with_named_sequence } {
  func.func @test_for_to_forall(%arg0: tensor<?xf16>, %lb: index, %ub: index, %step: index) -> (tensor<?xf16>, tensor<?xf16>)  {
    %c0 = arith.constant 0 : index
    %dim0 = tensor.dim %arg0, %c0 : tensor<?xf16>
    %0 = tensor.empty(%dim0) : tensor<?xf16>
    %1 = tensor.empty(%dim0) : tensor<?xf16>
    // CHECK-NOT: scf.for
    // CHECK: scf.forall
    // CHECK: scf.forall.in_parallel
    // CHECK: tensor.parallel_insert_slice
    // CHECK: tensor.parallel_insert_slice
    %2:2 = scf.for %arg2 = %lb to %ub step %step iter_args(%arg3 = %0, %arg4 = %1) -> (tensor<?xf16>, tensor<?xf16>) {
      %offset = "offset_compute"(%arg2, %lb, %ub, %dim0) : (index, index, index, index) -> (index)
      %size = "size_compute"(%arg2, %lb, %ub, %dim0) : (index, index, index, index) -> (index)
      %ret = "some_use"() : () -> (tensor<?xf16>)
      %inserted_slice = tensor.insert_slice %ret into %arg3[%offset] [%size] [1] : tensor<?xf16> into tensor<?xf16>
      %inserted_slice1 = tensor.insert_slice %ret into %arg4[%offset] [%size] [1] : tensor<?xf16> into tensor<?xf16>
      scf.yield %inserted_slice, %inserted_slice1 : tensor<?xf16>, tensor<?xf16>
    }
    return %2#0, %2#1 : tensor<?xf16>, tensor<?xf16>
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %1 = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %2 = transform.loop.for_to_forall %1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

module attributes { transform.with_named_sequence } {
  func.func @test_for_to_forall_with_mapping(%arg0: tensor<?xf16>, %lb: index, %ub: index, %step: index) -> tensor<?xf16>  {
    %c0 = arith.constant 0 : index
    %dim0 = tensor.dim %arg0, %c0 : tensor<?xf16>
    %0 = tensor.empty(%dim0) : tensor<?xf16>
    // CHECK: scf.forall
    // CHECK: mapping = [#hivm.block]
    %1 = scf.for %arg2 = %lb to %ub step %step iter_args(%arg1 = %0) -> (tensor<?xf16>) {
      %offset = "offset_compute"(%arg2, %lb, %ub, %dim0) : (index, index, index, index) -> (index)
      %size = "size_compute"(%arg2, %lb, %ub, %dim0) : (index, index, index, index) -> (index)
      %ret = "some_use"() : () -> (tensor<?xf16>)
      %inserted_slice = tensor.insert_slice %ret into %arg1[%offset] [%size] [1] : tensor<?xf16> into tensor<?xf16>
      scf.yield %inserted_slice : tensor<?xf16>
    }
    return %1 : tensor<?xf16>
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %1 = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %2 = transform.loop.for_to_forall %1 {mapping = [#hivm.block]} : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

module attributes { transform.with_named_sequence } {
  func.func @test_for_to_forall(%arg0: tensor<?xf16>, %lb: index, %ub: index, %step: index) -> tensor<?xf16>  {
    %c0 = arith.constant 0 : index
    %dim0 = tensor.dim %arg0, %c0 : tensor<?xf16>
    %0 = tensor.empty(%dim0) : tensor<?xf16>
    %some_index = "some_index"() : () -> (index)
    // expected-error @+1 {{the target loop can only yield tensor.insert_slices!}}
    %1:2 = scf.for %arg2 = %lb to %ub step %step iter_args(%arg3 = %0, %arg4 = %some_index) -> (tensor<?xf16>, index) {
      %offset = "offset_compute"(%arg2, %lb, %ub, %dim0) : (index, index, index, index) -> (index)
      %size = "size_compute"(%arg2, %lb, %ub, %dim0) : (index, index, index, index) -> (index)
      %ret = "some_use"() : () -> (tensor<?xf16>)
      %inserted_slice = tensor.insert_slice %ret into %arg3[%offset] [%size] [1] : tensor<?xf16> into tensor<?xf16>
      scf.yield %inserted_slice, %offset : tensor<?xf16>, index
    }
    return %1 : tensor<?xf16>
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %1 = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %2 = transform.loop.for_to_forall %1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

module attributes { transform.with_named_sequence } {
  func.func @test_for_to_forall_with_mapping(%arg0: tensor<?xf16>, %lb: index, %ub: index, %step: index) -> tensor<?xf16>  {
    %c0 = arith.constant 0 : index
    %dim0 = tensor.dim %arg0, %c0 : tensor<?xf16>
    %0 = tensor.empty(%dim0) : tensor<?xf16>
    // CHECK-NOT: scf.forall
    %1 = scf.for %arg2 = %lb to %ub step %step iter_args(%arg1 = %0) -> (tensor<?xf16>) {
      %offset = "offset_compute"(%arg2, %lb, %ub, %dim0) : (index, index, index, index) -> (index)
      %size = "size_compute"(%arg2, %lb, %ub, %dim0) : (index, index, index, index) -> (index)
      %ret = "some_use"() : () -> (tensor<?xf16>)
      %inserted_slice = tensor.insert_slice %ret into %arg1[%offset] [%size] [1] : tensor<?xf16> into tensor<?xf16>
      scf.yield %inserted_slice : tensor<?xf16>
    // CHECK: map_for_to_forall, mapping = [#hivm.block]
    }
    return %1 : tensor<?xf16>
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %1 = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %2 = transform.loop.for_to_forall %1 {mapping = [#hivm.block], annotate_only = true} : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
