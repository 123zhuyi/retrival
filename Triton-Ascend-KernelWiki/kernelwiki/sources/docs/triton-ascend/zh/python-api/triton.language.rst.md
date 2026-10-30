---
id: doc-triton-ascend-zh-python-api-triton-language
title: triton.language
repo: triton-lang/triton-ascend
url: https://github.com/triton-lang/triton-ascend/blob/e4c692be02c38f3d5e99ccaf314bf4148fa84413/docs/zh/python-api/triton.language.rst
source_category: official-doc
tags:
- triton_ascend
- official_doc
- python-api
retrieved_at: '2026-07-15'
upstream_commit: e4c692be02c38f3d5e99ccaf314bf4148fa84413
upstream_path: docs/zh/python-api/triton.language.rst
content_sha256: 1062b5db130608c49a67dc1451d71510097dce518a9f3d8f74cce0a2e80b27f7
snapshot_body_sha256: 1062b5db130608c49a67dc1451d71510097dce518a9f3d8f74cce0a2e80b27f7
document_language: zh
source_format: rst
license: MIT
generated_by: scripts/ingest_triton_ascend_docs.py
---

:orphan:

triton.language
================

.. currentmodule:: triton.language

Programming Model
-----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    tensor
    tensor_descriptor
    program_id
    num_programs

Creation Ops
------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    arange
    cat
    full
    zeros
    zeros_like
    cast

Shape Manipulation Ops
----------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    broadcast
    broadcast_to
    expand_dims
    interleave
    join
    permute
    ravel
    reshape
    split
    trans
    view

Linear Algebra Ops
------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    dot
    dot_scaled

Memory/Pointer Ops
------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    load
    store
    make_tensor_descriptor
    load_tensor_descriptor
    store_tensor_descriptor
    make_block_ptr
    advance

Indexing Ops
------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    flip
    where
    swizzle2d
    gather

Math Ops
--------

.. autosummary::
    :toctree: generated
    :nosignatures:

    abs
    cdiv
    ceil
    clamp
    cos
    div_rn
    erf
    exp
    exp2
    fdiv
    floor
    fma
    log
    log2
    maximum
    minimum
    rsqrt
    sigmoid
    sin
    softmax
    sqrt
    sqrt_rn
    umulhi

Reduction Ops
-------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    argmax
    argmin
    max
    min
    reduce
    sum
    xor_sum

Scan/Sort Ops
-------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    associative_scan
    cumprod
    cumsum
    histogram
    sort
    topk

Atomic Ops
----------

.. autosummary::
    :toctree: generated
    :nosignatures:

    atomic_add
    atomic_and
    atomic_cas
    atomic_max
    atomic_min
    atomic_or
    atomic_xchg
    atomic_xor

Random Number Generation
------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    randint4x
    randint
    rand
    randn

Iterators
---------

.. autosummary::
    :toctree: generated
    :nosignatures:

    range
    static_range

Compiler Hint Ops
-----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    assume
    debug_barrier
    max_constancy
    max_contiguous
    multiple_of

Debug Ops
---------

.. autosummary::
    :toctree: generated
    :nosignatures:

    static_print
    static_assert
    device_print
    device_assert

Inline Assembly
---------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    inline_asm_elementwise
