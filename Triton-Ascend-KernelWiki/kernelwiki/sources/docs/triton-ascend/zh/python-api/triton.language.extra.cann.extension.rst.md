---
id: doc-triton-ascend-zh-python-api-triton-language-extra-cann-extension
title: triton.language.extra.cann.extension
repo: triton-lang/triton-ascend
url: https://github.com/triton-lang/triton-ascend/blob/e4c692be02c38f3d5e99ccaf314bf4148fa84413/docs/zh/python-api/triton.language.extra.cann.extension.rst
source_category: official-doc
tags:
- triton_ascend
- official_doc
- python-api
retrieved_at: '2026-07-15'
upstream_commit: e4c692be02c38f3d5e99ccaf314bf4148fa84413
upstream_path: docs/zh/python-api/triton.language.extra.cann.extension.rst
content_sha256: 67c42fbfda0317eff0b6c5002f67f3103fe1b682f77da7c891149b4a12adb8c5
snapshot_body_sha256: 67c42fbfda0317eff0b6c5002f67f3103fe1b682f77da7c891149b4a12adb8c5
document_language: zh
source_format: rst
license: MIT
generated_by: scripts/ingest_triton_ascend_docs.py
hardware_features:
- fixpipe
---

:orphan:

triton.language.extra.cann.extension
====================================

.. currentmodule:: triton.language.extra.cann.extension

Core Types
----------

.. autosummary::
    :toctree: generated
    :nosignatures:

    scope
    ascend_address_space
    compile_hint
    multibuffer
    parallel

Data Movement
-------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    copy
    copy_from_ub_to_l1
    fixpipe

Synchronization
---------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    sync_block_all
    sync_block_set
    sync_block_wait
    debug_barrier

Vector Operations
-----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    sub_vec_id
    sub_vec_num
    conv1d

Enums
-----

.. autosummary::
    :toctree: generated
    :nosignatures:

    PIPE
    MODE
    CORE
    SYNC_IN_VF
    IteratorType
    FixpipeDMAMode
    FixpipeDualDstMode
    FixpipePreQuantMode
    FixpipePreReluMode

Vector/Memory Extension Ops
--------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    extract_slice
    insert_slice
    get_element
    sort
    flip
    cast
    index_put
    gather_out_to_ub
    scatter_ub_to_out
    index_select_simd

Custom Ops
----------

.. autosummary::
    :toctree: generated
    :nosignatures:

    custom
    custom_semantic
    register_custom_op

IR Affine Types
---------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    affine_expr
    affine_constant_expr
    affine_dim_expr
    affine_symbol_expr
    affine_binary_op_expr
    affine_map
