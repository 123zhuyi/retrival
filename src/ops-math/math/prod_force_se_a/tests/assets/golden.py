#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import numpy as np

__golden__ = {
    "kernel": {"prod_force_se_a": "prod_force_se_a_golden"},
}


def prod_force_se_a_golden(
    net_deriv, in_deriv, nlist, natoms, *, n_a_sel, n_r_sel, **kwargs
):
    """
    Golden function for prod_force_se_a.
    All the parameters (names and order) follow SE doc prototype definition without outputs.
    All the input Tensors are numpy.ndarray.

    Uses DeepMD-kit TensorFlow op_module.prod_force_se_a as the reference implementation,
    matching the competitor (DeepMD-kit) behavior exactly.

    Competitor interface:
        op_module.prod_force_se_a(net_deriv, in_deriv, nlist, natoms,
                                  n_a_sel=N, n_r_sel=N)
        Source: deepmd-kit source/op/tf/prod_force_multi_device.cc
        Kernel: deepmd-kit source/lib/src/prod_force.cc :: prod_force_a_cpu

    Args:
        net_deriv: np.ndarray, shape (nframes, nloc*nnei*4), dtype float32/float16
        in_deriv:  np.ndarray, shape (nframes, nloc*nnei*4*3), dtype float32/float16
        nlist:     np.ndarray, shape (nframes, nloc*nnei), dtype int32
        natoms:    np.ndarray, shape (2+ntypes,), dtype int32
        n_a_sel:   int, A type neighbor selection count (REQUIRED_ATTR)
        n_r_sel:   int, R type neighbor selection count (REQUIRED_ATTR)
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        atom_force: np.ndarray, shape (nframes, nall, 3), dtype same as input
    """
    from deepmd.tf.env import op_module, tf

    dtype = net_deriv.dtype
    # Competitor TF op supports float32/float64; float16 input requires precision promotion
    calc_dtype = np.float32 if dtype == np.float16 else dtype
    tf_dtype = tf.float32 if calc_dtype == np.float32 else tf.float64

    # Preprocess nlist: clip values >= nall to -1 (invalid neighbor marker)
    # DeepMD TF op does not check j_idx >= nall, causing out-of-bounds writes
    # that crash the golden process. Replace with -1 so the op safely skips them.
    nall = int(natoms[1])
    nlist_safe = nlist.copy()
    nlist_safe[nlist_safe >= nall] = -1

    net_deriv_t = tf.constant(net_deriv.astype(calc_dtype), dtype=tf_dtype)
    in_deriv_t = tf.constant(in_deriv.astype(calc_dtype), dtype=tf_dtype)
    nlist_t = tf.constant(nlist_safe, dtype=tf.int32)
    natoms_t = tf.constant(natoms, dtype=tf.int32)

    # Call DeepMD-kit ProdForceSeA TF op
    # Registration name "ProdForceSeA", Python call name prod_force_se_a
    force_t = op_module.prod_force_se_a(
        net_deriv_t,
        in_deriv_t,
        nlist_t,
        natoms_t,
        n_a_sel=n_a_sel,
        n_r_sel=n_r_sel,
    )

    # Execute computation (compatible with TF 1.x and 2.x)
    # DeepMD op returns SymbolicTensor even in TF2 eager mode, need Session to evaluate
    try:
        force = force_t.numpy()
    except AttributeError:
        with tf.compat.v1.Session() as sess:
            force = sess.run(force_t)

    # DeepMD TF op outputs shape (nframes, nall*3), reshape to (nframes, nall, 3)
    nframes = net_deriv.shape[0]
    force = force.reshape(nframes, nall, 3)

    return force.astype(dtype)
