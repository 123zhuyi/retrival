#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Generate curated ATK cases for FusedMulAddN (y = x1*x3[0] + x2).
#
# Constraints encoded directly (no random CaseGenerator needed):
#   - x1 / x2 share shape and dtype
#   - x3 is a single-element scalar tensor ([1] or [1,1]), same dtype
#
# Coverage matrix: 5 dtypes x {basic / 1d / single-elem / 3d / big-multicore} shapes,
# x3 forms ([1], [1,1]), and the x3=0 / x3=1 invariants. Default accuracy standard
# single_bm (float: rel+abs tolerance; int: auto binary-equal).
import json
import os

# FusedMulAddN is a fused mul+add: the kernel does fp32 Muls -> Add (two roundings) while a torch
# golden may fuse (FMA, one rounding), so float outputs differ by up to ~1 ULP. Per the ATK accuracy
# guidance, fused ops use single_bm:high_performance (looser, ULP-tolerant). Integer outputs auto-switch
# to binary-equal regardless of this setting.
ACC = {"acc": {"single_bm": {"type": "high_performance"}}, "perf": "not_key"}


def tensor(name, dtype, shape, rng):
    return {
        "name": name, "type": "tensor", "required": True, "dtype": dtype,
        "shape": shape, "range_values": list(rng), "backward": False,
    }


def case(cid, dtype, shape, x3_shape=(1,), rng=(-5, 5), x3_rng=None, note=""):
    x3_rng = rng if x3_rng is None else x3_rng
    c = {
        "id": cid, "name": "fused_mul_add_n",
        "api_type": "aclnn_fused_mul_add_n", "aclnn_name": "FusedMulAddN",
        "standard": ACC,
        "inputs": [
            tensor("x1", dtype, list(shape), rng),
            tensor("x2", dtype, list(shape), rng),
            tensor("x3", dtype, list(x3_shape), x3_rng),
        ],
    }
    if note:
        c["note"] = note
    return c


DTYPES = ["fp32", "fp16", "bf16", "int32", "int16"]
cases = []
cid = 0


def add(**kw):
    global cid
    cases.append(case(cid, **kw))
    cid += 1


# 1) all 5 dtypes, basic rank-2 shape, x3=[1]
for dt in DTYPES:
    add(dtype=dt, shape=(2, 3), note="basic rank2")

# 2) shape coverage (fp32): 1d / single-element / 3d / big multicore
add(dtype="fp32", shape=(8,), note="1d")
add(dtype="fp32", shape=(1,), note="single element")
add(dtype="fp32", shape=(2, 3, 4), note="rank3")
add(dtype="fp32", shape=(4096, 1024), note="big shape, multi-core")

# 3) x3 alternative form [1,1] (still single element)
add(dtype="fp32", shape=(2, 3), x3_shape=(1, 1), note="x3 form [1,1]")

# 4) invariants: x3=0 -> y==x2 ; x3=1 -> y==x1+x2
add(dtype="fp32", shape=(2, 3), x3_rng=(0, 0), note="x3=0 => y==x2")
add(dtype="fp32", shape=(2, 3), x3_rng=(1, 1), note="x3=1 => y==x1+x2")
add(dtype="int32", shape=(2, 3), x3_rng=(0, 0), note="int x3=0 => y==x2")

# 5) cast-path big shape + extra dtype/shape spread
add(dtype="fp16", shape=(4096, 1024), note="fp16 cast path, big")
add(dtype="bf16", shape=(2, 3, 4), note="bf16 cast path, rank3")
add(dtype="int32", shape=(1024,), note="int32 1d")
add(dtype="int16", shape=(2, 3, 4), note="int16 rank3")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "atk_aclnnFusedMulAddN.json")
with open(out, "w") as f:
    json.dump(cases, f, indent=1)
print(f"wrote {len(cases)} cases -> {out}")
