# ATK system test — FusedMulAddN (`aclnnFusedMulAddN`)

Black-box accuracy/performance test of the deployed `FusedMulAddN` operator using **ATK** (Ascend Test
Kit). ATK calls the already-built operator on the NPU and compares it against a CPU golden.

- **Operator**: `y_i = x1_i * x3[0] + x2_i` (fused mul + addn; x3 is a single-element scalar broadcast)
- **dtypes**: float32 / float16 / bfloat16 / int32 / int16 · **format**: ND · **chip**: ascend910b
- **device-under-test backend**: `pyaclnn` (the aclnn L2 API `aclnnFusedMulAddN`)
- **golden**: `cpu` (custom executor — there is no single torch op for this fused FMA)

## Files

| file | role |
|---|---|
| `executor_aclnnFusedMulAddN.py` | golden `BaseApi` (`api_type: aclnn_fused_mul_add_n`): fp32 reference for floats, native two-step wraparound for ints. The pyaclnn side resolves the built-in aclnn via the json `aclnn_name: FusedMulAddN`. |
| `gen_cases.py` | generates `atk_aclnnFusedMulAddN.json` — curated cases with coupled shapes (x1/x2/y share shape; x3 is `[1]`/`[1,1]`). |
| `atk_aclnnFusedMulAddN.json` | 17 cases: 5 dtypes × {basic / 1d / single-elem / rank3 / big-multicore `[4096,1024]`} + x3 forms + x3=0/x3=1 invariants. |
| `run_atk.sh` | bridges ATK to the repo-built `custom_math` package and runs accuracy / kernel / performance. |

## Prerequisites

1. ATK installed (`atk` CLI on PATH). Wheel: `OperatorDevtools/ATK/atk-*-cp310-*.whl`.
2. The custom package built **and unpacked** so its aclnn lib + kernels are discoverable:

   ```bash
   bash build.sh --pkg --experimental --soc=ascend910b --ops=fused_mul_add_n   # from worktree root
   # unpack the .run somewhere writable; run_atk.sh expects build_out/installed_custom_math/packages
   ```

   `run_atk.sh` points `ASCEND_CUSTOM_OPP_PATH` at the **parent of `vendors/`** (with a
   `vendors/config.ini` listing `custom_math`) — ATK resolves the op's `libcust_opapi.so` from
   `${ASCEND_CUSTOM_OPP_PATH}/vendors/<name>/op_api/lib/`. (Pointing it at the vendor dir itself, or at
   the stale system `opp/vendors/custom_math`, fails — the latter holds a different op.)

## Run

```bash
bash run_atk.sh                 # accuracy: pyaclnn vs cpu golden (-cp signature check)
TASK=performance bash run_atk.sh  # perf: pyaclnn(dev0) vs npu(dev1), ATK_BIND_CPU_TYPE=2
python3 gen_cases.py            # regenerate cases
```

Results: `atk_output/<run>/report/*.xlsx` (summary / statistic / failed-cases sheets) and `log/atk.log`.

## Accuracy standard

`single_bm: high_performance`. FusedMulAddN is a **fused** mul+add: the kernel does fp32 `Muls -> Add`
(two roundings) while a torch golden may fuse to one FMA rounding, so float outputs differ by up to ~1
ULP. The fused-op standard (`high_performance`) tolerates this; `high_precision` is too tight (a fp16
big-shape case fails at exactly 1 ULP = 2⁻⁹). Integer outputs auto-switch to binary-equal.

## Known limitation — int16 via `pyaclnn`

The two int16 cases fail at `aclnnFusedMulAddNGetWorkspaceSize` with **"Cannot find binary for op
FusedMulAddN"** (integral key `int16/ND/...`), even though the int16 kernel binary and its dispatch
entry are present and correct. This is an **ATK `pyaclnn` kernel-selection limitation for the int16
dtype on this custom op**, not an operator defect: int16 is independently validated correct on real NPU
via the torch_npu aclnn ST (`tests/st/torch`, e.g. `L1_014`) and the op_kernel UT.
fp32/fp16/bf16/int32 dispatch and pass normally.
