#!/usr/bin/env python3
"""Parse msprof `op` (board mode) output for a multi-kernel Triton operator.

Layout (CANN 9.1.0, Ascend910):
    <out>/OPPROF_<ts>_<id>/<KERNEL_NAME>/0/<Metric>_<ts>.csv

Per kernel: Task Duration(us), Block Dim, per-core-type pipe ratios
(mean/max/min over active cores), resource-conflict ratios, L2 hit rate.
含有效性闸：mean(aic_time)/TaskDuration 或 mean(aiv_time)/TaskDuration > 1.1 → 标记 invalid。

usage: parse_board.py <board_dir> [--json out.json] [--quiet]
"""
import argparse, glob, json, os, sys

PIPE_CUBE = ["aic_cube_ratio", "aic_mte1_ratio", "aic_mte2_ratio",
             "aic_mte3_ratio", "aic_fixpipe_ratio", "aic_scalar_ratio"]
PIPE_VEC = ["aiv_vec_ratio", "aiv_mte2_ratio", "aiv_mte3_ratio", "aiv_scalar_ratio"]
TIME_COLS = ["aic_time(us)", "aic_scalar_single_time(us)", "aic_scalar_dual_time(us)",
             "aic_scalar_wait_time(us)", "aiv_time(us)",
             "aiv_mte2_active_bw(GB/s)", "aiv_mte3_active_bw(GB/s)"]
CFLT = ["aiv_vec_total_cflt_ratio", "aiv_vec_bankgroup_cflt_ratio",
        "aiv_vec_bank_cflt_ratio", "aiv_vec_resc_cflt_ratio", "aiv_vec_mte_cflt_ratio"]
L2 = ["aic_total_hit_rate(%)", "aiv_total_hit_rate(%)"]


def _num(v):
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def read_csv(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
    if not lines:
        return [], []
    hdr = [h.strip() for h in lines[0].split(",")]
    rows = []
    for ln in lines[1:]:
        cells = ln.split(",")
        rows.append({hdr[i]: cells[i].strip() for i in range(min(len(hdr), len(cells)))})
    return hdr, rows


def stat(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"mean": None, "max": None, "min": None}
    return {"mean": sum(vals) / len(vals), "max": max(vals), "min": min(vals)}


def _metric(kdir, name):
    """CANN 9.1.0: <Metric>_<ts>.csv；CANN 9.0.0 扁平布局: <Metric>.csv"""
    hits = glob.glob(os.path.join(kdir, f"{name}_*.csv"))
    return hits[0] if hits else (glob.glob(os.path.join(kdir, f"{name}.csv")) or [None])[0]


def parse_kernel(kdir):
    """解析单个 kernel 目录 -> row dict；无数据返回 None。

    kdir 兼容两种形态：9.1.0 的 OPPROF_*/<kernel>/0/ 与 9.0.0 的扁平 OPPROF_*/
    （后者 kernel 名取自 OpBasicInfo.csv 的 Op Name 列）。"""
    rec = {"dir": kdir, "op_name": os.path.basename(os.path.dirname(kdir))}
    # OpBasicInfo
    ob = _metric(kdir, "OpBasicInfo")
    if not ob:
        return None
    _, rows = read_csv(ob)
    if not rows:
        return None
    r0 = rows[0]
    rec["op_name"] = r0.get("Op Name") or rec["op_name"]
    rec["op_type"] = r0.get("Op Type", "")
    rec["task_duration_us"] = _num(r0.get("Task Duration(us)"))
    rec["block_dim"] = r0.get("Block Dim", "NA")
    rec["mix_block_dim"] = r0.get("Mix Block Dim", "NA")
    rec["freq"] = r0.get("Current Freq", "NA")
    if rec["task_duration_us"] is None or rec["task_duration_us"] <= 0:
        return None
    # PipeUtilization
    pu = _metric(kdir, "PipeUtilization")
    cube_rows, vec_rows = [], []
    if pu:
        _, rows = read_csv(pu)
        for r in rows:
            sub = r.get("sub_block_id", "")
            if sub.startswith("cube"):
                cube_rows.append(r)
            elif sub.startswith("vector"):
                vec_rows.append(r)
    rec["n_cube_cores"] = len(cube_rows)
    rec["n_vec_cores"] = len(vec_rows)
    # 活跃核判定：time>0 或任一 pipe ratio 非 NA（个别采集 time 记 0 但 ratio 有效，如 vector 型 gate）
    def _active(rows, cols):
        out = []
        for r in rows:
            if (_num(r.get("aic_time(us)")) or 0) > 0 or (_num(r.get("aiv_time(us)")) or 0) > 0:
                out.append(r); continue
            if any(_num(r.get(c)) is not None for c in cols):
                out.append(r)
        return out
    cube_act = _active(cube_rows, PIPE_CUBE)
    vec_act = _active(vec_rows, PIPE_VEC)
    rec["n_cube_active"] = len(cube_act)
    rec["n_vec_active"] = len(vec_act)

    def agg(rows, cols, side):
        out = {}
        for c in cols:
            out[c] = stat([_num(r.get(c)) for r in rows])
        return out

    rec["cube"] = agg(cube_act, PIPE_CUBE + [c for c in TIME_COLS if c.startswith("aic")], "cube")
    rec["vec"] = agg(vec_act, PIPE_VEC + [c for c in TIME_COLS if c.startswith("aiv")], "vec")
    rec["cube_time_us"] = {"mean": rec["cube"]["aic_time(us)"]["mean"],
                           "max": rec["cube"]["aic_time(us)"]["max"]}
    rec["vec_time_us"] = {"mean": rec["vec"]["aiv_time(us)"]["mean"],
                          "max": rec["vec"]["aiv_time(us)"]["max"]}
    # 核间不均衡（活跃核 time min->max）
    rec["cube_spread_us"] = {"min": rec["cube"]["aic_time(us)"]["min"],
                             "max": rec["cube"]["aic_time(us)"]["max"]}
    # ResourceConflictRatio
    rc = _metric(kdir, "ResourceConflictRatio")
    cflt = {}
    if rc:
        _, rows = read_csv(rc)
        for c in CFLT:
            vals = [_num(r.get(c)) for r in rows]
            vals = [v for v in vals if v is not None]
            cflt[c] = max(vals) if vals else None
    rec["conflict_max"] = cflt
    # L2Cache
    l2f = _metric(kdir, "L2Cache")
    l2 = {}
    if l2f:
        _, rows = read_csv(l2f)
        for c in L2:
            side_rows = [r for r in rows if (r.get("sub_block_id","").startswith("cube")) == c.startswith("aic")]
            l2[c] = {"mean": stat([_num(r.get(c)) for r in side_rows])["mean"],
                     "min": stat([_num(r.get(c)) for r in side_rows])["min"]}
    rec["l2_hit"] = l2
    # 有效性闸
    dur = rec["task_duration_us"]
    checks = []
    for t in (rec["cube_time_us"]["mean"], rec["vec_time_us"]["mean"]):
        if t:
            checks.append(t / dur)
    rec["validity_ratio"] = max(checks) if checks else None
    rec["valid"] = (rec["validity_ratio"] is not None and rec["validity_ratio"] <= 1.1)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("board_dir")
    ap.add_argument("--json", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    kddirs = sorted(glob.glob(os.path.join(args.board_dir, "OPPROF_*", "*", "0")))
    if not kddirs:  # CANN 9.0.0 扁平布局：每个 OPPROF_* 目录即一次 op 采集
        kddirs = sorted(glob.glob(os.path.join(args.board_dir, "OPPROF_*")))
    kernels = []
    for kd in kddirs:
        try:
            rec = parse_kernel(kd)
        except Exception as e:
            print(f"[warn] parse fail {kd}: {e}", file=sys.stderr)
            rec = None
        if rec is None:
            kernels.append({"dir": kd, "op_name": os.path.basename(kd.rstrip("/")),
                            "task_duration_us": None, "valid": False})
        else:
            kernels.append(rec)
    total = sum(k["task_duration_us"] for k in kernels if k.get("task_duration_us"))
    out = {"board_dir": args.board_dir, "total_us": total, "kernels": kernels}
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
        if not args.quiet:
            print(f"[json] {args.json}")
    if not args.quiet:
        for k in kernels:
            dur = k.get("task_duration_us")
            v = "" if k.get("valid", True) else "  ✗INVALID(x%.1f)" % (k.get("validity_ratio") or 0)
            print(f"  {k['op_name']:40s} {dur if dur else float('nan'):10.1f} us{v}")
        print(f"  {'TOTAL':40s} {total:10.1f} us")


if __name__ == "__main__":
    main()
