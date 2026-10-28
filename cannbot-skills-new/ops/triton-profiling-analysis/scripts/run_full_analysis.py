#!/usr/bin/env python3
"""Triton-Ascend 算子性能分析端到端流水线（triton-profiling-analysis 流程脚本化）。

一条命令复现完整分析：board 上板判 bound → 仿真采集 → pipe/气泡/阻塞点 →
行级证据链 → report.md。产物自包含于 <workdir>/simulator/roundsN/（N 自动递增）。

用法:
  run_full_analysis.py --impl <generated.py> --task <task.py> --workdir <op_workdir> \
      [--device 0] [--cases 1-7] [--shrink 256] [--soc Ascend910_9372] \
      [--sim-jobs 3] [--sim-timeout 1500] [--skip-sim] [--skip-board]

依赖（同仓库 skill 脚本，按相对路径自动定位）:
  triton-simulator/scripts/{dispatch_kernel_enumerate.py,collect.sh}
  triton-profiling-analysis/scripts/{parse_board.py,pipe_overlap_analysis.py,blocking_point_analysis.py}

环境：需先 source CANN set_env.sh 或设置 ASCEND_HOME_PATH；board 用真实 NPU，
仿真用 camodel（CPU）。--shrink 仅适用于「张量第 1 维为序列维 + 含 cu_seqlens」
的 varlen 任务（GDN 风格）；其他任务请勿使用。
"""
import argparse, glob, json, os, re, subprocess, sys, time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL_OPS = os.path.dirname(os.path.dirname(HERE))           # ops/
TRI_SIM = os.path.join(SKILL_OPS, "triton-simulator", "scripts")
KERNEL_ENUM = os.path.join(TRI_SIM, "dispatch_kernel_enumerate.py")
COLLECT_SH = os.path.join(TRI_SIM, "collect.sh")
PARSE_BOARD = os.path.join(HERE, "parse_board.py")
PIPE_OVERLAP = os.path.join(HERE, "pipe_overlap_analysis.py")
BLOCKING = os.path.join(HERE, "blocking_point_analysis.py")
DUAL_PIPE = os.path.join(HERE, "dual_pipeline_check.py")
DUAL_VARS = os.path.join(HERE, "dual_buffer_vars.py")
PIPE_TIME = os.path.join(HERE, "pipe_time_model.py")
DETECT_SOC = os.path.join(HERE, "detect_soc.py")
RUN_EXTRACT = os.path.join(SKILL_OPS, "triton-latency-optimizer", "scripts", "run_and_extract.sh")


def find_bishengir_compile():
    """定位 bishengir-compile（triton-ascend 自带工具链）。"""
    import shutil
    w = shutil.which("bishengir-compile")
    if w: return w
    for pat in ("/usr/local/python*/lib/python*/site-packages/triton/backends/ascend/bishengir/bin/bishengir-compile",
                os.path.join(sys.prefix, "lib/python*/site-packages/triton/backends/ascend/bishengir/bin/bishengir-compile")):
        hits = glob.glob(pat)
        if hits: return hits[0]
    return None


# pass 级变换验证目标（双流水相关；pass 名以 --mlir-print-ir-after-all 全量 dump 实测为准）
DUAL_PIPE_PASSES = ("hivm-mark-multi-buffer", "cv-pipelining")  # mark=真正写 hivm.multi_buffer 注解的 pass；enable 不改 IR 无信息量


def find_ir_analysis_tool(name="diff_pass.sh"):
    """IR 分析工具（已从 triton-ir-analysis skill 合并进本 skill scripts/）。"""
    local = os.path.join(HERE, name)
    return local if os.path.exists(local) else None


def replay_last_pass(binc, ttadapter, out_mlir, multibuffer=True):
    """A/B 重放：与 run_and_extract.sh 完全相同的 flag 集，仅翻转
    --enable-auto-multi-buffer——对照 last_pass.mlir（False 口径）即可从 IR 结构
    判断双流水是否真正落地（槽位/分配差异）。抽最后一个 'IR Dump After' 块。"""
    target = os.environ.get("BISHENGIR_TARGET", "Ascend910_9372")
    help_txt = subprocess.run([binc, "--help"], capture_output=True, text=True).stdout or ""
    if "mlir-print-ir-after-all" in help_txt:
        pflag = "--mlir-print-ir-after-all"
    elif "print-after-all" in help_txt:
        pflag = "--print-after-all"
    else:
        return False
    cmd = [binc, f"--target={target}",
           f"--enable-auto-multi-buffer={'True' if multibuffer else 'False'}",
           "--enable-auto-bind-sub-block=True", "--enable-hfusion-compile=true",
           "--enable-hivm-compile=true", "--enable-triton-kernel-compile=true",
           pflag, "--mlir-print-debuginfo", os.path.basename(ttadapter)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(ttadapter))
    txt = (r.stdout or "") + (r.stderr or "")
    open(out_mlir + ".full", "w").write(txt)  # 全量 dump：双地址 pointer_cast 在中间 pass
    lines = txt.splitlines(keepends=True)
    last = None
    for i, l in enumerate(lines):
        if "IR Dump After" in l: last = i
    if last is None:
        if txt.strip():
            open(out_mlir, "w").write(txt); return True
        return False
    open(out_mlir, "w").write("".join(lines[last:]))
    return True

DRIVER = '''#!/usr/bin/env python3
"""自动生成：采集驱动（单 case 单次 forward）。env: CASE_IDX / T_SHRINK。"""
import os, importlib.util, torch
import torch_npu  # noqa
RD = os.path.dirname(os.path.abspath(__file__))
TASK = os.environ.get("GDN_TASK", {task!r})
CASE = int(os.environ.get("CASE_IDX", "1"))
T_SHRINK = int(os.environ.get("T_SHRINK", "0"))

def _load(p, n):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

task = _load(TASK, "task")
impl = _load(os.path.join(RD, {impl_name!r}), "impl")
dev = torch.device("npu")
torch.manual_seed(0); torch.npu.manual_seed(0)
groups = getattr(task, "get_input_groups", None)
inputs = list(groups()[CASE - 1]) if groups else list(task.get_inputs())
if T_SHRINK > 0:
    # GDN 风格 varlen：dim1 为序列维的张量切片 + 重建单段 cu_seqlens
    new = []
    for t in inputs:
        if torch.is_tensor(t) and t.dim() >= 2 and t.shape[1] >= T_SHRINK:
            new.append(t[:, :T_SHRINK].contiguous())
        else:
            new.append(t)
    inputs = new
    for i, t in enumerate(inputs):
        if torch.is_tensor(t) and t.dtype == torch.int32 and t.dim() == 1 and t.numel() >= 2:
            inputs[i] = torch.tensor([0, T_SHRINK], dtype=torch.int32)
inputs = [t.to(dev) if torch.is_tensor(t) else t for t in inputs]
mdl = impl.ModelNew().to(dev)
out = mdl(*inputs)
torch.npu.synchronize()
print("DONE case%d shrink=%d" % (CASE, T_SHRINK))
'''

VERIFY = '''#!/usr/bin/env python3
"""自动生成：策略B复验——full 与 T_SHRINK 的 kernel 集合 + 各 kernel launch_count 对比。
用法: shrink_verify.py <T_SHRINK>；输出 LAUNCHES=<json> 行。"""
import os, sys, json, collections, importlib.util
import torch
import torch_npu  # noqa
RD = os.path.dirname(os.path.abspath(__file__))
TASK = os.environ.get("GDN_TASK", {task!r})
CASE = int(os.environ.get("CASE_IDX", "1"))
T_SHRINK = int(sys.argv[1]) if len(sys.argv) > 1 else 0

def _load(p, n):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

task = _load(TASK, "task")
impl = _load(os.path.join(RD, {impl_name!r}), "impl")
from triton.runtime import jit as triton_jit
rec = collections.Counter()
_orig = triton_jit.JITFunction.run
def _patched(self, *args, grid=None, warmup=False, **kwargs):
    if not warmup:
        rec[self.__name__] += 1
    return _orig(self, *args, grid=grid, warmup=warmup, **kwargs)
triton_jit.JITFunction.run = _patched

dev = torch.device("npu")
torch.manual_seed(0); torch.npu.manual_seed(0)
groups = getattr(task, "get_input_groups", None)
inputs = list(groups()[CASE - 1]) if groups else list(task.get_inputs())
if T_SHRINK > 0:
    new = []
    for t in inputs:
        if torch.is_tensor(t) and t.dim() >= 2 and t.shape[1] >= T_SHRINK:
            new.append(t[:, :T_SHRINK].contiguous())
        else:
            new.append(t)
    inputs = new
    for i, t in enumerate(inputs):
        if torch.is_tensor(t) and t.dtype == torch.int32 and t.dim() == 1 and t.numel() >= 2:
            inputs[i] = torch.tensor([0, T_SHRINK], dtype=torch.int32)
inputs = [t.to(dev) if torch.is_tensor(t) else t for t in inputs]
mdl = impl.ModelNew().to(dev)
out = mdl(*inputs)
torch.npu.synchronize()
print("LAUNCHES=" + json.dumps(dict(rec), sort_keys=True))
'''

def sh(cmd, cwd=None, env=None, log=None, check=True):
    e = dict(os.environ); e.update(env or {})
    print(f"  $ {' '.join(str(c) for c in cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=cwd, env=env and e or None,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if log:
        open(log, "w").write(r.stdout)
    if check and r.returncode != 0:
        print(f"  ✗ exit={r.returncode}（日志 {log}）", flush=True)
    return r

def next_round_dir(simroot):
    n = 0
    for d in glob.glob(os.path.join(simroot, "rounds*")):
        m = re.match(r"rounds(\d+)$", os.path.basename(d))
        if m: n = max(n, int(m.group(1)))
    rd = os.path.join(simroot, f"rounds{n+1}")
    os.makedirs(os.path.join(rd, "log"), exist_ok=True)
    return rd, n + 1

# ---------------------------------------------------------------- board 侧

def bound_of(ratios):
    """判定表：占比最高且 >50%（SCALAR 特例 >30%）。ratios={ratio键: mean}。
    返回 pipe=短名（如 scalar），key=原始 CSV 键（如 aic_scalar_ratio，供 TRACE_MAP 查询）。"""
    items = sorted(((k, v) for k, v in ratios.items() if v is not None), key=lambda x: -x[1])
    if not items: return None
    key, val = items[0]
    short = key.replace("aic_", "").replace("aiv_", "").replace("_ratio", "")
    thr = 0.30 if "scalar" in short else 0.50
    return {"pipe": short, "key": key, "pct": val, "is_bound": val > thr, "top2": items[:2]}

# board pipe 名 → trace pipe 名
TRACE_MAP = {
    "aic_cube_ratio": ["CUBE"], "aic_scalar_ratio": ["FLOWCTRL", "SCALAR"],
    "aic_mte1_ratio": ["MTE1"], "aic_mte2_ratio": ["MTE2"], "aic_mte3_ratio": ["MTE3"],
    "aic_fixpipe_ratio": ["FIXP"],
    "aiv_vec_ratio": ["VECTOR"], "aiv_scalar_ratio": ["SCALAR", "SCALARLDST"],
    "aiv_mte2_ratio": ["MTE2"], "aiv_mte3_ratio": ["MTE3"],
}

def board_bounds(summary):
    """board_c*_summary.json → per kernel bound 判定。只取 *_ratio 列（防绝对时长污染排序）。"""
    out = {}
    for k in summary["kernels"]:
        if k.get("task_duration_us") is None: continue
        cube = {kk: (vv or {}).get("mean") for kk, vv in (k.get("cube") or {}).items() if kk.endswith("_ratio")}
        vec = {kk: (vv or {}).get("mean") for kk, vv in (k.get("vec") or {}).items() if kk.endswith("_ratio")}
        out[k["op_name"]] = {
            "dur": k["task_duration_us"], "block_dim": k.get("block_dim"),
            "cube": bound_of(cube), "vec": bound_of(vec),
            "conflict": (k.get("conflict_max") or {}).get("aiv_vec_total_cflt_ratio"),
            "l2": {kk: (vv or {}).get("mean") for kk, vv in (k.get("l2_hit") or {}).items()},
            "cube_spread": k.get("cube_spread_us"),
        }
    return out

# ---------------------------------------------------------------- trace 侧

def is_wait(n):
    return n and (n.startswith("WAIT_FLAG") or n == "BAR")

def pipe_line_attrib(trace_files, srclines):
    """{pipe: {line: [exec_us, wait_us]}}，合并给定 trace 文件。"""
    out = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    for f in trace_files:
        d = json.load(open(f))
        pid2name = {str(e["pid"]): e["args"]["name"] for e in d["traceEvents"]
                    if e.get("ph") == "M" and e.get("name") == "process_name" and e.get("args", {}).get("name")}
        for e in d["traceEvents"]:
            if e.get("ph") != "X": continue
            pipe = pid2name.get(str(e.get("pid")), str(e.get("tid", "?")))
            code = (e.get("args") or {}).get("code") or ""
            ln = None
            for l in code.splitlines():
                if l and not l.endswith("internal:0"):
                    ln = l; break
            if not ln: continue
            try: ln = int(ln.rsplit(":", 1)[-1])
            except ValueError: continue
            slot = out[pipe][ln]
            slot[1 if is_wait(e.get("name")) else 0] += e.get("dur") or 0
    return out

def top_lines(pipe_map, pipes, srclines, n=3):
    agg = defaultdict(lambda: [0.0, 0.0])
    for p in pipes:
        for ln, v in pipe_map.get(p, {}).items():
            agg[ln][0] += v[0]; agg[ln][1] += v[1]
    rows = sorted(agg.items(), key=lambda kv: -(kv[1][0] + kv[1][1]))[:n]
    def stmt(ln):
        if 1 <= ln <= len(srclines):
            s = srclines[ln - 1].strip()
            return s if len(s) <= 70 else s[:69] + "…"
        return ""
    return [{"line": ln, "src": stmt(ln), "exec_us": round(v[0], 1),
             "wait_us": round(v[1], 1), "total_us": round(v[0] + v[1], 1)} for ln, v in rows]

def blocking_lines(trace_file, srclines, top=4):
    r = subprocess.run([sys.executable, BLOCKING, trace_file],
                       capture_output=True, text=True)
    txt = r.stdout
    def grab(section, n):
        if section not in txt: return []
        seg = txt.split(section, 1)[1]
        seg = seg.split("\n--")[0]
        return [l.strip() for l in seg.splitlines() if l.strip().startswith(("CUBE", "FIXP", "MTE", "VEC", "SCALAR", "FLOWCTRL"))][:n]
    return {"exec": grab("按执行 pipe + 代码行", top), "wait": grab("按等待 pipe + 等待行", top)}

# ---------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--workdir", required=True, help="算子工作目录（simulator/roundsN 在其下创建）")
    ap.add_argument("--device", default="0")
    ap.add_argument("--cases", default="1-7")
    ap.add_argument("--shrink", type=int, default=0, help="策略B T 收缩（0=不收缩，full case 仿真）")
    ap.add_argument("--soc", default=None,
                    help="仿真 SOC 型号；缺省自动探测（detect_soc.py → 驱动权威值），"
                         "失败回退环境变量 SOC_VERSION")
    ap.add_argument("--sim-jobs", type=int, default=3)
    ap.add_argument("--sim-timeout", type=int, default=1500)
    ap.add_argument("--skip-sim", action="store_true")
    ap.add_argument("--skip-board", action="store_true")
    ap.add_argument("--reuse-round", default=None,
                    help="复用既有轮次目录（如 simulator/rounds2）：跳过全部采集，"
                         "仅重跑分析+证据链+报告（覆盖该目录 analysis_*/evidence/report）")
    ap.add_argument("--label", default="")
    ap.add_argument("--sketch", default=None,
                    help="agent 推导的计算草图 kernel_sketch.json（缺省探测 <workdir>/kernel_sketch.json；"
                         "缺失时 §2.5 只输出任务下限口径并标注实现量未推导）")
    a = ap.parse_args()
    if not a.soc:
        r = subprocess.run([sys.executable, DETECT_SOC], capture_output=True, text=True)
        a.soc = (r.stdout or "").strip() or os.environ.get("SOC_VERSION") or "Ascend910_9372"
        print(f"[soc] {a.soc}（{'自动探测' if r.stdout.strip() else '环境/默认'}）", flush=True)

    impl = os.path.abspath(a.impl); task = os.path.abspath(a.task)
    simroot = os.path.abspath(os.path.join(a.workdir, "simulator"))  # abs：防相对 workdir 下子进程 cwd=rd 时相对路径双重嵌套
    os.makedirs(simroot, exist_ok=True)
    reuse = bool(a.reuse_round)
    if reuse:
        rd = os.path.abspath(a.reuse_round)
        m = re.match(r"rounds(\d+)$", os.path.basename(rd))
        if not m: sys.exit(f"[reuse] 目录名须为 roundsN：{rd}")
        rnum = int(m.group(1)); tag = f"round{rnum}"
        cands = [f for f in os.listdir(rd) if f.endswith(".py")
                 and f not in ("prof_driver.py", "shrink_verify.py", "parse_board.py")]
        if not cands: sys.exit(f"[reuse] {rd} 内未找到被测实现 .py")
        impl_name = cands[0]; impl = os.path.join(rd, impl_name)
        a.skip_board = True; a.skip_sim = True
        print(f"[reuse] {rd}（impl={impl_name}），跳过采集仅重分析", flush=True)
    else:
        rd, rnum = next_round_dir(simroot)
        tag = f"round{rnum}"
        impl_name = os.path.basename(impl)
        print(f"[setup] rounds{rnum}: {rd}", flush=True)

    # 阶段 0：自包含（reuse 模式跳过——impl 已在轮目录内）
    if not reuse:
        subprocess.run(["cp", impl, os.path.join(rd, impl_name)], check=True)
        with open(os.path.join(rd, "prof_driver.py"), "w") as f:
            f.write(DRIVER.format(task=task, impl_name=impl_name))
        os.chmod(os.path.join(rd, "prof_driver.py"), 0o755)
        with open(os.path.join(rd, "shrink_verify.py"), "w") as f:
            f.write(VERIFY.format(task=task, impl_name=impl_name))
        os.chmod(os.path.join(rd, "shrink_verify.py"), 0o755)
    import hashlib
    md5 = hashlib.md5(open(os.path.join(rd, impl_name), "rb").read()).hexdigest()[:12]
    if not reuse:
        json.dump({"round": f"rounds{rnum}", "label": a.label, "impl": impl, "task": task,
                   "code_md5": md5, "device": a.device, "cases": a.cases, "shrink": a.shrink,
                   "soc": a.soc, "ts": time.strftime("%Y-%m-%d %H:%M:%S")},
                  open(os.path.join(rd, "round_meta.json"), "w"), ensure_ascii=False, indent=1)
    env = {"ASCEND_RT_VISIBLE_DEVICES": a.device, "TRITON_NPU_DEVICE_ID": a.device,
           "TRITON_CACHE_DIR": os.path.join(rd, ".triton_cache"), "SOC_VERSION": a.soc}

    # 阶段 1：board 采集 + 解析
    cases = []
    for part in a.cases.split(","):
        if "-" in part:
            lo, hi = part.split("-"); cases.extend(range(int(lo), int(hi) + 1))
        else: cases.append(int(part))
    kernels_bound = None
    if not a.skip_board:
        for c in cases:
            outd, js = f"board_c{c}", os.path.join(rd, f"board_c{c}_summary.json")
            if os.path.exists(js): print(f"[board] c{c} 已存在，跳过"); continue
            print(f"[board] c{c} 采集中...", flush=True)
            sh(["msprof", "op", "--application=./prof_driver.py", f"--output={outd}",
                "--aic-metrics=PipeUtilization,ResourceConflictRatio,L2Cache,MemoryDetail", "--launch-count=8"],
               cwd=rd, env={**env, "CASE_IDX": str(c)}, log=os.path.join(rd, "log", f"board_c{c}.log"), check=False)
            sh([sys.executable, PARSE_BOARD, outd, "--json", js], cwd=rd,
               log=os.path.join(rd, "log", f"parse_c{c}.log"), check=False)
        # c1 判定表
        c1 = json.load(open(os.path.join(rd, f"board_c{cases[0]}_summary.json")))
        kernels_bound = board_bounds(c1)
        print(f"[board] c{cases[0]} total={c1['total_us']:.1f}µs；逐 kernel bound：")
        def _fmt_side(bd):
            if not bd: return "—"
            return f"{bd['pipe']} {bd['pct']*100:.0f}%{'*' if bd['is_bound'] else ''}"
        for kn, b in kernels_bound.items():
            print(f"  {kn}: {b['dur']:.1f}µs cube[{_fmt_side(b['cube'])}] vec[{_fmt_side(b['vec'])}]")
    else:
        # reuse / skip-board：summary 缺失才重解析；已存在且完整的 summary（含人工补齐的
        # gap-fill）不覆盖——避免把修复后的判定冲掉
        for c in cases:
            outd = os.path.join(rd, f"board_c{c}")
            js = os.path.join(rd, f"board_c{c}_summary.json")
            need = True
            if os.path.exists(js):
                try:
                    _d = json.load(open(js))
                    need = any(k.get("task_duration_us") is None for k in _d.get("kernels", []))
                except Exception:
                    need = True
            if need and os.path.isdir(outd):
                sh([sys.executable, PARSE_BOARD, f"board_c{c}", "--json", f"board_c{c}_summary.json"],
                   cwd=rd, log=os.path.join(rd, "log", f"parse_c{c}.log"), check=False)
        c1 = json.load(open(os.path.join(rd, f"board_c{cases[0]}_summary.json")))
        kernels_bound = board_bounds(c1)

    # 阶段 1.5：kernel 保留决策（上板已测全部 kernel；最多保留两个）
    # 占比口径：跨全部已采 board case 聚合 Σtask_duration_us / Σtotal_us。
    # 规则：存在占比 >50% 的 kernel → 只保留该一个；否则按占比保留 top-2。
    # 之后所有分析（仿真采集 / IR 补齐与重放 / 证据链 / 报告分析节）仅针对保留 kernel。
    kernels_bound_all = dict(kernels_bound or {})
    shares, tot_all = {}, 0.0
    for c in cases:
        p = os.path.join(rd, f"board_c{c}_summary.json")
        if not os.path.exists(p):
            continue
        try:
            s = json.load(open(p))
        except Exception:
            continue
        tot_all += s.get("total_us") or 0.0
        for k in s.get("kernels", []):
            shares[k["op_name"]] = shares.get(k["op_name"], 0.0) + (k.get("task_duration_us") or 0.0)
    if tot_all > 0 and shares:
        for kn_ in shares:
            shares[kn_] = shares[kn_] / tot_all
        dom = max(shares, key=shares.get)
        kept_kernels = [dom] if shares[dom] > 0.5 else \
            [kn_ for kn_, _ in sorted(shares.items(), key=lambda x: -x[1])[:2]]
    else:
        kept_kernels = list(kernels_bound or {})
    dropped = sorted(set(kernels_bound_all) - set(kept_kernels))
    kernels_bound = {kn_: v for kn_, v in kernels_bound.items() if kn_ in kept_kernels}
    json.dump({"shares": {k: round(v, 4) for k, v in shares.items()},
               "kept": kept_kernels, "dropped": dropped,
               "rule": "share>50% -> keep 1; else keep top-2"},
              open(os.path.join(rd, "kernel_keep_decision.json"), "w"),
              ensure_ascii=False, indent=1)
    if kept_kernels:
        print(f"[keep] kernel 保留决策（跨case聚合占比）：保留 {kept_kernels}；"
              f"过滤 {dropped or '无'}；后续分析仅针对保留 kernel", flush=True)

    # 阶段 2：kernel 枚举（硬门禁：状态表须含全部枚举 kernel，含未触发）
    r = sh([sys.executable, KERNEL_ENUM, os.path.join(rd, impl_name)], log=os.path.join(rd, "log", "kernels.log"))
    all_kernels = [l.split()[0] for l in r.stdout.splitlines()
                   if l.strip().startswith("_") and "(" not in l.split()[0]]
    # 名称归一：board 侧运行期名可能带 _mix_aic 后缀，与源码名对齐后再判未触发
    triggered_src_names = {k.replace("_mix_aic", "") for k in (kernels_bound or {})}
    untriggered = sorted(set(all_kernels) - triggered_src_names)

    # 阶段 2b：策略B 复验闸门（--shrink>0 时，仿真前必须通过）
    if not a.skip_sim and a.shrink > 0:
        vf = os.path.join(rd, "shrink_verify.py")
        r1 = sh([sys.executable, vf, "0"], cwd=rd, env=env,
                log=os.path.join(rd, "log", "shrink_verify_full.log"), check=False)
        r2 = sh([sys.executable, vf, str(a.shrink)], cwd=rd, env=env,
                log=os.path.join(rd, "log", "shrink_verify_shrunk.log"), check=False)
        lfull = next((l for l in r1.stdout.splitlines() if l.startswith("LAUNCHES=")), "")
        lshrk = next((l for l in r2.stdout.splitlines() if l.startswith("LAUNCHES=")), "")
        if lfull and lfull == lshrk:
            open(os.path.join(rd, "shrink_verify_ok"), "w").write(lfull)
            print(f"[verify] 策略B复验通过：{lfull}", flush=True)
        else:
            json.dump({"full": lfull, "shrunk": lshrk},
                      open(os.path.join(rd, "shrunk_rejected.json"), "w"))
            sys.exit(f"[verify] 策略B复验失败（发射路径不一致），中止仿真。\n  full  ={lfull}\n  shrunk={lshrk}")

    # 阶段 3：仿真采集（逐 kernel collect.sh）
    sim_kernels = list(kernels_bound or [])  # board 上实际出现（有时长）的 kernel
    if not a.skip_sim:
        procs = []
        env_sim = {**env, "CASE_IDX": str(cases[0]), "T_SHRINK": str(a.shrink)}
        for kn in sim_kernels:
            meta = {"case_idx": cases[0], "kernel": kn, "code_md5": md5, "round": f"rounds{rnum}",
                    "sim_mode": "shrunk" if a.shrink else "real"}
            mp = os.path.join(rd, f"meta__{kn}.json")
            json.dump(meta, open(mp, "w"), ensure_ascii=False)
            logp = os.path.join(rd, "log", f"sim_{kn}.log")
            print(f"[sim] {kn} 采集启动", flush=True)
            e = dict(os.environ); e.update(env_sim)
            procs.append((kn, subprocess.Popen(
                ["bash", COLLECT_SH, "--case", "prof_driver.py", "--kernel", kn,
                 "--tag", tag, "--timeout", str(a.sim_timeout), "--meta", mp],
                cwd=rd, env=e, stdout=open(logp, "w"), stderr=subprocess.STDOUT)))
            while sum(1 for _, p in procs if p.poll() is None) >= a.sim_jobs:
                time.sleep(10)
        for kn, p in procs:
            p.wait()
            st = glob.glob(os.path.join(rd, f"{tag}__{kn}", "capture_status.json"))
            v = json.load(open(st[0]))["verdict"] if st else "missing"
            print(f"[sim] {kn}: {v}", flush=True)

    # 阶段 3.5：IR 提取（委托 triton-latency-optimizer run_and_extract.sh，best-effort）
    # 产出 <round>/ir_dump/<kernel>_{ttir,ttadapter,last_pass}.mlir；last_pass 含 PlanMemory
    # pointer_cast 字面字节偏移（缓冲地址层）。部分 kernel 可能在强制重编译的严格环境下
    # 触发 aicore exception 而缺失——容忍，缺失者回落 cache 旋钮证据。
    ir_dump = os.path.join(rd, "ir_dump")
    if not reuse and not glob.glob(os.path.join(ir_dump, "*_last_pass.mlir")):
        binc = find_bishengir_compile()
        if binc:
            print(f"[ir] run_and_extract.sh 提取中（bishengir={binc}）", flush=True)
            env_ir = {**os.environ, **env, "CASE_IDX": str(cases[0]), "T_SHRINK": "0",
                      "BISHENGIR_COMPILE": binc, "IR_OUTPUT_DIR": ir_dump,
                      "TRITON_CACHE_DIR": os.path.join(rd, ".triton_cache_ir")}
            subprocess.run(["bash", RUN_EXTRACT, "prof_driver.py"], cwd=rd, env=env_ir,
                           stdout=open(os.path.join(rd, "log", "ir_extract.log"), "w"),
                           stderr=subprocess.STDOUT)
            n_lp = len(glob.glob(os.path.join(ir_dump, "*_last_pass.mlir")))
            print(f"[ir] 提取完成：{n_lp} 个 kernel 的 last_pass.mlir（best-effort，失败者见日志）", flush=True)
        else:
            print("[ir] 未定位到 bishengir-compile，跳过 IR 提取", flush=True)

    dp_sh = find_ir_analysis_tool("diff_pass.sh")
    fv_sh = find_ir_analysis_tool("find_variants.py")
    # 回落：ir_dump 缺失的 kernel（如严格重编译环境下 aicore exception 的）
    # 从 triton cache 补 ttadapter——diff_pass/重放只需 ttadapter 文件，纯编译不上板，
    # 不受设备端故障影响。多变体时选 multibuffer=True（设备口径）中最新的。
    src_cache = os.path.join(rd, ".triton_cache")
    have = {os.path.basename(pp).replace("_ttadapter.mlir", "")
            for pp in glob.glob(os.path.join(ir_dump, "*_ttadapter.mlir"))}
    for kn in (kn for kn in (kernels_bound or {})
               if kn.replace("_mix_aic", "") in {x.replace("_mix_aic", "") for x in kept_kernels}):
        src_name = kn.replace("_mix_aic", "")
        if src_name in have: continue
        cands = []
        for f in glob.glob(os.path.join(src_cache, "**", f"{src_name}.ttadapter"), recursive=True):
            jf = os.path.join(os.path.dirname(f), src_name + ".json")
            try: mb = json.load(open(jf)).get("multibuffer")
            except Exception: mb = None
            cands.append((f, mb, os.path.getmtime(f)))
        mb_t = [c for c in cands if c[1] is True] or cands
        if mb_t:
            pick = max(mb_t, key=lambda c: c[2])
            dst = os.path.join(ir_dump, f"{src_name}_ttadapter.mlir")
            subprocess.run(["cp", pick[0], dst], check=False)
            # 变体审计留痕（triton-ir-analysis find_variants.py，供人工复核挑选）
            if fv_sh:
                subprocess.run([sys.executable, fv_sh, src_name, src_cache],
                               stdout=open(os.path.join(rd, "log", f"ir_variants_{src_name}.txt"), "w"),
                               stderr=subprocess.STDOUT)
            print(f"[ir] 回落补齐 {src_name}: ttadapter ← cache"
                  f"（{len(cands)} 变体中选 multibuffer=True 最新，审计见 log/ir_variants_{src_name}.txt）", flush=True)

    # A/B 重放（幂等，reuse 也会补齐）：翻转 multibuffer=True 生成对照件
    kept_src = {x.replace("_mix_aic", "") for x in kept_kernels}
    binc = find_bishengir_compile()
    if binc:
        for tta in sorted(glob.glob(os.path.join(ir_dump, "*_ttadapter.mlir"))):
            kn = os.path.basename(tta).replace("_ttadapter.mlir", "")
            if kn not in kept_src:
                continue  # kernel 保留决策：非保留 kernel 不做重放/切片
            off_out = os.path.join(ir_dump, f"{kn}_last_pass.mlir")
            mb_out = os.path.join(ir_dump, f"{kn}_last_pass_mb.mlir")
            if not os.path.exists(off_out):  # run_and_extract 崩掉的 kernel 无 off 侧，补 False 重放
                replay_last_pass(binc, tta, off_out, multibuffer=False)
            if os.path.exists(mb_out): continue
            ok = replay_last_pass(binc, tta, mb_out, multibuffer=True)
            print(f"[ir] A/B 重放 {kn}: multibuffer=True {'OK' if ok else 'FAIL'}", flush=True)

    # pass 级变换验证（委托 triton-ir-analysis skill 的 diff_pass.sh）：
    # --mlir-print-ir-before/after=<pass> 直接判定 pass 转没转（比旋钮 A/B 更精确）
    if dp_sh:
        print(f"[ir] diff_pass.sh: {dp_sh}", flush=True)
        for tta in sorted(glob.glob(os.path.join(ir_dump, "*_ttadapter.mlir"))):
            kn = os.path.basename(tta).replace("_ttadapter.mlir", "")
            if kn not in kept_src:
                continue  # kernel 保留决策：非保留 kernel 不做 pass 切片
            out_txt = os.path.join(ir_dump, f"{kn}_pass_diff.txt")
            if os.path.exists(out_txt): continue
            with open(out_txt, "w") as fh:
                for p in DUAL_PIPE_PASSES:
                    fh.write(f"===== pass={p}\n")
                    r = subprocess.run(["bash", dp_sh, tta, p], capture_output=True, text=True)
                    fh.write(r.stdout or "")
            print(f"[ir] pass diff {kn}: 完成（{', '.join(DUAL_PIPE_PASSES)}）", flush=True)
    else:
        print("[ir] 未找到 triton-ir-analysis 的 diff_pass.sh（skill 未同步到本 checkout？），跳过 pass 级验证", flush=True)

    # 阶段 4：分析（pipe 四口径 + 阻塞点 + 行级证据链）
    srclines = open(os.path.join(rd, impl_name)).readlines()
    evidence = {}
    for kn in sim_kernels:
        simdirs = glob.glob(os.path.join(rd, f"{tag}__{kn}", "OPPROF_*", "simulator"))
        if not simdirs: continue
        sd = simdirs[0]
        sh([sys.executable, PIPE_OVERLAP, "--dir", sd, "--json", os.path.join(rd, f"analysis_{kn}.json")],
           log=os.path.join(rd, f"analysis_{kn}.txt"), check=False)
        # 双流水生效性判定（IR 意图 × trace 实效）
        dp_json = os.path.join(rd, f"dual_pipeline__{kn}.json")
        ir_dir = os.path.join(rd, ".triton_cache")
        sh([sys.executable, DUAL_PIPE, "--dir", sd, "--ir-dir", ir_dir,
            "--ir-dump-dir", ir_dump,
            "--kernel", kn.replace("_mix_aic", ""), "--json", dp_json],
           log=os.path.join(rd, "log", f"dual_pipeline_{kn}.log"), check=False)
        try:
            dp_result = json.load(open(dp_json))
        except Exception:
            dp_result = None
        # 双流水源码变量统计（IR 双地址判据，与 trace 掩盖度交叉验证）
        full_mb = os.path.join(ir_dump, f"{kn.replace('_mix_aic', '')}_last_pass_mb.mlir.full")
        dbv_data = None
        if os.path.exists(full_mb):
            dbv_json = os.path.join(rd, f"dual_buffer_vars__{kn}.json")
            sh([sys.executable, DUAL_VARS, full_mb, "--src", os.path.join(rd, impl_name),
                "--quiet", "--json", dbv_json],
               log=os.path.join(rd, "log", f"dual_buffer_vars_{kn}.log"), check=False)
            try:
                dbv_data = list(json.load(open(dbv_json)).values())[0]
            except Exception:
                dbv_data = None
        traces = sorted(glob.glob(os.path.join(sd, "core0.*", "trace.json")))
        pm = pipe_line_attrib(traces, srclines)
        b = kernels_bound.get(kn, {})
        chain = {"dur": b.get("dur"), "bound": [], "blocking": {},
                 "dual_pipeline": dp_result, "dual_buffer_vars": dbv_data}
        for side in ("cube", "vec"):
            bd = b.get(side)
            if not bd: continue
            tpipes = TRACE_MAP.get(bd.get("key", ""), [bd["pipe"].upper()])
            chain["bound"].append({"side": side, "board_pipe": bd["pipe"],
                                   "board_pct": round(bd["pct"] * 100, 1), "is_bound": bd["is_bound"],
                                   "trace_pipes": tpipes,
                                   "top_lines": top_lines(pm, tpipes, srclines)})
        # 阻塞点：cube 核优先，vector 型 kernel 用 veccore0
        for sub in ("cubecore0", "veccore0"):
            tf = os.path.join(sd, f"core0.{sub}", "trace.json")
            if os.path.exists(tf):
                chain["blocking"][sub] = blocking_lines(tf, srclines)
        evidence[kn] = chain
        print(f"[analyze] {kn} 完成", flush=True)
    json.dump(evidence, open(os.path.join(rd, "evidence_chains.json"), "w"), ensure_ascii=False, indent=1)

    # 管线理论时间模型（前置 bound 综合判断）：CUBE/VECTOR 理论下限 + 搬运实测
    ptm_json = os.path.join(rd, "pipe_time_model.json")
    board_c1_js = os.path.join(rd, "board_c1_summary.json")
    kn_src = (kept_kernels[0].replace("_mix_aic", "") if kept_kernels else None)
    sketch_path = a.sketch or os.path.join(a.workdir, "kernel_sketch.json")
    sketch_use = sketch_path if os.path.exists(sketch_path) else None
    ir_ttir = (os.path.join(rd, "ir_dump", f"{kept_kernels[0]}_ttir.mlir")
               if kept_kernels and os.path.exists(
                   os.path.join(rd, "ir_dump", f"{kept_kernels[0]}_ttir.mlir")) else None)
    r = sh([sys.executable, PIPE_TIME, "--task", task, "--cases", a.cases, "--json", ptm_json]
           + (["--board", board_c1_js] if os.path.exists(board_c1_js) else [])
           + (["--kernel-name", kn_src] if kn_src else [])
           + (["--sketch", sketch_use, "--expect-md5", md5[:10]] if sketch_use else [])
           + (["--ir", ir_ttir] if ir_ttir else []),
           log=os.path.join(rd, "log", "pipe_time_model.log"), check=False)

    # 片上层实现量清单（last_pass intrinsic，用户指正固化的第二层口径）
    lp_mlir = (os.path.join(rd, "ir_dump", f"{kept_kernels[0]}_last_pass.mlir")
               if kept_kernels else None)
    lp_inv = os.path.join(rd, "last_pass_inventory.json")
    if lp_mlir and os.path.exists(lp_mlir):
        sh([sys.executable,
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_pass_inventory.py"),
            lp_mlir, "--json", lp_inv],
           log=os.path.join(rd, "log", "last_pass_inventory.log"), check=False)
        if os.path.exists(lp_inv):
            try:
                inv = json.load(open(lp_inv))
                ln = [f"## 片上层实现量（last_pass intrinsic，{inv['kernel_body']}，"
                      f"{inv['total_call_sites']} 站点）", ""]
                for pv, iv in inv["pipes"].items():
                    ln.append(f"- **{pv}** {iv['sem']}：{iv['sites']} 站点"
                              + (f"，实测 {iv['instr_per_site']} 条/站点（编译器子 tile 化）"
                                 if iv.get("instr_per_site") else ""))
                ln += [f"- 同步：set_flag {inv['sync']['set_flag']} + wait_flag "
                       f"{inv['sync']['wait_flag']}",
                       "- MTE1/FIXP 站点为 ttir 不可见的编译器新增片上层；GM 层口径见 §2.5", ""]
                r = sh(["python3", "-c",
                        "import sys; p=sys.argv[1]; s=open(p).read(); "
                        "anchor='## 6. 循环状态'; "
                        "s=s.replace(anchor, chr(10).join(sys.argv[2:])+chr(10)+anchor, 1) "
                        "if anchor in s else s; open(p,'w').write(s)",
                        os.path.join(rd, "report.md")] + ln,
                       log=os.path.join(rd, "log", "report_patch.log"), check=False)
            except Exception as e:
                print(f"[last_pass_inventory] 报告补写失败（不影响产物）：{e}", flush=True)
    try:
        pipe_time = json.load(open(ptm_json))
        if not pipe_time:
            raise ValueError("JSON 为空")
    except Exception as e:
        pipe_time = {}
        print(f"[warn] §2.5 理论时间模型生成失败（报告将缺该节）：{e}", flush=True)
        print(f"[warn] 常见原因：环境未 source set_env.sh（缺 torch_npu/ASCEND_HOME_PATH）", flush=True)

    # 阶段 5：报告
    rp = os.path.join(rd, "report.md")
    L = []
    L.append(f"# Triton 算子性能分析报告：{os.path.basename(a.workdir)}（rounds{rnum}）\n")
    L.append(f"> 端到端流水线自动生成（run_full_analysis.py）；impl md5 `{md5}`；{a.label}\n")
    L.append("## 1. 数据采集清单\n")
    L.append("| 数据路 | 产物 | 完整性 |\n|---|---|---|")
    for c in cases:
        ok = os.path.exists(os.path.join(rd, f"board_c{c}_summary.json"))
        L.append(f"| board c{c} | `board_c{c}/` + summary.json | {'✅' if ok else '✗'} |")
    for kn in sim_kernels:
        st = glob.glob(os.path.join(rd, f"{tag}__{kn}", "capture_status.json"))
        v = json.load(open(st[0]))["verdict"] if st else "未采集"
        L.append(f"| sim {kn} | `{tag}__{kn}/` | {v} |")
    for kn in untriggered:
        L.append(f"| sim {kn} | — | **未触发**（board 全 case 无该 kernel 目录） |")
    L.append("\n## 2. 上板 bound 判定（c{}）\n".format(cases[0]))
    L.append("> kernel 保留决策：上板测全部 kernel 后，最多保留两个进入后续分析"
             "（跨 case 聚合占比 >50% → 只保留该一个；否则保留 top-2）。"
             f"本轮保留 **{kept_kernels}**，过滤 **{dropped or '无'}**"
             f"（聚合占比：{ {k: f'{v*100:.1f}%' for k, v in shares.items()} }）；"
             "明细见 `kernel_keep_decision.json`。\n")
    L.append("| kernel | 时长µs | 占比 | cube bound | vec bound | 附带问题 |")
    L.append("|---|---|---|---|---|---|")
    for kn, b in sorted(kernels_bound_all.items(), key=lambda kv: -kv[1]["dur"]):
        def bcell(bd):
            if not bd: return "—"
            return f"**{bd['pipe'].upper()} {bd['pct']*100:.0f}%**" if bd["is_bound"] else f"无（{bd['pipe']} {bd['pct']*100:.0f}%）"
        issues = []
        if b.get("conflict") and b["conflict"] > 0.05: issues.append(f"bank cflt {b['conflict']*100:.1f}%⚠")
        for kk, vv in (b.get("l2") or {}).items():
            if vv is not None and vv < 50: issues.append(f"L2 {kk[:4]} {vv:.0f}%⚠")
        sp = b.get("cube_spread") or {}
        if sp.get("max") and sp.get("min") is not None and (sp["max"] - sp["min"]) / sp["max"] > 0.1:
            issues.append(f"核间 {(sp['max']-sp['min'])/sp['max']*100:.0f}%⚠")
        L.append(f"| {kn} | {b['dur']:.1f} | {b['dur']/c1['total_us']*100:.1f}% | {bcell(b['cube'])} | {bcell(b['vec'])} | {' '.join(issues)} |")
    # §2.5 逐 case 逐 pipe 理论时间模型（C/V 拆分，带宽+数据量+时间）
    if pipe_time:
        _meta = pipe_time.get("_meta", {}) if isinstance(pipe_time, dict) else {}
        _verdict = _meta.get("sketch_verdict") or "未提供"
        _sk_err = _meta.get("sketch_error")
        L.append("\n### 2.5 理论时间模型（任务下限 + agent 草图 + 实测利用率）\n")
        L.append(f"草图状态：**{_verdict}**"
                 + (f"（{_sk_err}）" if _sk_err else "")
                 + "。三级口径：任务下限 = 输入各读一遍的字节和（机械推导，任何正确实现的最小读入）；"
                   "实现草图量 = agent 判断产物（受表达式白名单/下限/指纹三闸约束）；"
                   "利用率 = 理论 ÷ board 实测。**理论 bound = 该 kernel 估计用时最大的 pipe。**\n")
        if not _sk_err and _verdict in ("accepted", "rejected"):
            pass
        for case_key in sorted(pipe_time.keys()):
            if case_key.startswith("_"):
                continue
            cd = pipe_time[case_key]
            _shp = cd.get("in_shapes") or []
            L.append(f"**{case_key}**  输入形状={_shp}  任务下限(读入)={cd.get('floor_in_bytes', 0)/1e6:.2f}MB\n")
            L.append("| kernel | pipe | 数据量 | 理论速率 | 估计用时 µs |")
            L.append("|---|---|---|---|---|")
            for kn in cd.get("kernels", {}):
                ps = cd["kernels"][kn]
                bound = ps.get("_bound", "")
                for pname, d in sorted(ps.items(), key=lambda kv: -kv[1].get("time_us", 0) if not kv[0].startswith("_") else 0):
                    if pname.startswith("_"): continue
                    mark = " **←bound**" if pname == bound else ""
                    L.append(f"| {kn} | {pname} | {d['workload']} | {d['rate']} | {d['time_us']}{mark} |")
            for _n in cd.get("redundancy_notes", []) or []:
                L.append(f"\n> 冗余注记（agent 判断）：{_n}")
            L.append("")
        if not _sk_err and _verdict == "未提供":
            L.append("\n> ⚠ 实现草图未推导（无 kernel_sketch.json）：上表为任务下限口径，"
                     "实现搬运量与冗余度未知；按 triton-profiling-analysis SKILL 阶段 1.6 由 agent 补推导后重跑。\n")
        # 利用率表（理论 ÷ 实测）
        util = pipe_time.get("c1", {}).get("utilization", {})
        if util:
            L.append("\n**Pipe 利用率（理论时间 ÷ board 实测时间）**\n")
            L.append("| kernel | pipe | 利用率 | 说明 |")
            L.append("|---|---|---|---|")
            for kn in util:
                for pipe, u in sorted(util[kn].items(), key=lambda kv: -kv[1]):
                    note = "✅ 接近极限" if u >= 60 else ("⚠ 串行损耗" if u >= 20 else "🔴 严重气泡")
                    L.append(f"| {kn} | {pipe} | **{u}%** | {note} |")
            L.append("")

    L.append("\n## 3. 行级证据链（bound pipe → Top3 代码行 → 阻塞点）\n")
    L.append(f"行号 = `{impl_name}`；trace 为 c{cases[0]}" + (f" 策略B 收缩 T={a.shrink}；" if a.shrink else "；") + "口径见各 analysis_*.json。\n")
    for kn, ev in sorted(evidence.items(), key=lambda kv: -(kv[1].get("dur") or 0)):
        L.append(f"### `{kn}`（{(ev.get('dur') or 0):.1f}µs）\n")
        for bd in ev["bound"]:
            L.append(f"**bound** [{bd['side']}] board {bd['board_pipe'].upper()} {bd['board_pct']}%"
                     f"{'（达成）' if bd['is_bound'] else '（未达成）'} → trace {bd['trace_pipes']} Top3 行：\n")
            L.append("| 行 | 源码 | 等待µs | 执行µs |\n|---|---|---|---|")
            for tl in bd["top_lines"]:
                L.append(f"| L{tl['line']} | `{tl['src']}` | {tl['wait_us']} | {tl['exec_us']} |")
            L.append("")
        for sub, bl in ev.get("blocking", {}).items():
            if bl.get("exec"):
                L.append(f"**阻塞点（严格：独占执行+主要pipe全等待+周期≥3次，{sub}）**：")
                for l in bl["exec"][:3]: L.append(f"- {l}")
            if bl.get("wait"):
                L.append(f"**等待行 Top**（严格阻塞期间）：")
                for l in bl["wait"][:3]: L.append(f"- {l}")
            L.append("")
    # 双流水章节：三维度判定（c内 / v内 / cv）+ 未成功位置
    has_dp = any(ev.get("dual_pipeline") for ev in evidence.values())
    if has_dp:
        import re as _re
        L.append("## 4. 双流水判定（c内 / v内 / cv，≥30% 重叠为成功）\n")
        L.append("- **c 内双流水**：cube 子核内 搬运×CUBE计算 重叠（C_MTE1_MAC/C_MTE2_MAC 取大）；"
                 "**v 内双流水**：vector 子核内 搬运×VECTOR计算 重叠（V_MTE2_VEC）；"
                 "**cv 双流水**：Cube 与 Vector 两子核并行重叠（CVPipelining 目标）。\n"
                 "- 重叠 = 活跃区间交集/min(活跃)，<30% 记未成功并标注等待驻留的代码位置。\n")
        L.append("- **为何有的变量没拿到双缓冲**（据 bishengir MarkMultiBuffer.cpp 决策树）：mark 只给"
                 "「ND2NZ/Fixpipe/Load/Store 搬运 op 的片上侧 → 回溯根为独立 alloc → 在 scf.for/while 循环内"
                 " → 地址空间未被溢出回退整组关闭」的缓冲打标（深度默认 2）；计算型/原地物化/循环外不变量"
                 "天然出局。明细分类：**⚠ 应双未双** = 大且热的搬运/物化缓冲未获标（重点）；"
                 "无需类 = 小缓冲/不变量/计算型。\n")
        L.append("| kernel | c内双流水 | v内双流水 | cv双流水 | 未成功位置 |")
        L.append("|---|---|---|---|---|")

        def _pos(blocking, sub):
            """从阻塞表等待行提取 位置标记（L行号 + 源码摘要）。"""
            for line in (blocking.get(sub, {}) or {}).get("wait", [])[:1]:
                m = _re.search(r"(\S+\.py):(\d+)\s*\|\s*(.*)", line)
                if m:
                    src = m.group(3).split(" S ")[0].strip()[:42]
                    return f"L{m.group(2)} `{src}`"
            return "—"

        for kn, ev in sorted(evidence.items(), key=lambda kv: -(kv[1].get("dur") or 0)):
            dp = ev.get("dual_pipeline") or {}
            dims = (dp.get("verdict") or {}).get("dims") or {}
            blocking = ev.get("blocking") or {}
            cells, fails = [], []
            for k, name in (("c", "c内"), ("v", "v内"), ("cv", "cv")):
                if k not in dims:
                    cells.append("—"); continue
                d = dims[k]
                cells.append(f"{'✅' if d['ok'] else '❌'} {d['overlap']}%")
                if not d["ok"]:
                    fails.append(k)
            def _unbuf_main(spaces):
                rows = [r for r in ((ev.get("dual_buffer_vars") or {}).get("vars") or [])
                        if not r["verdict"].startswith("双流水") and r.get("space") in spaces]
                return max(rows, key=lambda r: r.get("ops", 0)) if rows else None
            pos = []
            if "c" in fails or "cv" in fails:
                tag = ("c内/cv" if "cv" in fails and "c" in fails else ("c内" if "c" in fails else "cv"))
                u = _unbuf_main({"cbuf", "cc"})
                extra = f"（主导单缓冲 `{u['var'] or '?'}` L{u['line']}）" if u else ""
                pos.append(f"{tag} {_pos(blocking, 'cubecore0')}{extra}")
            if "v" in fails:
                u = _unbuf_main({"ub"})
                extra = f"（主导单缓冲 `{u['var'] or '?'}` L{u['line']}）" if u else ""
                pos.append(f"v内 {_pos(blocking, 'veccore0')}{extra}")
            L.append(f"| {kn} | {cells[0]} | {cells[1]} | {cells[2]} | {'；'.join(pos) or '—'} |")
        L.append("")
        # 每 kernel 证据块（保留明细，术语已换）
        for kn, ev in sorted(evidence.items(), key=lambda kv: -(kv[1].get("dur") or 0)):
            dp = ev.get("dual_pipeline") or {}
            dbv = ev.get("dual_buffer_vars") or {}
            tr = dp.get("trace") or {}
            pairs = tr.get("pairs") or {}
            if not (dbv or pairs): continue
            L.append(f"### `{kn}` 双流水明细\n")
            dims = (dp.get("verdict") or {}).get("dims") or {}
            if dims:
                L.append("- 三维度：" + "，".join(
                    f"{name}{'成功' if dims[k]['ok'] else '未成功'}（重叠 {dims[k]['overlap']}%）"
                    for k, name in (("c", "c内"), ("v", "v内"), ("cv", "cv")) if k in dims))
            ir_lines = []
            if dbv:
                ir_lines.append(f"- 双缓冲变量 **{dbv.get('dual_ok')}/{dbv.get('vars_total')}**"
                                f"（判据：变量拿到 ≥2 个互不相同缓冲地址）")
                for r in (dbv.get("vars") or [])[:5]:
                    if r["verdict"].startswith("双流水"):
                        d = f"副本间距 {r['delta_bytes'][0]}B" if r.get("delta_bytes") else ""
                        ir_lines.append(f"  - `{r['var'] or '?'}` L{r['line']}：{r['distinct_addrs']} 份缓冲副本"
                                        + (f"，其中双缓冲对 ×{r['dual_pairs']}" if r["dual_pairs"] else "")
                                        + (f" {d}" if d else ""))
            if dbv:
                unbuf = [r for r in (dbv.get("vars") or [])
                         if not r["verdict"].startswith("双流水")]
                should = [r for r in unbuf if r.get("should_dual")]
                nolink = [r for r in unbuf if not r.get("should_dual")]
                if should:
                    ir_lines.append(f"- **⚠ 应双流水但未双缓冲（{len(should)} 个，重点）**：")
                    for r in sorted(should, key=lambda r: -(r.get("size_bytes") or 0)):
                        ir_lines.append(f"  - `{r['var'] or '?'}` L{r['line']}："
                                        f"{(r.get('size_bytes') or 0)//1024}KB × 访问{r['ops']}次 —— {r['reason']}")
                if nolink:
                    nolink.sort(key=lambda r: -r.get("ops", 0))
                    shown = "，".join(
                        f"`{r['var'] or '?'}` L{r['line']}（{r['reason'].split('，')[0]}）" for r in nolink[:10])
                    more = f" 等 {len(nolink)} 个" if len(nolink) > 10 else f" 共 {len(nolink)} 个"
                    ir_lines.append(f"- 未双缓冲-无需类（{more}）：{shown}")
            pd = dp.get("pass_diff") or {}
            if pd:
                ir_lines.append("- pass 执行：" + "，".join(f"{k}={v}" for k, v in pd.items()))
            if ir_lines:
                L.append("**IR 侧**：")
                L.extend(ir_lines)
                L.append("")
            if pairs:
                L.append("**运行时重叠**（全部掩盖对）：" +
                         "，".join(f"{k}={v}%" for k, v in sorted(pairs.items(), key=lambda x: -x[1])))
                L.append("")
    # §5 优化建议（规则驱动，自动生成，需人工复核）
    recs = []  # (优先级, 建议, 位置, 依据, 预期方向)

    def _pos_of(blocking, sub):
        import re as _re2
        for line in (blocking.get(sub, {}) or {}).get("wait", [])[:1]:
            m = _re2.search(r"(\S+\.py):(\d+)\s*\|\s*(.*)", line)
            if m:
                return f"L{m.group(2)} `{m.group(3).split(' S ')[0].strip()[:36]}`"
        return "—"

    for kn, ev in sorted(evidence.items(), key=lambda kv: -(kv[1].get("dur") or 0)):
        dp = ev.get("dual_pipeline") or {}
        dbv = ev.get("dual_buffer_vars") or {}
        b = kernels_bound.get(kn, {})
        vd = (dp.get("verdict") or {})
        dims = vd.get("dims") or {}
        blocking = ev.get("blocking") or {}
        dur = ev.get("dur") or 0
        # R1 双流水未成功维度 → 串行消除
        for k, name, sub in (("c", "c内", "cubecore0"), ("v", "v内", "veccore0"), ("cv", "cv", "cubecore0")):
            if k in dims and not dims[k]["ok"]:
                recs.append((1, f"{name}双流水未成功（重叠 {dims[k]['overlap']}% <30%）——消除该处串行依赖/增强掩盖",
                             f"{kn} @ {_pos_of(blocking, sub)}",
                             f"重叠度={dims[k]['overlap']}% @ dual_pipeline__{kn}.json",
                             "核内多链交织 / L0C 累加常驻（砍 dot→FIXP→dot 往返）"))
        # R2 应双未双 + 枝5 → 空间腾挪
        for r in (dbv.get("vars") or []):
            if r.get("should_dual") and "整组关闭" in r.get("reason", ""):
                recs.append((2, f"UB 空间腾挪以恢复多缓冲：`{r['var'] or '?'}` L{r['line']}"
                                f"（{r.get('size_bytes', 0)//1024}KB×{r['ops']}次）",
                                f"{kn} L{r['line']}",
                                f"{r['reason']} @ dual_buffer_vars__{kn}.json",
                                "先压缩 UB 其它占用/改布局，否则 mark 的 UB 组恒被溢出回退关闭"))
                break  # 每 kernel 一条汇总即可
        # R3 roofline 不建议
        cb = b.get("cube") or {}
        if cb and "mte2" in (cb.get("pipe") or "") and (cb.get("pct") or 0) > 0.9:
            l2 = (b.get("l2") or {}).get("aic_total_hit_rate(%)")
            recs.append((4, "已贴 HBM roofline，不建议再优化该 kernel", kn,
                         f"cube MTE2 {cb['pct']*100:.0f}%，L2 {l2 or '?'}%", "转向其它 kernel"))
        # R4 附带问题
        cflt = b.get("conflict")
        if cflt and cflt > 0.05:
            recs.append((3, f"bank conflict {cflt*100:.1f}% 超 5% 线——改访存 stride/tile 切分", kn,
                         f"aiv_vec_total_cflt_ratio={cflt:.3f} @ board 汇总", "+0.5~1%"))
        sp = b.get("cube_spread") or {}
        if sp.get("max") and sp.get("min") is not None and (sp["max"] - sp["min"]) / sp["max"] > 0.3:
            recs.append((3, f"核间负载不均衡 {(sp['max']-sp['min'])/sp['max']*100:.0f}%——persistent grid/动态取块",
                         kn, f"aic_time {sp['min']:.0f}→{sp['max']:.0f}µs @ board 汇总", "回收空转核时间"))
        vb = b.get("vec") or {}
        if vb and "scalar" in (vb.get("pipe") or "") and (vb.get("pct") or 0) > 0.9:
            recs.append((2, "vec 侧 SCALAR bound >90%——疑似标量递退（如 cumsum），向量化/并行化改造",
                         kn, f"vec scalar {vb['pct']*100:.0f}% @ board 汇总", "消除标量瓶颈（收益集中该 kernel）"))

    # 去重（同 kernel 同建议文本）并按优先级渲染
    seen, rows = set(), []
    for pri, sug, loc, basis, expect in sorted(recs, key=lambda r: r[0]):
        key = (sug, loc)
        if key in seen: continue
        seen.add(key)
        rows.append((pri, sug, loc, basis, expect))
    if rows:
        L.append("## 5. 优化建议（规则驱动自动生成，需人工复核）\n")
        L.append("| # | 优先级 | 建议 | 位置 | 判定依据 | 预期方向 |")
        L.append("|---|---|---|---|---|---|")
        for i, (pri, sug, loc, basis, expect) in enumerate(rows, 1):
            L.append(f"| {i} | {'🔴' if pri == 1 else ('🟡' if pri <= 2 else '⚪')} | {sug} | {loc} | {basis} | {expect} |")
        L.append("")
    L.append("## 6. 证据附录\n")
    L.append("- 机器可读：`evidence_chains.json`、`analysis_*.json`、`board_c*_summary.json`")
    L.append("- 复核：本脚本同参数重跑即可复现（产物落 simulator/roundsN+1，不覆盖本轮）")
    open(rp, "w").write("\n".join(L))
    print(f"[report] {rp}", flush=True)

if __name__ == "__main__":
    main()
