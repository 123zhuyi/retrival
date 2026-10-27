#!/usr/bin/env python3
"""理论时间模型（算子无关，零内置算子模板）。

三级构成（语义不同，禁止互相替代）：
  (a) 任务级下限    ：输入张量字节和（任何正确实现都必须各读 ≥1 遍）——机械推导，永远可得
  (b) 实现草图量    ：agent 判断产物 kernel_sketch.json（表达式 + IR 引用 + 冗余注记），
                      受三道机械闸约束：表达式白名单 / gm_in ≥ 任务下限 / 代码指纹 md5
  (c) board 实测利用率：理论 ÷ 实测（--board）

历史版本曾在代码内置算子草图（r39 五段、GDN 融合段）——它们绑定具体实现，
对新算子/新实现必然张冠李戴（2026-10-26 事故：五段草图对 201_GDN 静默产错值，
T←头数、H←序列长）。现全部删除：草图判断交给 agent，代码只保留机械校验。

草图 schema（kernel_sketch.json）：
{
  "code_md5_prefix": "<最终代码 md5 前 10 位>",
  "constants":  {"BT": 64},
  "aliases":    {"K": "in0_d3", "Hv": "in1_d1", "NT": "nseg*cdiv(in0_d2,BT)", ...},
  "kernels": {"<保留kernel名>": {
      "ir_refs": ["last_pass.mlir:<行号> tt.load …"],
      "exprs": {"c_flop": "...", "v_flop": "...", "gm_in": "...", "gm_out": "...", "fixp_bytes": "..."},
      "redundancy_notes": ["…无效搬运/无效计算判断…"]}}}

机械符号表（本脚本提供，表达式只允许引用这些 + constants/aliases）：
  in{i}_d{k} / in{i}_bytes（第 i 个浮点输入张量的各维与字节数）、nseg、total_len
表达式白名单：数字、符号、+ - * / //、一元负号、cdiv(a,b)。其余语法一律拒绝。
"""
import argparse, ast, importlib.util, json, os, subprocess, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def load_theory():
    r = subprocess.run([sys.executable, os.path.join(HERE, "chip_spec.py"), "--json"],
                       capture_output=True, text=True)
    return json.loads(r.stdout)


DT_SIZE = {"bf16": 2, "f16": 2, "fp16": 2, "f32": 4, "fp32": 4, "i32": 4, "i64": 8, "u32": 4}


def ir_traffic(mlir_path):
    """机械统计 ttir 的 tt.load/tt.store 字节（算子无关）。

    以第一条 scf.for 分割：循环内 = 逐迭代执行；循环外 = 每 program 一次
    （链首 hi 等循环不变 load）。返回 dict，含明细（供报告审计）。
    """
    import re
    lines = open(mlir_path, errors="ignore").readlines()
    for_idx = next((i for i, l in enumerate(lines) if "scf.for" in l), len(lines))
    regions = {"pre_loop": lines[:for_idx], "in_loop": lines[for_idx:]}
    out = {}
    for rname, ls in regions.items():
        for op in ("tt.load", "tt.store"):
            b, det = 0, []
            for line in ls:
                m = re.search(rf"{op}.*?tensor<([0-9dx?]+)x!tt\.ptr<(\w+)>", line)
                if not m:
                    continue
                try:
                    dims = [int(x) for x in m.group(1).split("x")]
                except ValueError:
                    continue
                n = 1
                for d in dims:
                    n *= d
                n *= DT_SIZE.get(m.group(2), 2)
                b += n
                det.append(f"{op} {'x'.join(map(str, dims))}x{m.group(2)}={n}B")
            out[f"{rname}_{op.replace('.', '_')}"] = b
            out[f"{rname}_{op.replace('.', '_')}_det"] = det
    return out


def eval_expr(expr, env):
    """白名单表达式求值：数字/符号/四则/整除/一元负号/cdiv(a,b)。其他语法拒绝。"""
    def _ev(n):
        if isinstance(n, ast.Expression):
            return _ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.Name):
            if n.id not in env:
                raise ValueError(f"未知符号 {n.id}")
            return env[n.id]
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv)):
            # 统一 float 语义：int/float 混合时 int.__add__(float) 会返回 NotImplemented
            # （不抛错、以值传播），必须先把两操作数都收敛为 float
            import operator as _opmod
            a, b = float(_ev(n.left)), float(_ev(n.right))
            return {ast.Add: _opmod.add, ast.Sub: _opmod.sub, ast.Mult: _opmod.mul,
                    ast.Div: _opmod.truediv, ast.FloorDiv: _opmod.floordiv}[type(n.op)](a, b)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -float(_ev(n.operand))
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "cdiv"
                and len(n.args) == 2 and not n.keywords):
            x, y = float(_ev(n.args[0])), float(_ev(n.args[1]))
            return float(-(-x // y))
        raise ValueError(f"不允许的语法: {ast.dump(n)[:80]}")
    return float(_ev(ast.parse(str(expr).strip(), mode="eval")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--cases", default="1-7")
    ap.add_argument("--json", default=None)
    ap.add_argument("--board", default=None, help="board_cN_summary.json 路径（有则输出 pipe 利用率）")
    ap.add_argument("--sketch", default=None, help="agent 推导的计算草图 kernel_sketch.json")
    ap.add_argument("--expect-md5", default=None, help="最终代码 md5 前 10/12 位（与草图指纹比对）")
    ap.add_argument("--ir", default=None, help="ir_dump/*_ttir.mlir（机械统计 tt.load/tt.store 逐迭代字节，"
                    "与草图 gm_in/gm_out 对账；偏差>30% 标 warning）")
    ap.add_argument("--kernel-name", default=None, help="无草图时任务下限条目的键名（保留 kernel 名）")
    a = ap.parse_args()

    spec = importlib.util.spec_from_file_location("task", a.task)
    task = importlib.util.module_from_spec(spec); spec.loader.exec_module(task)
    torch.manual_seed(0)

    chip = load_theory()
    rates = {
        "CUBE":  chip["cube_peak_tflops_fp16_bf16_dense"] * 1e12,
        "VECTOR": chip["vector_tflops_fp32"] * 1e12,
    }
    for k, v in chip["pipe_view"].items():
        pipe = k.split("(")[0]; side = k.split("(")[1][0]
        rates[f"{pipe}({side})"] = v["aggregate_tbs"] * 1e12

    cases = []
    for part in a.cases.split(","):
        if "-" in part: lo, hi = part.split("-"); cases.extend(range(int(lo), int(hi)+1))
        else: cases.append(int(part))

    # ── 草图装载与指纹闸 ──
    sketch, sketch_err = None, None
    if a.sketch and os.path.exists(a.sketch):
        try:
            sketch = json.load(open(a.sketch))
            if a.expect_md5 and sketch.get("code_md5_prefix") and \
               not a.expect_md5.startswith(sketch["code_md5_prefix"]):
                sketch_err = f"指纹不匹配：草图 {sketch['code_md5_prefix']} ≠ 最终代码 {a.expect_md5[:10]}"
                sketch = None
        except Exception as e:
            sketch_err = f"草图 JSON 解析失败：{e}"
            sketch = None

    result = {"_meta": {"sketch": a.sketch, "sketch_verdict": None, "sketch_error": sketch_err}}
    rejected = False
    for ci in cases:
        inputs = list(task.get_input_groups()[ci - 1])

        # cu_seqlens 探测（算子无关）：任意 1D 整型、首元素 0 且单调不减
        def _is_cu(t):
            return (torch.is_tensor(t) and t.dim() == 1 and t.numel() >= 2
                    and t.dtype in (torch.int32, torch.int64))
        cu_t = next((t for t in inputs if _is_cu(t)), None)
        if cu_t is not None:
            if not ((cu_t[0] == 0) and bool((cu_t[1:] >= cu_t[:-1]).all())):
                cu_t = None

        # 机械符号表 + 任务级下限（浮点输入各读一遍的字节和）
        env, floor_in, ti, in_shapes = {}, 0, 0, []
        for t in inputs:
            if torch.is_tensor(t) and t.dim() >= 1 and t.is_floating_point():
                for k, s in enumerate(t.shape):
                    env[f"in{ti}_d{k}"] = int(s)
                b = int(t.numel() * t.element_size())
                env[f"in{ti}_bytes"] = b
                floor_in += b
                in_shapes.append(list(t.shape))
                ti += 1
        if cu_t is not None:
            env["nseg"] = len(cu_t) - 1
            env["total_len"] = int(cu_t[-1].item())

        env.update(sketch.get("constants", {}) if sketch else {})
        # 别名（agent 判断的语义符号），按序求值、允许链式引用
        alias_err = None
        if sketch:
            for name, expr in (sketch.get("aliases") or {}).items():
                try:
                    env[name] = eval_expr(expr, env)
                except Exception as e:
                    alias_err = f"别名 {name}={expr} 求值失败：{e}"
                    break

        case_data = {"in_shapes": in_shapes, "floor_in_bytes": floor_in, "kernels": {}}

        if sketch and not alias_err and not rejected:
            for kn, kd in (sketch.get("kernels") or {}).items():
                pipes = {}
                for key, pipe, rate_key in [
                    ("c_flop", "CUBE(C)", "CUBE"),
                    ("v_flop", "VECTOR(V)", "VECTOR"),
                    ("gm_in", "MTE2", None), ("gm_out", "MTE3", None),
                    ("fixp_bytes", "FIXP", None),
                ]:
                    expr = (kd.get("exprs") or {}).get(key)
                    if not expr:
                        continue
                    try:
                        val = eval_expr(expr, env)
                    except Exception as e:
                        rejected = True
                        result["_meta"]["sketch_verdict"] = f"rejected: {kn}.{key} 表达式失败：{e}"
                        break
                    if pipe in ("CUBE(C)", "VECTOR(V)"):
                        rate = rates[rate_key]
                        pipes[pipe] = {"workload": f"{val/1e9:.2f} GFLOP",
                                       "rate": f"{rate/1e12:.1f} TFLOPS",
                                       "time_us": round(val / rate * 1e6, 2), "expr": expr}
                    else:
                        bw = rates.get(f"{pipe}(C)", rates.get(f"{pipe}(V)", 57.6e9))
                        pipes[pipe] = {"workload": f"{val/1e6:.2f} MB",
                                       "rate": f"{bw/1e9:.0f} GB/s",
                                       "time_us": round(val / bw * 1e6, 2), "expr": expr}
                if rejected:
                    pipes = {}
                # 机械闸：实现声称的读入不得低于任务下限（读少一遍 = 少算工作）
                if not rejected and "MTE2" in pipes:
                    gmv = float(pipes["MTE2"]["workload"].split()[0]) * 1e6
                    if gmv < floor_in * 0.999:
                        rejected = True
                        result["_meta"]["sketch_verdict"] = (
                            f"rejected: {kn} gm_in={gmv/1e6:.2f}MB < 任务下限 {floor_in/1e6:.2f}MB"
                            "（实现不可能读得比输入还少——草图低估，疑似少算）")
                        pipes = {}
                if pipes:
                    bound = max(pipes, key=lambda k: pipes[k]["time_us"])
                    pipes["_bound"] = bound
                    pipes["_theo_floor_us"] = max(pipes[k]["time_us"] for k in pipes if not k.startswith("_"))
                case_data["kernels"][kn] = pipes

            # IR 交叉对账（独立于 agent 的机械估计，第一个带 gm_in 的 kernel）
            if a.ir and os.path.exists(a.ir) and "programs" in env and "NT" in env:
                try:
                    irv = ir_traffic(a.ir)
                    kn0 = next((k for k in case_data["kernels"] if "MTE2" in case_data["kernels"][k]), None)
                    if kn0:
                        est = (env["programs"] * env["NT"] * irv["in_loop_tt_load"]
                               + env["programs"] * irv["pre_loop_tt_load"])
                        agent_v = float(case_data["kernels"][kn0]["MTE2"]["workload"].split()[0]) * 1e6
                        dev = abs(agent_v - est) / est * 100 if est else 0
                        case_data["ir_check"] = {
                            "kernel": kn0, "agent_gm_in_mb": round(agent_v/1e6, 2),
                            "ir_est_mb": round(est/1e6, 2), "dev_pct": round(dev, 1),
                            "per_iter_load_bytes": irv["in_loop_tt_load"],
                            "pre_loop_load_bytes": irv["pre_loop_tt_load"],
                            "loads": (irv["in_loop_tt_load_det"] + irv["pre_loop_tt_load_det"])[:12],
                            "verdict": "pass" if dev <= 25 else "DEVIATION>25% 需人工复核"}
                        if dev > 25:
                            result["_meta"]["sketch_verdict"] = (
                                f"accepted-with-warning: gm_in 与 IR 推导偏差 {dev:.0f}%")
                except Exception as e:
                    case_data["ir_check"] = {"error": str(e)}
            case_data["redundancy_notes"] = [n for kd in (sketch.get("kernels") or {}).values()
                                             for n in (kd.get("redundancy_notes") or [])]
            case_data["ir_refs"] = {kn: kd.get("ir_refs") for kn, kd in (sketch.get("kernels") or {}).items()}
        elif not sketch:
            # 无草图：只有任务下限一个条目（MTE2 口径；实现量与冗余度未知，报告须标注）
            kname = a.kernel_name or "task_floor"
            bw = rates.get("MTE2(C)", rates.get("MTE2(V)", 57.6e9))
            case_data["kernels"][kname] = {
                "MTE2": {"workload": f"{floor_in/1e6:.2f} MB", "rate": f"{bw/1e9:.0f} GB/s",
                         "time_us": round(floor_in / bw * 1e6, 2), "expr": "floor_in"},
                "_bound": "MTE2",
                "_theo_floor_us": round(floor_in / bw * 1e6, 2)}

        if rejected and result["_meta"]["sketch_verdict"] and not case_data["kernels"]:
            # 草图被闸拒绝 → 全部 case 回落任务下限口径
            kname = a.kernel_name or "task_floor"
            bw = rates.get("MTE2(C)", rates.get("MTE2(V)", 57.6e9))
            case_data["kernels"][kname] = {
                "MTE2": {"workload": f"{floor_in/1e6:.2f} MB", "rate": f"{bw/1e9:.0f} GB/s",
                         "time_us": round(floor_in / bw * 1e6, 2), "expr": "floor_in"},
                "_bound": "MTE2", "_theo_floor_us": round(floor_in / bw * 1e6, 2)}
        result[f"c{ci}"] = case_data

    if sketch and not result["_meta"]["sketch_verdict"]:
        result["_meta"]["sketch_verdict"] = "accepted"

    if a.json:
        json.dump(result, open(a.json, "w"), ensure_ascii=False, indent=1)
        print(f"\n[json] {a.json}")


if __name__ == "__main__":
    main()
