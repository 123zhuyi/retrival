#!/usr/bin/env python3
"""迭代归档规范校验器（triton-profiling-analysis skill §5.1 的执行器，skill 自包含）。

用法: python3 <skill>/scripts/check_archive.py <算子工作目录>
退出码: 0=通过; 1=存在违规
规则: profiling 轮次须含 log/manifest/profiling 报告与指纹链；collections 登记的
采集目录必须存在（只增不删）。manifest 含 "profiling_flow": false 的迭代仅查基础文件。
本脚本不依赖编排层任何文件——由 skill 在归档步骤末尾自动调用。
"""
import argparse, glob, hashlib, json, os, re, sys

def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()[:10]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workdir")
    a = ap.parse_args()
    out = os.path.join(a.workdir, "output")
    iters = sorted(d for d in glob.glob(os.path.join(out, "iter_*")) + glob.glob(os.path.join(out, "opt_iter_*"))
                   if os.path.isdir(d))
    if not iters:
        print("FAIL: 无迭代目录"); return 1
    errs, warns = [], []
    for d in iters:
        name = os.path.basename(d)
        mf = os.path.join(d, "manifest.json")
        m = None
        if not os.path.exists(mf):
            errs.append(f"{name}: manifest.json 缺失")
        else:
            try:
                m = json.load(open(mf))
                for k in ("iter", "code"):
                    if k not in m: errs.append(f"{name}: manifest 缺字段 {k}")
            except Exception as e:
                errs.append(f"{name}: manifest 解析失败 {e}")
        code = None
        for c in ("optimized_code.py", "code.py", "generated_code.py"):
            p = os.path.join(d, c)
            if os.path.exists(p): code = p; break
        if code and m and m.get("code", {}).get("md5") and md5(code) != m["code"]["md5"]:
            errs.append(f"{name}: 代码 md5 与 manifest 不一致")
        lg = os.path.join(d, "log.md")
        if not os.path.exists(lg):
            errs.append(f"{name}: log.md 缺失")
        elif code:
            lm = re.search(r"代码指纹[:：]\s*([0-9a-f]{10})", open(lg).read())
            if lm and lm.group(1) != md5(code):
                errs.append(f"{name}: log.md 指纹与代码不一致")
        for j in ("verify/verify_result.json", "verify/perf_result.json"):
            if not os.path.exists(os.path.join(d, j)):
                warns.append(f"{name}: {j} 缺失")
        flow_off = bool(m and m.get("profiling_flow") is False)
        for r in ([] if flow_off else glob.glob(os.path.join(d, "profiling", "*_report.md"))):
            txt = open(r).read()
            for sec in ("## 1. 分析对象与环境", "## 3. 上板 bound 判定",
                        "## 5. 优化建议", "## 7. 证据附录"):
                if sec not in txt:
                    errs.append(f"{name}/profiling/{os.path.basename(r)}: 缺章节「{sec}」")
        if m:
            for r in m.get("profiling_reports", []):
                if not os.path.exists(os.path.join(d, r.get("file", "?"))):
                    errs.append(f"{name}: manifest 声明的报告不存在 {r.get('file')}")
            for c in m.get("collections", []):
                if not os.path.isdir(os.path.join(d, c.get("path", "?"))):
                    errs.append(f"{name}: collections 采集目录缺失 {c.get('path')}（只增不删）")
    for e in errs: print("ERROR:", e)
    for w in warns: print("WARN :", w)
    ok = not errs
    print(f"\n[check_archive] {len(iters)} 个迭代目录，{len(errs)} 错误 / {len(warns)} 警告 → {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
