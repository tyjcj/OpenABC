#!/usr/bin/env python3
import argparse
import csv
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


ELAPSE_RE = re.compile(r"^elapse:\s*([0-9.]+)\s*seconds,\s*total:\s*([0-9.]+)\s*seconds\s*$")
STEP_START_RE = re.compile(r"^__STEP_START__(\d+)__\s*$")
STEP_END_RE = re.compile(r"^__STEP_END__(\d+)__\s*$")
BENCH_BEGIN_RE = re.compile(r"^__BENCH_BEGIN__\s*$")
BENCH_END_RE = re.compile(r"^__BENCH_END__\s*$")
STIME_BEGIN_RE = re.compile(r"^__STIME_BEGIN__\s*$")
STIME_END_RE = re.compile(r"^__STIME_END__\s*$")
AREA_DELAY_STIME_RE = re.compile(r"Area\s*=\s*([0-9.]+).*?Delay\s*=\s*([0-9.]+)\s*ps", re.IGNORECASE)
AND_LINE_RE = re.compile(r"^\s*([^=\s]+)\s*=\s*AND\(([^,]+),([^\)]+)\)\s*$")
NOT_LINE_RE = re.compile(r"^\s*([^=\s]+)\s*=\s*NOT\(([^\)]+)\)\s*$")
BUFF_LINE_RE = re.compile(r"^\s*([^=\s]+)\s*=\s*BUFF\(([^\)]+)\)\s*$")
INPUT_RE = re.compile(r"^\s*INPUT\(([^\)]+)\)\s*$")
OUTPUT_RE = re.compile(r"^\s*OUTPUT\(([^\)]+)\)\s*$")


@dataclass
class StepResult:
    script: str
    stage: str  # init|script
    step_index: int
    command_original: str
    command_executed: str
    elapsed_seconds: Optional[float]
    total_seconds: Optional[float]
    ok: bool
    notes: str


def _read_script_lines(path: Path) -> List[str]:
    lines: List[str] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def _rewrite_write_bench(cmd: str, out_dir: Path, step_index: int) -> str:
    stripped = cmd.strip()
    if not stripped.startswith("write_bench"):
        return cmd
    if out_dir.as_posix() == "-":
        # Keep the step but do not write any files.
        return f"echo SKIP_WRITE_BENCH {step_index}"
    # Handle: write_bench [-l] <path>
    parts = stripped.split()
    has_l = "-l" in parts[1:3]
    out_path = out_dir / f"write_bench_step{step_index}.bench"
    if has_l:
        return f"write_bench -l {out_path.as_posix()}"
    return f"write_bench {out_path.as_posix()}"


def _build_abc_command_sequence(
    *,
    script_name: str,
    init_cmds: List[Tuple[str, str]],  # (orig, exec)
    script_cmds: List[Tuple[str, str]],  # (orig, exec)
) -> Tuple[str, List[Tuple[str, str, str, int]]]:
    """
    Returns:
      - combined ABC command string
      - step metadata: (stage, orig, exec, step_index)
    """
    meta: List[Tuple[str, str, str, int]] = []
    seq: List[str] = []
    seq.append(f"echo __SCRIPT_BEGIN__ {script_name}")

    def add_step(stage: str, orig: str, exe: str, idx: int) -> None:
        meta.append((stage, orig, exe, idx))
        seq.append(f"echo __STEP_START__{len(meta)-1}__")
        seq.append("time -c")
        seq.append(exe)
        seq.append("time")
        seq.append(f"echo __STEP_END__{len(meta)-1}__")

    # init steps first (their step_index is negative, but we expose original idx in CSV)
    for idx, (orig, exe) in enumerate(init_cmds):
        add_step("init", orig, exe, idx)

    for idx, (orig, exe) in enumerate(script_cmds):
        add_step("script", orig, exe, idx)

    seq.append(f"echo __SCRIPT_END__ {script_name}")
    return "; ".join(seq), meta


def _run_abc(abc_bin: str, abc_cmds: str, cwd: Path) -> str:
    proc = subprocess.run(
        [abc_bin, "-c", abc_cmds],
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return proc.stdout


def _parse_output(
    *,
    out: str,
    script_name: str,
    meta: List[Tuple[str, str, str, int]],
) -> List[StepResult]:
    results: List[StepResult] = []

    current_step: Optional[int] = None
    current_lines: List[str] = []

    def finalize(step_no: int, chunk: List[str]) -> None:
        stage, orig, exe, idx_in_stage = meta[step_no]
        elapsed: Optional[float] = None
        total: Optional[float] = None
        notes: List[str] = []

        for ln in chunk:
            m = ELAPSE_RE.match(ln.strip())
            if m:
                elapsed = float(m.group(1))
                total = float(m.group(2))

        # Heuristic failure detection (ABC often exits 0 even on errors).
        lowered = "\n".join(chunk).lower()
        ok = True
        if "error:" in lowered or "has failed" in lowered or "** cmd error" in lowered:
            ok = False
            for ln in chunk:
                if ("error:" in ln.lower()) or ("has failed" in ln.lower()) or ("** cmd error" in ln.lower()):
                    notes.append(ln.strip())
            notes = notes[:5]

        if elapsed is None:
            ok = False
            notes = notes or ["missing timing output (no 'elapse:' line)"]

        results.append(
            StepResult(
                script=script_name,
                stage=stage,
                step_index=idx_in_stage,
                command_original=orig,
                command_executed=exe,
                elapsed_seconds=elapsed,
                total_seconds=total,
                ok=ok,
                notes=" | ".join(notes),
            )
        )

    for raw in out.splitlines():
        line = raw.rstrip("\n")
        m_start = STEP_START_RE.match(line.strip())
        if m_start:
            current_step = int(m_start.group(1))
            current_lines = []
            continue

        m_end = STEP_END_RE.match(line.strip())
        if m_end and current_step is not None:
            end_step = int(m_end.group(1))
            if end_step == current_step:
                finalize(current_step, current_lines)
            current_step = None
            current_lines = []
            continue

        if current_step is not None:
            current_lines.append(line)

    return results


def _write_csv(results: Iterable[StepResult], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "script",
                "stage",
                "step_index",
                "ok",
                "elapsed_seconds",
                "total_seconds",
                "command_original",
                "command_executed",
                "notes",
            ]
        )
        for r in results:
            w.writerow(
                [
                    r.script,
                    r.stage,
                    r.step_index,
                    int(r.ok),
                    "" if r.elapsed_seconds is None else f"{r.elapsed_seconds:.6f}",
                    "" if r.total_seconds is None else f"{r.total_seconds:.6f}",
                    r.command_original,
                    r.command_executed,
                    r.notes,
                ]
            )


def _print_summary(results: List[StepResult]) -> None:
    by_script: dict[str, List[StepResult]] = {}
    for r in results:
        by_script.setdefault(r.script, []).append(r)

    for script, rs in by_script.items():
        total_ok = sum(1 for r in rs if r.ok and r.elapsed_seconds is not None)
        total_steps = len(rs)
        total_time = sum((r.elapsed_seconds or 0.0) for r in rs if r.stage == "script")
        print(f"\n=== {script} ===")
        print(f"steps: {total_steps} (ok={total_ok}) | script_elapsed_sum={total_time:.3f}s")
        worst = sorted(
            (r for r in rs if r.elapsed_seconds is not None),
            key=lambda x: x.elapsed_seconds or 0.0,
            reverse=True,
        )[:5]
        print("top5 by elapsed_seconds:")
        for r in worst:
            print(f"  [{r.stage}:{r.step_index:02d}] {r.elapsed_seconds:.6f}s | {r.command_executed}")


def _derive_design_key(bench_path: Path) -> str:
    name = bench_path.stem
    if name.endswith("_orig"):
        name = name[: -len("_orig")]
    return name


def _bench_metrics_from_text(bench_text: str) -> Tuple[int, int, int]:
    """
    Returns (ANDgates, NOTgates, lpLen) using the same interpretation as
    datagen/utilities/andAIG2Graphml.py: NOTs are counted as NOT edges, not nodes.
    """
    node_name_to_id: Dict[str, int] = {}
    single_gate_io: Dict[str, str] = {}
    po_list: List[str] = []
    depth_by_id: Dict[int, int] = {}

    idx = 0
    and_count = 0
    not_edge_count = 0

    def add_pi(name: str) -> None:
        nonlocal idx
        node_name_to_id[name] = idx
        depth_by_id[idx] = 0
        idx += 1

    def get_src_id_and_edge(inp: str) -> Tuple[int, bool]:
        nonlocal not_edge_count
        if inp in node_name_to_id:
            return node_name_to_id[inp], False
        # In the dataset pipeline, this means inp is the output of a NOT/BUFF line and is treated as inverted.
        base = single_gate_io.get(inp)
        if base is None:
            raise ValueError(f"Unresolved signal reference: {inp}")
        src_id = node_name_to_id.get(base)
        if src_id is None:
            raise ValueError(f"Unresolved base signal reference: {base} (from {inp})")
        not_edge_count += 1
        return src_id, True

    for raw in bench_text.splitlines():
        line = raw.strip()
        if not line or "ABC" in line:
            continue

        m = INPUT_RE.match(line)
        if m:
            add_pi(m.group(1))
            continue

        if "vdd" in line and "=" in line:
            # Matches andAIG2Graphml.py behavior: treat vdd assignment as PI.
            left = line.replace(" ", "").split("=", 1)[0]
            add_pi(left)
            continue

        m = OUTPUT_RE.match(line)
        if m:
            po_list.append(m.group(1))
            continue

        compact = re.sub(r"\s+", "", line)
        m = AND_LINE_RE.match(compact)
        if m:
            out = m.group(1)
            in1 = m.group(2)
            in2 = m.group(3)
            src1, _ = get_src_id_and_edge(in1)
            src2, _ = get_src_id_and_edge(in2)
            d = max(depth_by_id[src1] + 1, depth_by_id[src2] + 1)
            node_name_to_id[out] = idx
            depth_by_id[idx] = d
            and_count += 1
            and_id = idx
            idx += 1
            if out in po_list:
                # Add PO buffer node.
                node_name_to_id[out + "_buff"] = idx
                depth_by_id[idx] = depth_by_id[and_id] + 1
                idx += 1
            continue

        m = NOT_LINE_RE.match(compact)
        if m:
            out = m.group(1)
            inp = m.group(2)
            single_gate_io[out] = inp
            if out in po_list:
                # PO inverter node adds a NOT edge.
                src_id = node_name_to_id.get(inp)
                if src_id is None:
                    base = single_gate_io.get(inp)
                    if base is None:
                        raise ValueError(f"Unresolved PO inverter input: {inp}")
                    src_id = node_name_to_id[base]
                not_edge_count += 1
                node_name_to_id[out + "_inv"] = idx
                depth_by_id[idx] = depth_by_id[src_id] + 1
                idx += 1
            continue

        m = BUFF_LINE_RE.match(compact)
        if m:
            out = m.group(1)
            inp = m.group(2)
            single_gate_io[out] = inp
            if out in po_list:
                # PO buffer node; may come after NOT chain.
                if inp in node_name_to_id:
                    src_id = node_name_to_id[inp]
                    add_not = False
                else:
                    base = single_gate_io.get(inp)
                    if base is None:
                        raise ValueError(f"Unresolved PO buffer input: {inp}")
                    src_id = node_name_to_id[base]
                    add_not = True
                if add_not:
                    not_edge_count += 1
                node_name_to_id[out + "_buff"] = idx
                depth_by_id[idx] = depth_by_id[src_id] + 1
                idx += 1
            continue

        # Ignore other lines (comments, etc.)

    lp_len = max(depth_by_id.values()) if depth_by_id else 0
    return and_count, not_edge_count, lp_len


def _extract_between_markers(out: str) -> str:
    lines = out.splitlines()
    # Try marker-based extraction first.
    start = None
    end = None
    for i, ln in enumerate(lines):
        if BENCH_BEGIN_RE.match(ln.strip()):
            start = i + 1
        elif BENCH_END_RE.match(ln.strip()) and start is not None:
            end = i
            break
    if start is not None and end is not None and end > start:
        chunk = "\n".join(lines[start:end])
        if "INPUT(" in chunk:
            return chunk

    # Fallback: some ABC outputs (notably write_bench to /dev/stdout) may be flushed
    # before "echo" output, so markers can appear after the dump. In that case, capture
    # the first BENCH-like block at the beginning of output until the next ABC banner.
    bench_start = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("# Benchmark") or s.startswith("INPUT("):
            bench_start = i
            break
    if bench_start is None:
        raise RuntimeError("Failed to locate BENCH dump in ABC output.")

    bench_end = None
    for i in range(bench_start + 1, len(lines)):
        if lines[i].startswith("ABC command line:"):
            bench_end = i
            break
    if bench_end is None:
        bench_end = len(lines)
    return "\n".join(lines[bench_start:bench_end])


def _extract_area_delay(out: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract the *global* mapped `Area=` and `Delay=` reported by `stime -c`.

    This matches the dataset pipeline (see datagen/utilities/collectAreaAndDelay.py),
    which parses the last line printed after `map; topo; stime -c`.
    """
    lines = out.splitlines()
    start = None
    end = None
    for i, ln in enumerate(lines):
        if STIME_BEGIN_RE.match(ln.strip()):
            start = i + 1
        elif STIME_END_RE.match(ln.strip()) and start is not None:
            end = i
            break

    if start is not None and end is None:
        end = len(lines)
    search_text = "\n".join(lines[start:end]) if start is not None and end is not None and end > start else out

    area = None
    delay = None
    for m in AREA_DELAY_STIME_RE.finditer(search_text):
        area = float(m.group(1))
        delay = float(m.group(2))
    return area, delay


def _extract_recipe_heuristics(script_lines: List[str]) -> List[str]:
    """
    Reproduce OpenABC-D script generation behavior:
      - Ignore the header (`strash`, `write_bench ..._orig`)
      - Ignore the footer (final `write_bench ..._syn`, plus mapping/STA/resize commands)

    Equivalent to automate_synthesisScriptGen.py's `fileLines[2:-8]` slice, with
    extra filtering for robustness.
    """
    lines = [ln.strip() for ln in script_lines if ln.strip()]
    if len(lines) >= 10 and lines[0].lower().startswith("strash"):
        # Reference scripts are of the form:
        #   0: strash
        #   1: write_bench -l ..._orig.bench
        #   2..-9: heuristic ops (20 steps)
        #   -8..-1: write_bench ..._syn + mapping/timing/size ops
        candidate = lines[2:-8]
    else:
        candidate = lines

    drop_prefixes = ("write_bench", "read_bench", "read_lib", "read ", "strash", "echo")
    drop_cmds = {"dch", "map", "topo", "stime", "buffer", "upsize", "dnsize"}
    heuristics: List[str] = []
    for ln in candidate:
        low = ln.lower().strip()
        if any(low.startswith(p) for p in drop_prefixes):
            continue
        cmd = low.split()[0] if low.split() else ""
        if cmd in drop_cmds:
            continue
        heuristics.append(ln)
    return heuristics


def _build_recipe_run_commands(
    *,
    lib_path: Path,
    bench_path: Path,
    script_path: Path,
) -> str:
    script_lines = _read_script_lines(script_path)
    heuristics = _extract_recipe_heuristics(script_lines)

    cmds: List[str] = [
        f"read_lib {lib_path.as_posix()}",
        f"read_bench {bench_path.as_posix()}",
        "strash",
        *heuristics,
        # Dump the *final AIG* (pre-mapping) for proxy metrics.
        "echo __BENCH_BEGIN__",
        "write_bench -l /dev/stdout",
        "echo __BENCH_END__",
        # Tech mapping + global timing/area (dataset uses these, no dch/buffer/upsize/dnsize here).
        "map -B 0.9",
        "topo",
        "echo __STIME_BEGIN__",
        "stime -c",
    ]

    return "; ".join(cmds)


def run_recipe_metrics(
    *,
    abc_bin: str,
    lib_path: Path,
    bench_path: Path,
    scripts: List[Path],
    out_csv: str,
    stats_pkl: Optional[Path],
) -> int:
    design_key = _derive_design_key(bench_path)

    expected = None
    if stats_pkl is not None and stats_pkl.exists():
        import pickle

        with stats_pkl.open("rb") as f:
            expected = pickle.load(f)

    rows: List[Tuple[str, int, int, int, int, Optional[float], Optional[float], str]] = []
    for script_path in scripts:
        m = re.match(r"abc(\d+)\.script$", script_path.name)
        if not m:
            raise ValueError(f"Unexpected script name: {script_path.name} (expected abcN.script)")
        recipe_id = int(m.group(1))

        abc_cmds = _build_recipe_run_commands(lib_path=lib_path, bench_path=bench_path, script_path=script_path)
        out = _run_abc(abc_bin, abc_cmds, cwd=Path.cwd())

        bench_text = _extract_between_markers(out)
        and_g, not_g, lp = _bench_metrics_from_text(bench_text)
        area, delay = _extract_area_delay(out)

        note = ""
        if expected is not None and design_key in expected and recipe_id < len(expected[design_key][0]):
            exp_and = expected[design_key][0][recipe_id]
            exp_not = expected[design_key][1][recipe_id]
            exp_lp = expected[design_key][2][recipe_id]
            exp_area = expected[design_key][3][recipe_id]
            exp_delay = expected[design_key][4][recipe_id]
            if (and_g, not_g, lp) != (int(exp_and), int(exp_not), int(exp_lp)):
                note = f"mismatch_proxy exp=({exp_and},{exp_not},{exp_lp})"
            if area is not None and delay is not None:
                # Tolerate numeric formatting differences; just note mismatch.
                if abs(float(exp_area) - area) > 1e-6 or abs(float(exp_delay) - delay) > 1e-6:
                    note = (note + " " if note else "") + f"mismatch_ad exp=({exp_area},{exp_delay})"

        rows.append((design_key, recipe_id, and_g, not_g, lp, area, delay, note))

    # Output CSV (or stdout).
    header = ["Design", "RecipeID", "ANDgates", "NOTgates", "lpLen", "area", "delay"]
    if out_csv == "-":
        w = csv.writer(os.sys.stdout)
        w.writerow(header)
        for d, rid, a, n, lp, area, delay, _note in rows:
            w.writerow([d, rid, a, n, lp, "" if area is None else f"{area:.2f}", "" if delay is None else f"{delay:.2f}"])
    else:
        try:
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(header)
                for d, rid, a, n, lp, area, delay, _note in rows:
                    w.writerow([d, rid, a, n, lp, "" if area is None else f"{area:.2f}", "" if delay is None else f"{delay:.2f}"])
        except PermissionError:
            print("WARNING: cannot write CSV output file; re-run with --out - to print CSV to stdout")
            w = csv.writer(os.sys.stdout)
            w.writerow(header)
            for d, rid, a, n, lp, area, delay, _note in rows:
                w.writerow([d, rid, a, n, lp, "" if area is None else f"{area:.2f}", "" if delay is None else f"{delay:.2f}"])

    # Print a small comparison summary (if available).
    if expected is not None and design_key in expected:
        mism = [r for r in rows if r[-1]]
        if mism:
            print(f"NOTE: {len(mism)}/{len(rows)} recipes mismatch synthesisStatistics.pickle for design '{design_key}'.")
            for r in mism[:5]:
                print(f"  recipe {r[1]}: {r[-1]}")
        else:
            print(f"OK: all {len(rows)} recipes match synthesisStatistics.pickle for design '{design_key}'.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Time each command in ABC scripts (human-readable CSV output).")
    ap.add_argument("--mode", choices=["time", "metrics"], default="time", help="Run mode: per-command timing or per-recipe metrics")
    ap.add_argument("--bench", default="bench_openabcd/ac97_ctrl_orig.bench", help="Input .bench path")
    ap.add_argument(
        "--script",
        action="append",
        default=[
            "bench_openabcd/referenceScripts/referenceScripts/abc0.script",
            "bench_openabcd/referenceScripts/referenceScripts/abc1.script",
        ],
        help="ABC script path (can be repeated)",
    )
    ap.add_argument("--script-dir", default="bench_openabcd/referenceScripts/referenceScripts", help="Directory containing abcN.script")
    ap.add_argument("--recipe-start", type=int, default=0, help="First recipe id (abcN.script) for metrics mode")
    ap.add_argument("--recipe-end", type=int, default=10, help="Last recipe id (abcN.script) for metrics mode (inclusive)")
    ap.add_argument("--stats-pkl", default="test_pt/synthesisStatistics.pickle", help="Optional synthesisStatistics.pickle for comparison")
    ap.add_argument("--abc-bin", default="yosys-abc", help="ABC binary (default: yosys-abc)")
    ap.add_argument(
        "--lib",
        default=os.environ.get("ABC_LIB", "abc/Nangate45_typ.lib"),
        help="Liberty library path for mapping/STA (optional; also reads $ABC_LIB)",
    )
    ap.add_argument(
        "--out",
        default="test_time_results.csv",
        help="Output CSV path (use '-' for stdout)",
    )
    ap.add_argument(
        "--dump-dir",
        default="timing_dumps",
        help="Where to rewrite write_bench outputs (use '-' to skip writing benches)",
    )
    args = ap.parse_args()

    repo_root = Path.cwd()
    bench_path = (repo_root / args.bench).resolve() if not Path(args.bench).is_absolute() else Path(args.bench)
    if not bench_path.exists():
        raise FileNotFoundError(f"bench not found: {bench_path}")

    lib_path: Optional[Path] = None
    if args.lib:
        lp = (repo_root / args.lib).resolve() if not Path(args.lib).is_absolute() else Path(args.lib)
        lib_path = lp if lp.exists() else None

    if args.mode == "metrics":
        if lib_path is None:
            raise FileNotFoundError(f"Liberty library not found: {args.lib}")
        script_dir = (repo_root / args.script_dir).resolve() if not Path(args.script_dir).is_absolute() else Path(args.script_dir)
        scripts = [script_dir / f"abc{i}.script" for i in range(args.recipe_start, args.recipe_end + 1)]
        missing = [str(p) for p in scripts if not p.exists()]
        if missing:
            raise FileNotFoundError("Missing script files:\n" + "\n".join(missing))
        stats_pkl = (repo_root / args.stats_pkl).resolve() if args.stats_pkl else None
        return run_recipe_metrics(
            abc_bin=args.abc_bin,
            lib_path=lib_path,
            bench_path=bench_path,
            scripts=scripts,
            out_csv=args.out,
            stats_pkl=stats_pkl,
        )

    all_results: List[StepResult] = []

    for script_path_str in args.script:
        script_path = (repo_root / script_path_str).resolve() if not Path(script_path_str).is_absolute() else Path(script_path_str)
        if not script_path.exists():
            raise FileNotFoundError(f"script not found: {script_path}")

        script_name = script_path.name
        if args.dump_dir == "-":
            dump_dir = Path("-")
        else:
            dump_dir = (repo_root / args.dump_dir / script_name).resolve()
            try:
                dump_dir.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                dump_dir = Path("-")

        script_lines = _read_script_lines(script_path)
        script_cmds: List[Tuple[str, str]] = []
        for i, cmd in enumerate(script_lines):
            exe = _rewrite_write_bench(cmd, dump_dir, i)
            script_cmds.append((cmd, exe))

        init_cmds: List[Tuple[str, str]] = []
        if lib_path is not None:
            init_cmds.append((f"read_lib {lib_path}", f"read_lib {lib_path.as_posix()}"))
        else:
            init_cmds.append(
                (
                    "(no Liberty library provided)",
                    "echo WARNING: no Liberty library (map/stime/buffer/upsize/dnsize may fail)",
                )
            )
        init_cmds.append((f"read_bench {bench_path}", f"read_bench {bench_path.as_posix()}"))

        abc_cmds, meta = _build_abc_command_sequence(script_name=script_name, init_cmds=init_cmds, script_cmds=script_cmds)
        out = _run_abc(args.abc_bin, abc_cmds, cwd=repo_root)
        all_results.extend(_parse_output(out=out, script_name=script_name, meta=meta))

    if args.out == "-":
        w = csv.writer(os.sys.stdout)
        w.writerow(
            [
                "script",
                "stage",
                "step_index",
                "ok",
                "elapsed_seconds",
                "total_seconds",
                "command_original",
                "command_executed",
                "notes",
            ]
        )
        for r in all_results:
            w.writerow(
                [
                    r.script,
                    r.stage,
                    r.step_index,
                    int(r.ok),
                    "" if r.elapsed_seconds is None else f"{r.elapsed_seconds:.6f}",
                    "" if r.total_seconds is None else f"{r.total_seconds:.6f}",
                    r.command_original,
                    r.command_executed,
                    r.notes,
                ]
            )
        out_csv = None
    else:
        out_csv = (repo_root / args.out).resolve()
        try:
            _write_csv(all_results, out_csv)
        except PermissionError:
            out_csv = None
            print("WARNING: cannot write CSV output file; re-run with --out - to print CSV to stdout")
    _print_summary(all_results)
    if out_csv is not None:
        print(f"\nWrote CSV: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
