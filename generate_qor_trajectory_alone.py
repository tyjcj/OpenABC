#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple


ANSI_RE = re.compile(r"\x1B\[[0-9;]*m")
AREA_DELAY_STIME_RE = re.compile(r"Area\s*=\s*([0-9.]+).*?Delay\s*=\s*([0-9.]+)\s*ps", re.IGNORECASE)


def read_script_lines(path: Path) -> List[str]:
    lines: List[str] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def extract_heuristic_steps(lines: List[str], *, expected_len: int = 20) -> List[str]:
    """
    Extract the 20 heuristic operations from a reference abcN.script.

    For bench_openabcd/referenceScripts/referenceScripts/abcN.script, the layout is:
      1) strash
      2) write_bench -l <..._orig.bench>
      3..22) 20 heuristic steps
      23) write_bench -l <..._synN.bench>
      24..30) dch/map/topo/stime/buffer/upsize/dnsize
    """
    if len(lines) >= 2 + expected_len and lines[0].lower().startswith("strash") and lines[1].lower().startswith("write_bench"):
        steps = lines[2 : 2 + expected_len]
        return steps

    # Fallback: take the first `expected_len` non-I/O, non-mapping commands.
    drop_prefixes = ("write_bench", "read_bench", "read_lib", "read ", "strash", "echo")
    drop_cmds = {"dch", "map", "topo", "stime", "buffer", "upsize", "dnsize"}
    steps: List[str] = []
    for ln in lines:
        low = ln.lower().strip()
        if any(low.startswith(p) for p in drop_prefixes):
            continue
        cmd = low.split()[0] if low.split() else ""
        if cmd in drop_cmds:
            continue
        steps.append(ln)
        if len(steps) >= expected_len:
            break
    if len(steps) != expected_len:
        raise ValueError(f"Failed to extract {expected_len} heuristic steps (got {len(steps)}).")
    return steps


ABBREV_RULES: List[Tuple[re.Pattern[str], str]] = [
    (re.compile(r"^balance$", re.IGNORECASE), "b"),
    (re.compile(r"^rewrite\s*-z$", re.IGNORECASE), "rwz"),
    (re.compile(r"^rewrite$", re.IGNORECASE), "rw"),
    (re.compile(r"^refactor\s*-z$", re.IGNORECASE), "rfz"),
    (re.compile(r"^refactor$", re.IGNORECASE), "rf"),
    (re.compile(r"^resub\s*-z$", re.IGNORECASE), "rsz"),
    (re.compile(r"^resub$", re.IGNORECASE), "rs"),
    (re.compile(r"^rs\s+.*$", re.IGNORECASE), "rs"),
]


def abbreviate_steps(steps: List[str]) -> List[str]:
    out: List[str] = []
    for step in steps:
        s = step.strip()
        abbrev = None
        for pat, rep in ABBREV_RULES:
            if pat.match(s):
                abbrev = rep
                break
        out.append(abbrev if abbrev is not None else s)
    return out


def run_abc(abc_bin: str, abc_cmd: str, cwd: Path) -> str:
    proc = subprocess.run(
        [abc_bin, "-c", abc_cmd],
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return proc.stdout


def extract_area_delay_from_stime(output: str) -> Tuple[Optional[float], Optional[float]]:
    text = ANSI_RE.sub("", output)
    area = None
    delay = None
    for m in AREA_DELAY_STIME_RE.finditer(text):
        area = float(m.group(1))
        delay = float(m.group(2))
    return area, delay


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate per-step mapped QoR trajectory (Area/Delay) CSV.")
    ap.add_argument("--bench", default="bench_openabcd/ac97_ctrl_orig.bench", help="Input .bench path")
    ap.add_argument("--lib", default="abc/Nangate45_typ.lib", help="Liberty library path")
    ap.add_argument("--abc-bin", default="yosys-abc", help="ABC binary (default: yosys-abc)")
    ap.add_argument("--script-dir", default="bench_openabcd/referenceScripts", help="Directory containing abcN.script")
    ap.add_argument("--recipe-start", type=int, default=0, help="First recipe id (abcN.script)")
    ap.add_argument("--recipe-end", type=int, default=10, help="Last recipe id (inclusive)")
    ap.add_argument("--out", default="qor_trajectory_ac97_ctrl.csv", help="Output CSV path")
    ap.add_argument("--bias", type=float, default=0.9, help="map -B bias value (default: 0.9)")
    ap.add_argument("--steps", type=int, default=20, help="Number of heuristic steps (default: 20)")
    args = ap.parse_args()

    repo_root = Path.cwd()
    bench_path = (repo_root / args.bench).resolve() if not Path(args.bench).is_absolute() else Path(args.bench)
    lib_path = (repo_root / args.lib).resolve() if not Path(args.lib).is_absolute() else Path(args.lib)
    script_dir = (repo_root / args.script_dir).resolve() if not Path(args.script_dir).is_absolute() else Path(args.script_dir)

    if not bench_path.exists():
        raise FileNotFoundError(f"bench not found: {bench_path}")
    if not lib_path.exists():
        raise FileNotFoundError(f"lib not found: {lib_path}")
    if not script_dir.exists():
        raise FileNotFoundError(f"script-dir not found: {script_dir}")

    design = bench_path.name  # keep full filename, e.g. ac97_ctrl_orig.bench
    recipes = list(range(args.recipe_start, args.recipe_end + 1))

    header = ["Design", "Recipe"]
    for k in range(0, args.steps + 1):
        header += [f"delay_{k}", f"area_{k}"]

    out_csv = (repo_root / args.out).resolve() if args.out != "-" else None
    out_fh = None
    if out_csv is not None:
        try:
            out_fh = open(out_csv, "w", newline="", encoding="utf-8")
        except PermissionError:
            out_csv = None
            print("WARNING: cannot write CSV output file; re-run with --out - to print CSV to stdout")
    try:
        writer = csv.writer(out_fh if out_fh is not None else __import__("sys").stdout)
        writer.writerow(header)

        for rid in recipes:
            script_path = script_dir / f"abc{rid}.script"
            if not script_path.exists():
                raise FileNotFoundError(f"missing script: {script_path}")

            lines = read_script_lines(script_path)
            steps = extract_heuristic_steps(lines, expected_len=args.steps)
            recipe_abbrev = "; ".join(abbreviate_steps(steps))

            row: List[object] = [design, recipe_abbrev]
            for k in range(0, args.steps + 1):
                prefix = "; ".join(steps[:k])
                cmd = (
                    f"read_lib {lib_path.as_posix()}; "
                    f"read_bench {bench_path.as_posix()}; "
                    f"strash; "
                    f"{prefix + '; ' if prefix else ''}"
                    f"map -B {args.bias}; topo; stime -c"
                )
                out = run_abc(args.abc_bin, cmd, cwd=repo_root)
                area, delay = extract_area_delay_from_stime(out)
                row += [
                    "" if delay is None else f"{delay:.2f}",
                    "" if area is None else f"{area:.2f}",
                ]

            writer.writerow(row)
    finally:
        if out_fh is not None:
            out_fh.close()

    if out_csv is not None:
        print(f"Wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
