#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch process all .bench files and generate Area CSV for each.
Similar to generate_csv_batch.py but computes AND-gate counts (area).
"""

import subprocess
import csv
import re
import os
import argparse
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from functools import lru_cache
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import threading

# ---------------- Default config ----------------
# ---------------- Default config ----------------
ABC_BIN_DEFAULT = "/usr/local/bin/abc"
ABC_RC_DEFAULT = "/mnt/d/allproject/LSOformer/GraphMT-main/lsoformer/abc/abc.rc"
# BENCH_DIR_DEFAULT = "/mnt/d/allproject/LSOformer/GraphMT-main/lsoformer/datasets/EPFL"
BENCH_DIR_DEFAULT = "/mnt/d/allproject/LSOformer/GraphMT-main/lsoformer/datasets/EPFL_ctrl"
SCRIPTS_DIR_DEFAULT = "/mnt/d/allproject/LSOformer/GraphMT-main/lsoformer/data_files_epfl/referenceScripts"
OUTPUT_DIR_DEFAULT = "/mnt/d/allproject/LSOformer/GraphMT-main/lsoformer/data_files_epfl/dataset"
DEBUG_DIR_DEFAULT = "debug_abc_outputs"

SCRIPT_LINE_START = 3
SCRIPT_LINE_END = 22
TOTAL_STEPS = SCRIPT_LINE_END - SCRIPT_LINE_START + 1

SCRIPT_INDEX_RE = re.compile(r'(\d+)')
ALIAS_RE = re.compile(r"^alias\s+(\S+)\s+\"([^\"]+)\"")
AREA_PATTERN = re.compile(r"\band\s*=\s*(\d+)", re.IGNORECASE)

_thread_local = threading.local()

# ---------------- Utility functions ----------------
def run_abc(abc_bin: str, abc_rc: str, cmd: str) -> subprocess.CompletedProcess:
    """Run abc with rc file and aliases active."""
    full_cmd = f"source {abc_rc}; {cmd}"
    try:
        cp = subprocess.run(
            [abc_bin, "-F", abc_rc, "-c", full_cmd],
            capture_output=True, text=True, timeout=60
        )
        return cp
    except subprocess.TimeoutExpired:
        class MockProcess:
            def __init__(self):
                self.stdout = "TIMEOUT"
                self.stderr = "Command timed out"
        return MockProcess()

def parse_area(out_text: str) -> Optional[int]:
    """Parse 'and' from print_stats output."""
    if not out_text or "TIMEOUT" in out_text or "ERROR" in out_text:
        return None
    m = AREA_PATTERN.findall(out_text)
    if m:
        return int(m[-1])
    # fallback
    for ln in out_text.splitlines():
        if "and" in ln.lower():
            ints = re.findall(r"(\d+)", ln)
            if ints:
                return int(ints[-1])
    return None

def get_thread_cache():
    if not hasattr(_thread_local, "cache"):
        _thread_local.cache = {}
    return _thread_local.cache

def eval_prefix_get_area(abc_bin: str, abc_rc: str, bench_path: str, prefix_key: str) -> Tuple[Optional[int], str]:
    """Apply prefix steps to bench and return area (with thread-local caching)."""
    cache = get_thread_cache()
    cache_key = f"{bench_path}:{prefix_key}:area"
    if cache_key in cache:
        return cache[cache_key]

    prefix_steps = prefix_key.split(";") if prefix_key else []
    bench_esc = str(bench_path).replace('"', '\\"')
    variants = [
        f'read_bench "{bench_esc}"; strash; ' + "; ".join(prefix_steps) + "; print_stats",
        f'read "{bench_esc}"; strash; ' + "; ".join(prefix_steps) + "; print_stats",
    ]
    last_out = ""
    for cmd in variants:
        try:
            cp = run_abc(abc_bin, abc_rc, cmd)
            out = (cp.stdout or "") + "\n" + (cp.stderr or "")
            last_out += "\n=== CMD: " + cmd + " ===\n" + out
            area = parse_area(out)
            if area is not None:
                result = (area, out)
                cache[cache_key] = result
                return result
        except Exception as e:
            last_out += f"\n=== ERROR: {cmd} - {e}\n"
            continue
    result = (None, last_out)
    cache[cache_key] = result
    return result

def read_script_lines(script_path: Path) -> List[str]:
    with script_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    selected = lines[SCRIPT_LINE_START - 1:SCRIPT_LINE_END]
    return [ln.strip() for ln in selected if ln.strip()]

def build_custom_mapping() -> List[Tuple[re.Pattern, str]]:
    """Custom mapping rules for commands with parameters."""
    rules = [
        (re.compile(r"^balance$"), "b"),
        (re.compile(r"^rewrite\s*-z$"), "rwz"),
        (re.compile(r"^rewrite$"), "rw"),
        (re.compile(r"^refactor\s*-z$"), "rfz"),
        (re.compile(r"^refactor$"), "rf"),
        (re.compile(r"^resub\s*-z$"), "rsz"),
        (re.compile(r"^resub$"), "rs"),
        (re.compile(r"^rs\s+.*"), "rs"),
    ]
    return rules

def map_steps_to_alias(steps: List[str], custom_rules: List[Tuple[re.Pattern, str]]) -> List[str]:
    mapped = []
    for step in steps:
        replaced = None
        for pattern, alias in custom_rules:
            if pattern.match(step):
                replaced = alias
                break
        if replaced is None:
            replaced = step
        mapped.append(replaced)
    return mapped

def process_single_script(args_tuple):
    (idx, script_path, bench_path, abc_bin, abc_rc, init_area, debug_dir, bench_name, custom_rules) = args_tuple

    steps = read_script_lines(script_path)
    alias_steps = map_steps_to_alias(steps, custom_rules)
    recipe_str = "; ".join(alias_steps)

    areas = []
    last_valid = init_area if init_area is not None else 0

    current_prefix = []
    for i, step in enumerate(steps, 1):
        current_prefix.append(step)
        prefix_key = ";".join(current_prefix)
        area, raw_out = eval_prefix_get_area(abc_bin, abc_rc, bench_path, prefix_key)

        if area is None:
            if debug_dir:
                bench_debug_dir = Path(debug_dir) / bench_name
                os.makedirs(bench_debug_dir, exist_ok=True)
                dbgfile = bench_debug_dir / f"script{idx}_step{i}.log"
                dbgfile.write_text(raw_out)
            area = last_valid
        else:
            last_valid = area

        areas.append(area)

    return [bench_name, recipe_str, TOTAL_STEPS] + areas, None

def process_single_bench(abc_bin, abc_rc, bench_path, scripts_dir, output_dir, debug_dir, max_workers=4):
    bench_name = Path(bench_path).stem
    out_csv = Path(output_dir) / f"{bench_name}.csv"
    bench_debug_dir = Path(debug_dir) / bench_name
    os.makedirs(bench_debug_dir, exist_ok=True)

    start_time = time.time()

    try:
        init_cp = run_abc(abc_bin, abc_rc, f'read_bench "{bench_path}"; strash; print_stats')
        init_text = (init_cp.stdout or "") + "\n" + (init_cp.stderr or "")
    except Exception as e:
        init_text = f"EXCEPTION: {e}"
    init_area = parse_area(init_text)
    if init_area is None:
        Path(bench_debug_dir).joinpath(f"init_{bench_name}.txt").write_text(init_text)

    header = ["Design", "Recipe", "TotalSteps"] + [f"Area_{i+1}" for i in range(TOTAL_STEPS)]
    rows = []

    script_files = sorted(Path(scripts_dir).glob("*.script"),
                          key=lambda p: int(SCRIPT_INDEX_RE.search(p.name).group(1)))
    custom_rules = build_custom_mapping()

    script_args = []
    for idx, script_path in enumerate(script_files):
        script_args.append((idx, script_path, bench_path, abc_bin, abc_rc,
                            init_area, debug_dir, bench_name, custom_rules))

    with ThreadPoolExecutor(max_workers=min(max_workers, len(script_files))) as executor:
        with tqdm(total=len(script_files), desc=f"Processing {bench_name}", unit="script", leave=False) as pbar:
            future_to_idx = {executor.submit(process_single_script, args): i for i, args in enumerate(script_args)}
            results = [None] * len(script_files)
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result, warning = future.result()
                    if result:
                        results[idx] = result
                    if warning:
                        tqdm.write(warning)
                except Exception as e:
                    tqdm.write(f"Error processing script {idx}: {e}")
                pbar.update(1)

    rows = [r for r in results if r is not None]

    os.makedirs(output_dir, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    processing_time = time.time() - start_time
    print(f"Completed {bench_name} in {processing_time:.2f}s -> {out_csv}")

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="Batch process all .bench files and generate Area CSVs")
    ap.add_argument("--abc", default=ABC_BIN_DEFAULT, help="abc binary")
    ap.add_argument("--rc", default=ABC_RC_DEFAULT, help="abc.rc path")
    ap.add_argument("--bench-dir", default=BENCH_DIR_DEFAULT, help="directory containing .bench files")
    ap.add_argument("--scripts-dir", default=SCRIPTS_DIR_DEFAULT, help="directory with .script files")
    ap.add_argument("--output-dir", default=OUTPUT_DIR_DEFAULT, help="output directory for CSV files")
    ap.add_argument("--debug-dir", default=DEBUG_DIR_DEFAULT, help="where to dump debug logs")
    ap.add_argument("--max-processes", type=int, default=8, help="max number of processes")
    ap.add_argument("--max-threads", type=int, default=4, help="max threads per process")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.debug_dir, exist_ok=True)

    bench_dir_path = Path(args.bench_dir)
    bench_files = list(bench_dir_path.glob("*.bench"))

    if not bench_files:
        print(f"No .bench files found in {args.bench_dir}")
        return

    if args.max_processes is None:
        max_processes = min(cpu_count(), len(bench_files))
    else:
        max_processes = args.max_processes

    print(f"Found {len(bench_files)} .bench files to process")

    process_args = [
        (args.abc, args.rc, bench_file, args.scripts_dir, args.output_dir, args.debug_dir, args.max_threads)
        for bench_file in bench_files
    ]

    with ProcessPoolExecutor(max_workers=max_processes) as executor:
        with tqdm(total=len(bench_files), desc="Processing bench files", unit="file") as pbar:
            future_to_file = {executor.submit(process_single_bench, *pa): pa[2] for pa in process_args}
            for future in as_completed(future_to_file):
                bench_file = future_to_file[future]
                try:
                    future.result()
                except Exception as e:
                    tqdm.write(f"Error processing {bench_file.name}: {e}")
                pbar.update(1)

if __name__ == "__main__":
    main()
    print("Batch processing (area) completed!")
"""

# Original command with parallelization:
python3 generate_area_csv.py \
  --abc /usr/local/bin/abc \
  --rc /mnt/d/Users/14696/Desktop/postStudy/OpenABC-master/abc/abc.rc \
  --bench-dir /mnt/d/Users/14696/Desktop/postStudy/OpenABC-master/EPFL_bench/bench_area \
  --scripts-dir /mnt/d/Users/14696/Desktop/postStudy/OpenABC-master/bench_openabcd/referenceScripts  \
  --output-dir EPFL_bench/area \
  --debug-dir debug_logs \
  --max-processes 1 \
  --max-threads 2
"""