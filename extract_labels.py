#!/usr/bin/env python3
"""
Extract label information from OpenABC-D .pt archives into a CSV.

The .pt files in `test_pt/` are actually zip archives that contain a
`archive/data.pkl` pickle plus the raw tensor storages in `archive/data/*`.
This script unpickles them without needing torch/torch_geometric by
providing lightweight stubs, then writes a summary CSV with the label
fields that appear in the dataset (PIs, POs, node counts, depth, etc.).
"""

from __future__ import annotations

import csv
import io
import pickle
import sys
import types
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


def install_stubs() -> Dict[str, Optional[types.ModuleType]]:
    """Install minimal stub modules so the pickle can be loaded without torch."""
    backups: Dict[str, Optional[types.ModuleType]] = {}
    modules = {
        "torch": types.ModuleType("torch"),
        "torch._utils": types.ModuleType("torch._utils"),
        "torch_geometric": types.ModuleType("torch_geometric"),
        "torch_geometric.data": types.ModuleType("torch_geometric.data"),
        "torch_geometric.data.data": types.ModuleType("torch_geometric.data.data"),
    }
    for name, module in modules.items():
        backups[name] = sys.modules.get(name)
        sys.modules[name] = module
    return backups


def restore_stubs(backups: Dict[str, Optional[types.ModuleType]]) -> None:
    for name, module in backups.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def load_data_from_archive(zf: zipfile.ZipFile) -> Any:
    """Load the `archive/data.pkl` payload from a zipfile containing a .pt."""
    raw = zf.read("archive/data.pkl")
    storages_cache: Dict[str, np.ndarray] = {}

    backups = install_stubs()
    torch_mod = sys.modules["torch"]
    utils_mod = sys.modules["torch._utils"]
    data_module = sys.modules["torch_geometric.data.data"]

    class Data:  # minimal stand-in for torch_geometric.data.Data
        def __init__(self) -> None:
            self.__dict__["_store"] = {}

        def __setattr__(self, key: str, value: Any) -> None:
            self._store[key] = value

        def __getattr__(self, key: str) -> Any:
            try:
                return self._store[key]
            except KeyError as exc:
                raise AttributeError(key) from exc

        def __setstate__(self, state: Dict[str, Any]) -> None:
            self.__dict__["_store"] = state

        def items(self):
            return self._store.items()

        def keys(self):
            return self._store.keys()

    data_module.Data = Data

    class LongStorage:
        pass

    torch_mod.LongStorage = LongStorage

    def dtype_from_storage(storage_type: Any) -> Any:
        name = getattr(storage_type, "__name__", str(storage_type))
        if "Long" in name:
            return np.int64
        if "Int" in name:
            return np.int32
        if "Float" in name:
            return np.float32
        if "Double" in name:
            return np.float64
        if "Byte" in name:
            return np.uint8
        raise ValueError(f"Unknown storage type {name}")

    def _rebuild_tensor_v2(
        storage: np.ndarray,
        storage_offset: int,
        size: Iterable[int],
        stride: Iterable[int],
        requires_grad: bool,
        backward_hooks: Any,
        metadata: Any = None,
    ) -> np.ndarray:
        """Rebuild a tensor as a NumPy view with the original strides."""
        itemsize = storage.dtype.itemsize
        np_stride = tuple(s * itemsize for s in stride)
        return np.ndarray(
            shape=tuple(size),
            dtype=storage.dtype,
            buffer=storage,
            offset=storage_offset * itemsize,
            strides=np_stride,
        )

    utils_mod._rebuild_tensor_v2 = _rebuild_tensor_v2

    def persistent_load(saved_id: Any) -> np.ndarray:
        if isinstance(saved_id, tuple) and saved_id[0] == "storage":
            _, storage_type, key, _location, size = saved_id
            if key in storages_cache:
                return storages_cache[key]
            data = zf.read(f"archive/data/{key}")
            dt = dtype_from_storage(storage_type)
            arr = np.frombuffer(data, dtype=dt, count=size)
            storages_cache[key] = arr
            return arr
        raise ValueError(f"Unhandled persistent id {saved_id}")

    class CustomUnpickler(pickle.Unpickler):
        def persistent_load(self, saved_id: Any) -> Any:
            return persistent_load(saved_id)

    try:
        return CustomUnpickler(io.BytesIO(raw)).load()
    finally:
        restore_stubs(backups)


def load_pt(path: Path) -> Any:
    """Open a .pt or .pt.zip file and return the decoded Data object."""
    with zipfile.ZipFile(path) as outer:
        names = set(outer.namelist())
        if "archive/data.pkl" in names:
            return load_data_from_archive(outer)

        # Handle wrapper zip (e.g., *.pt.zip) that contains a .pt file.
        inner_pts = [n for n in names if n.endswith(".pt")]
        if len(inner_pts) == 1:
            inner_bytes = outer.read(inner_pts[0])
            with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
                return load_data_from_archive(inner)

        raise ValueError(f"Cannot find archive/data.pkl inside {path}")


def to_int(value: Any) -> Optional[int]:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return int(value)
        return value.size
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (list, tuple)) and len(value) == 1:
        inner = value[0]
        if isinstance(inner, (np.integer, int)):
            return int(inner)
        return inner
    return None


def extract_labels(data: Any) -> Dict[str, Any]:
    """Pick out the fields described under Labels."""
    labels: Dict[str, Any] = {}

    labels["pi"] = to_int(getattr(data, "pi", None))
    labels["po"] = to_int(getattr(data, "po", None))
    labels["num_nodes"] = getattr(data, "__num_nodes__", None)
    labels["and_nodes"] = to_int(getattr(data, "and_nodes", None))
    labels["not_edges"] = to_int(getattr(data, "not_edges", None))
    labels["edge_count"] = getattr(getattr(data, "edge_index", None), "shape", (None, None))[1]
    labels["depth"] = to_int(getattr(data, "longest_path", None))

    des_name = getattr(data, "desName", None)
    labels["ip_name"] = des_name[0] if isinstance(des_name, list) and des_name else des_name
    labels["syn_id"] = to_int(getattr(data, "synID", None))
    labels["step_id"] = to_int(getattr(data, "stepID", None))

    return labels


def find_pt_files(base: Path) -> List[Path]:
    pts = sorted(p for p in base.glob("*.pt") if p.is_file())
    pt_set = {p.name for p in pts}

    extra = []
    for zpath in sorted(p for p in base.glob("*.pt.zip") if p.is_file()):
        counterpart = Path(zpath.stem)  # removes only .zip, leaves .pt
        if counterpart.name in pt_set:
            continue  # prefer the plain .pt copy
        extra.append(zpath)

    return pts + extra


def main() -> None:
    base = Path("test_pt")
    if not base.exists():
        print("test_pt directory not found", file=sys.stderr)
        sys.exit(1)

    out_path = base / "labels.csv"
    rows = []
    for path in find_pt_files(base):
        try:
            data = load_pt(path)
            labels = extract_labels(data)
        except Exception as exc:  # noqa: BLE001 - want to continue on failures
            print(f"[WARN] Failed to read {path}: {exc}", file=sys.stderr)
            continue

        row = {"file": path.name}
        row.update(labels)
        rows.append(row)

    if not rows:
        print("No .pt files processed, CSV not written", file=sys.stderr)
        sys.exit(1)

    fieldnames = [
        "file",
        "ip_name",
        "syn_id",
        "step_id",
        "pi",
        "po",
        "num_nodes",
        "and_nodes",
        "not_edges",
        "edge_count",
        "depth",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
