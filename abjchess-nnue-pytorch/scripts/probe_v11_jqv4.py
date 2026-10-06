"""Fast JQv4 source and optional native-loader probe for V11 training.

The source probe validates every file header, footer, UUID, feature identity,
and contiguous range without reading block payloads.  Full file SHA-256 is
opt-in because the production tree is large.  ``--native-batches`` adds a
small ctypes-only loader check, which is useful for diagnosing DLL/runtime
problems before starting a Torch process.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

from create_v11_jqv4_manifest import (
    ManifestError,
    enumerate_jqv4_sources,
    validate_expected_inventory,
)


def probe_tree(root: Path | str, *, verify_sha256: bool = False, shard_group: str = "1-4") -> dict[str, Any]:
    """Validate and summarize a JQv4 tree.

    ``enumerate_jqv4_sources`` performs the same structural checks used by the
    manifest generator.  The default skips payload hashing; callers that need
    hashes explicitly pass ``verify_sha256=True``.
    """

    resolved_root = Path(root).expanduser().resolve()
    sources = enumerate_jqv4_sources(resolved_root, hash_files=verify_sha256, allow_extra_shards=True, shard_group=shard_group)
    validate_expected_inventory(sources)
    shards: dict[str, dict[str, int]] = {}
    for source in sources:
        shard = shards.setdefault(source.shard, {
            "files": 0,
            "records": 0,
            "bytes": 0,
        })
        shard["files"] += 1
        shard["records"] += source.record_end - source.record_start
        shard["bytes"] += source.bytes

    return {
        "input_root": str(resolved_root),
        "shard_group": shard_group,
        "files": len(sources),
        "records": sum(item["records"] for item in shards.values()),
        "bytes": sum(item["bytes"] for item in shards.values()),
        "dataset_uuid": sources[0].dataset_uuid,
        "feature_set": sources[0].feature_set,
        "sha256_verified": bool(verify_sha256),
        "sha256_failures": [],
        "shards": {key: shards[key] for key in sorted(shards, key=int)},
    }


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read manifest {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("input_kind") != "jqv4":
        raise ManifestError(f"unsupported JQv4 manifest: {path}")
    return payload


def _verify_manifest_sources(summary: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    """Check that a manifest's listed source identities match its tree."""

    manifest = _load_manifest(manifest_path)
    root = Path(summary["input_root"]).resolve()
    listed = manifest.get("files")
    if not isinstance(listed, list) or not listed:
        raise ManifestError(f"manifest has no files: {manifest_path}")
    by_relative = {}
    for item in listed:
        if not isinstance(item, dict):
            raise ManifestError(f"manifest file entry is not an object: {manifest_path}")
        relative = item.get("relative_path")
        if not isinstance(relative, str) or not relative:
            raise ManifestError(f"manifest file entry has no relative_path: {manifest_path}")
        by_relative[relative.replace("\\", "/")] = item

    # Re-enumerate with hashes only for the manifest's listed files.  The
    # source probe already checked the complete tree; this pass avoids silently
    # accepting a hand-edited hash or path in the manifest.
    all_sources = enumerate_jqv4_sources(root, hash_files=True, allow_extra_shards=True, shard_group=summary["shard_group"])
    failures: list[str] = []
    for source in all_sources:
        item = by_relative.get(source.relative_path)
        if item is None:
            continue
        if item.get("sha256", "").lower() != source.sha256:
            failures.append(source.relative_path)
    if failures:
        raise ManifestError(
            "manifest source SHA-256 mismatch: " + ", ".join(failures[:5])
        )
    return {
        "manifest": str(manifest_path.resolve()),
        "partition": manifest.get("split", {}).get("partition", manifest.get("partition")),
        "files": len(listed),
        "total_records": manifest.get("total_records"),
        "teacher_dataset_sha256": manifest.get("teacher_dataset_sha256"),
    }


class NativeProbeError(RuntimeError):
    """Raised when the optional ctypes loader probe cannot start."""


class _SparseBatch(ctypes.Structure):
    _fields_ = [
        ("num_inputs", ctypes.c_int),
        ("size", ctypes.c_int),
        ("is_white", ctypes.POINTER(ctypes.c_float)),
        ("outcome", ctypes.POINTER(ctypes.c_float)),
        ("score", ctypes.POINTER(ctypes.c_float)),
        ("num_active_white_features", ctypes.c_int),
        ("num_active_black_features", ctypes.c_int),
        ("max_active_features", ctypes.c_int),
        ("white", ctypes.POINTER(ctypes.c_int)),
        ("black", ctypes.POINTER(ctypes.c_int)),
        ("white_values", ctypes.POINTER(ctypes.c_float)),
        ("black_values", ctypes.POINTER(ctypes.c_float)),
        ("layer_stack_indices", ctypes.POINTER(ctypes.c_int)),
        ("eval_weight", ctypes.POINTER(ctypes.c_float)),
        # Append-only V11 tail: q0.8 blend represented as float / 255.
        ("layer_stack_blend", ctypes.POINTER(ctypes.c_float)),
    ]


def _loader_error(loader: Any) -> str:
    getter = getattr(loader, "get_training_data_loader_last_error", None)
    if getter is None:
        return ""
    getter.restype = ctypes.c_char_p
    getter.argtypes = []
    value = getter()
    return value.decode("utf-8", errors="replace") if value else ""


def probe_native_loader(
    manifest: Path | str,
    loader_path: Path | str,
    *,
    feature_set: str = "HalfKAv2_hm_jieqi_v11^",
    batch_size: int = 256,
    batches: int = 1,
) -> dict[str, Any]:
    """Load ``batches`` JQv4 batches through the native ABI without Torch."""

    if batch_size <= 0 or batches <= 0:
        raise ValueError("batch_size and batches must be positive")
    manifest = Path(manifest).expanduser().resolve()
    loader_path = Path(loader_path).expanduser().resolve()
    if not manifest.is_file():
        raise NativeProbeError(f"manifest is missing: {manifest}")
    if not loader_path.is_file():
        raise NativeProbeError(f"loader DLL is missing: {loader_path}")

    dll_dir_handle = None
    if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
        dll_dir_handle = os.add_dll_directory(str(loader_path.parent))
    try:
        try:
            loader = ctypes.CDLL(str(loader_path))
        except OSError as exc:
            raise NativeProbeError(
                f"cannot load {loader_path}: {exc}. Check that the DLL is the "
                "V11 static-runtime build and does not require "
                "libstdc++-6.dll, libgcc_s_seh-1.dll, or libwinpthread-1.dll"
            ) from exc

        abi = getattr(loader, "training_data_loader_abi_version", None)
        if abi is None or abi() != 110:
            raise NativeProbeError("V11 loader ABI mismatch; expected 110")
        creator = getattr(loader, "create_sparse_batch_stream_v2", None)
        if creator is not None:
            creator.restype = ctypes.c_void_p
            creator.argtypes = [
                ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                ctypes.c_bool, ctypes.c_bool, ctypes.c_int, ctypes.c_int,
                ctypes.c_int,
            ]
        else:
            creator = loader.create_sparse_batch_stream
            creator.restype = ctypes.c_void_p
            creator.argtypes = [
                ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                ctypes.c_bool, ctypes.c_bool, ctypes.c_int,
            ]
        fetch = loader.fetch_next_sparse_batch
        fetch.restype = ctypes.POINTER(_SparseBatch)
        fetch.argtypes = [ctypes.c_void_p]
        destroy_batch = loader.destroy_sparse_batch
        destroy_batch.restype = None
        destroy_batch.argtypes = [ctypes.POINTER(_SparseBatch)]
        destroy_stream = loader.destroy_sparse_batch_stream
        destroy_stream.restype = None
        destroy_stream.argtypes = [ctypes.c_void_p]

        encoded_feature = feature_set.encode("utf-8")
        encoded_manifest = os.fsencode(str(manifest))
        if getattr(loader, "create_sparse_batch_stream_v2", None) is not None:
            stream = creator(encoded_feature, 1, encoded_manifest, batch_size,
                             False, False, 0, 0, 1)
        else:
            stream = creator(encoded_feature, 1, encoded_manifest, batch_size,
                             False, False, 0)
        if not stream:
            raise NativeProbeError(
                "native stream creation failed: "
                + (_loader_error(loader) or "unknown loader error")
            )

        result_batches: list[dict[str, Any]] = []
        try:
            for index in range(batches):
                pointer = fetch(stream)
                if not pointer:
                    raise NativeProbeError(
                        f"native fetch failed at batch {index + 1}: "
                        + (_loader_error(loader) or "end of stream")
                    )
                try:
                    batch = pointer.contents
                    if batch.size <= 0 or batch.num_inputs <= 0:
                        raise NativeProbeError(
                            f"native batch {index + 1} has invalid shape "
                            f"size={batch.size}, num_inputs={batch.num_inputs}"
                        )
                    is_v11 = feature_set.startswith("HalfKAv2_hm_jieqi_v11")
                    blend_values = None
                    if is_v11 and not bool(batch.layer_stack_blend):
                        raise NativeProbeError(
                            "V11 native batch is missing the append-only "
                            "layer_stack_blend tail"
                        )
                    if bool(batch.layer_stack_blend):
                        blend_values = [
                            float(value)
                            for value in ctypes.cast(
                                batch.layer_stack_blend,
                                ctypes.POINTER(ctypes.c_float * batch.size),
                            ).contents
                        ]
                        if any(
                            not math.isfinite(value) or value < 0.0 or value > 1.0
                            for value in blend_values
                        ):
                            raise NativeProbeError(
                                "native layer_stack_blend values must be finite "
                                "and lie in [0,1]"
                            )
                    result_batches.append({
                        "index": index + 1,
                        "rows": int(batch.size),
                        "num_inputs": int(batch.num_inputs),
                        "max_active_features": int(batch.max_active_features),
                        "white_active": int(batch.num_active_white_features),
                        "black_active": int(batch.num_active_black_features),
                        "has_eval_weight": bool(batch.eval_weight),
                        "has_layer_stack_blend": bool(batch.layer_stack_blend),
                        "layer_stack_blend_first": (
                            blend_values[0] if blend_values else None
                        ),
                        "layer_stack_blend_min": (
                            min(blend_values) if blend_values else None
                        ),
                        "layer_stack_blend_max": (
                            max(blend_values) if blend_values else None
                        ),
                    })
                finally:
                    destroy_batch(pointer)
        finally:
            destroy_stream(stream)
        return {
            "loader": str(loader_path),
            "manifest": str(manifest),
            "feature_set": feature_set,
            "batches": result_batches,
        }
    finally:
        if dll_dir_handle is not None:
            dll_dir_handle.close()


def _format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    number = float(value)
    for unit in units:
        if number < 1024.0 or unit == units[-1]:
            return f"{number:.1f} {unit}" if unit != "B" else f"{int(number)} B"
        number /= 1024.0
    return f"{value} B"


def _print_human(summary: dict[str, Any]) -> None:
    print("JQv4 V11 probe")
    print(f"input_root={summary['input_root']}")
    print(f"files={summary['files']} records={summary['records']} bytes={_format_bytes(summary['bytes'])}")
    print(f"dataset_uuid={summary['dataset_uuid']}")
    print(f"feature_set={summary['feature_set']}")
    print(f"sha256_verified={str(summary['sha256_verified']).lower()}")
    for shard, item in summary["shards"].items():
        print(f"shard={shard} files={item['files']} records={item['records']} bytes={_format_bytes(item['bytes'])}")
    for manifest in summary.get("manifests", []):
        print(
            f"manifest={manifest['manifest']} partition={manifest['partition']} "
            f"files={manifest['files']} total_records={manifest['total_records']}"
        )
    native = summary.get("native")
    if native is not None:
        print(f"native_loader={native['loader']}")
        for item in native["batches"]:
            print(
                f"native_batch={item['index']} rows={item['rows']} "
                f"inputs={item['num_inputs']} "
                f"eval_weight={str(item['has_eval_weight']).lower()} "
                f"layer_stack_blend={str(item['has_layer_stack_blend']).lower()}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root", type=Path,
        default=Path("data"),
    )
    parser.add_argument("--manifest", action="append", type=Path,
                        help="optional manifest to verify; may be repeated")
    parser.add_argument("--shard-group", default="1-4", choices=["1-4", "5-8", "9-12", "13-16"])
    parser.add_argument("--verify-sha256", action="store_true")
    parser.add_argument("--loader", type=Path,
                        help="native loader DLL/SO for --native-batches")
    parser.add_argument("--native-batches", type=int, default=0,
                        help="fetch this many batches through the native ABI")
    parser.add_argument("--feature-set", default="HalfKAv2_hm_jieqi_v11^")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    try:
        summary = probe_tree(args.input_root, verify_sha256=args.verify_sha256, shard_group=args.shard_group)
        if args.manifest:
            summary["manifests"] = [
                _verify_manifest_sources(summary, path)
                for path in args.manifest
            ]
        if args.native_batches:
            if args.loader is None:
                raise NativeProbeError("--loader is required with --native-batches")
            if not args.manifest:
                raise NativeProbeError("--manifest is required with --native-batches")
            summary["native"] = probe_native_loader(
                args.manifest[0], args.loader,
                feature_set=args.feature_set,
                batch_size=args.batch_size,
                batches=args.native_batches,
            )
    except (ManifestError, NativeProbeError, OSError, ValueError) as exc:
        print(f"probe_error={exc}", file=sys.stderr)
        return 2

    if args.as_json:
        print(json.dumps(summary, ensure_ascii=True, sort_keys=True, indent=2))
    else:
        _print_human(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
