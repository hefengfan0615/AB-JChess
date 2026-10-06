"""Produce an authenticated V11 layer-stack coverage report.

The production runner can provide counts collected while consuming the native
JQv4 stream through ``--counts``.  Keeping the scanner's output independent of
Torch makes it usable on the data-preparation host as well as the trainer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import head_balance_v11
from head_balance_v11 import make_coverage_report, write_coverage_report


DEFAULT_FEATURE_NAME = "HalfKAv2_hm_jieqi_v11^"


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _validate_training_manifest(path: Path) -> dict:
    """Authenticate the manifest before opening the native stream."""

    # Keep this import lazy.  The report parser is useful on a data-preparation
    # host where the optional native loader DLL is not installed.
    try:
        from train_v11 import read_jqv4_manifest
        payload = read_jqv4_manifest(path)
    except (ImportError, OSError, RuntimeError, SystemExit) as exc:
        raise ValueError(f"cannot validate JQv4 manifest {path}: {exc}") from exc
    if payload is None:
        raise ValueError("head coverage requires an authenticated JQv4 manifest")
    if payload.get("split", {}).get("partition") != "train":
        raise ValueError("head coverage requires a training manifest")
    feature_name = payload.get("feature_set")
    if feature_name not in {
        "HalfKAv2_hm_jieqi_v11", "HalfKAv2_hm_jieqi_v11^",
    }:
        raise ValueError("head coverage requires a V11 Jieqi feature manifest")
    return payload


def _parse_counts(value: str) -> list[int]:
    try:
        counts = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise SystemExit(f"invalid --counts value: {exc}") from exc
    if len(counts) != 16 or any(item < 0 for item in counts):
        raise SystemExit("--counts must contain 16 non-negative integers")
    return counts


def _manifest_hash(path: Path) -> str:
    """Return the SHA-256 of the exact authenticated manifest bytes."""

    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read manifest {path}: {exc}") from exc
    if not raw:
        raise ValueError(f"manifest is empty: {path}")
    return hashlib.sha256(raw).hexdigest()


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _head_indices(batch):
    """Extract and validate the layer-stack floor from a native batch."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - scanner requires Torch
        raise RuntimeError("head coverage scanning requires PyTorch") from exc
    if isinstance(batch, dict):
        indices = batch.get("layer_stack_indices")
        if indices is None:
            raise RuntimeError("native V11 batch has no layer_stack_indices")
    else:
        if len(batch) < 10:
            raise RuntimeError("native V11 stream returned an invalid batch")
        indices = batch[8]
    indices = torch.as_tensor(indices)
    if indices.ndim == 2 and indices.shape[1] == 1:
        indices = indices[:, 0]
    if indices.ndim != 1:
        raise RuntimeError("native V11 layer_stack_indices must be a vector")
    if indices.dtype not in {
        torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
    }:
        raise RuntimeError("native V11 layer_stack_indices must use an integral dtype")
    selected = indices.detach().to(device="cpu", dtype=torch.long)
    if bool(((selected < 0) | (selected >= head_balance_v11.HEAD_COUNT)).any().item()):
        raise RuntimeError("native V11 stream returned an invalid head index")
    return selected


def _dataset_module():
    """Load the native dataset module, allowing tests to inject a stream."""

    module = globals().get("nnue_dataset")
    if module is not None:
        return module
    try:
        import nnue_dataset
    except ImportError as exc:  # pragma: no cover - scanner requires Torch
        raise RuntimeError("head coverage scanning requires nnue_dataset and PyTorch") from exc
    globals()["nnue_dataset"] = nnue_dataset
    return nnue_dataset


def scan_head_coverage(
    manifest: str | Path,
    *,
    records: int,
    output: str | Path,
    cap: float = head_balance_v11.DEFAULT_CAP,
    batch_size: int = 2048,
    num_workers: int = 1,
    feature_name: str = DEFAULT_FEATURE_NAME,
) -> dict:
    """Consume a bounded native V11 stream and write an authenticated report."""

    if isinstance(records, bool) or not isinstance(records, int) or records <= 0:
        raise ValueError("records must be a positive integer")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError("num_workers must be a non-negative integer")
    try:
        cap_value = float(cap)
    except (TypeError, ValueError) as exc:
        raise ValueError("cap must be finite and positive") from exc
    if not math.isfinite(cap_value) or cap_value <= 0.0:
        raise ValueError("cap must be finite and positive")
    if feature_name not in {"HalfKAv2_hm_jieqi_v11", DEFAULT_FEATURE_NAME}:
        raise ValueError("feature_name must be a V11 Jieqi feature set")

    manifest_path = Path(manifest).resolve()
    output_path = Path(output).resolve()
    payload = _validate_training_manifest(manifest_path)
    manifest_feature = payload.get("feature_set")
    if manifest_feature != feature_name:
        raise ValueError(
            f"coverage feature {feature_name!r} does not match manifest "
            f"feature_set {manifest_feature!r}"
        )
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - scanner requires Torch
        raise RuntimeError("head coverage scanning requires PyTorch") from exc
    nnue_dataset = _dataset_module()

    native_batch_size = min(records, batch_size)
    dataset = nnue_dataset.SparseBatchDataset(
        feature_name,
        str(manifest_path),
        native_batch_size,
        cyclic=False,
        num_workers=num_workers,
        filtered=False,
        random_fen_skipping=0,
        device="cpu",
        rank=0,
        world_size=1,
    )
    iterator = iter(dataset)
    counts = torch.zeros(head_balance_v11.HEAD_COUNT, dtype=torch.int64)
    scanned = 0
    try:
        while scanned < records:
            try:
                batch = next(iterator)
            except StopIteration as exc:
                raise RuntimeError(
                    f"native stream ended after {scanned} records; "
                    f"the explicit budget is {records}"
                ) from exc
            selected = _head_indices(batch)
            take = min(records - scanned, int(selected.shape[0]))
            if take <= 0:
                raise RuntimeError("native V11 stream returned an empty batch")
            counts += torch.bincount(
                selected[:take], minlength=head_balance_v11.HEAD_COUNT)
            scanned += take
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()

    report = make_coverage_report(
        [int(value) for value in counts.tolist()],
        manifest_sha256=_manifest_hash(manifest_path),
        cap=cap_value,
    )
    write_coverage_report(output_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan V11 layer-stack head coverage")
    parser.add_argument("manifest", type=Path, nargs="?", help="authenticated JQv4 manifest")
    parser.add_argument("--manifest", dest="manifest_option", type=Path,
                        help="authenticated JQv4 manifest (named form)")
    parser.add_argument("--records", type=_positive_integer,
                        help="bounded number of native records to consume")
    parser.add_argument("--counts",
                        help="16 comma-separated head counts from an audited native scan")
    parser.add_argument("--output", type=Path, required=True, help="coverage report JSON path")
    parser.add_argument("--cap", type=float, default=8.0)
    parser.add_argument("--batch-size", type=_positive_integer, default=2048)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--feature-name", default=DEFAULT_FEATURE_NAME)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = args.manifest_option or args.manifest
    if manifest is None:
        raise SystemExit("a JQv4 manifest is required")
    if args.counts is not None:
        if args.records is not None:
            raise SystemExit("--counts and --records are mutually exclusive")
        _validate_training_manifest(manifest)
        report = make_coverage_report(
            _parse_counts(args.counts),
            manifest_sha256=_manifest_hash(manifest),
            cap=args.cap,
        )
        write_coverage_report(args.output, report)
    elif args.records is not None:
        try:
            scan_head_coverage(
                manifest,
                records=args.records,
                output=args.output,
                cap=args.cap,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                feature_name=args.feature_name,
            )
        except (RuntimeError, ValueError, OSError) as exc:
            raise SystemExit(str(exc)) from exc
        report = json.loads(Path(args.output).read_text(encoding="utf-8"))
    else:
        raise SystemExit("provide either --records or --counts")
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
