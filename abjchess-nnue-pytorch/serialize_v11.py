"""Export a four-chunk ABJNNUE container consumed by the V11 engine.

Virtual rows are coalesced once at export time, so the runtime stores exactly
31,776 feature rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
from typing import Mapping

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - doctor can run without Torch
    torch = None

from architecture_v11 import ARCHITECTURE_SHA256
import features_v11
import model_v11
from weight_policy_v11 import WeightPolicyError, validate_before_torch_load


PACKAGE_VERSION = 110
PACKAGE_SCHEMA = "abjchess-v11-sfnn-inventory-context-v1"
# Native V11 package magic identifies this package format.
MAGIC = b"ABJCHESSV11" + b"\0" * 5
HEADER_SIZE = 24

ACCUMULATOR_WIDTH = 2048
PSQT_BUCKETS = 0
REAL_FEATURES = features_v11.REAL_INPUTS
TRANSFORMER_BIAS_SIZE = ACCUMULATOR_WIDTH * 2
FEATURE_WEIGHTS_SIZE = REAL_FEATURES * ACCUMULATOR_WIDTH * 2
CONTEXT_BIAS_SIZE = model_v11.CONTEXT_HIDDEN * 4
CONTEXT_WEIGHT_SIZE = model_v11.CONTEXT_INPUTS * model_v11.CONTEXT_HIDDEN
TRANSFORMER_PAYLOAD_SIZE = (
    TRANSFORMER_BIAS_SIZE + FEATURE_WEIGHTS_SIZE + CONTEXT_BIAS_SIZE + CONTEXT_WEIGHT_SIZE
)
PRIMARY_SIZE = TRANSFORMER_PAYLOAD_SIZE
EVAL_HEAD_BUCKET_SIZE = 68_544
LAYER_STACKS = 16
EVAL_HEADS_SIZE = EVAL_HEAD_BUCKET_SIZE * LAYER_STACKS
PROBABILITY_SCORE_TO_MASS_SIZE = 4001 * 4
PROBABILITY_MASS_TO_SCORE_SIZE = 1901 * 4

L1_BIAS_OFFSET = 0x0000
L1_WEIGHT_OFFSET = 0x0080
L2_BIAS_OFFSET = 0x10280
L2_WEIGHT_OFFSET = 0x10300
FINAL_BIAS_OFFSET = 0x10B00
FINAL_WEIGHT_OFFSET = 0x10B40

CHUNK_NAMES = (
    "primary_runtime_nnue_container.bin",
    "eval_heads_runtime.bin",
    "probability_score_to_mass.i32le",
    "probability_mass_to_score.i32le",
)


class SerializeV11Error(ValueError):
    pass


def validate_source_suffix(source: str | Path) -> None:
    suffix = Path(source).suffix.lower()
    if suffix == ".nnue":
        raise SerializeV11Error("V11 runtime packages cannot be used as PyTorch training checkpoints")
    if suffix not in {".pt", ".ckpt"}:
        raise SerializeV11Error("V11 exporter accepts only .pt or .ckpt sources")


def _require_torch() -> None:
    if torch is None:
        raise SerializeV11Error("Torch is required to export V11 weights")


def _tensor_numpy(value, dtype) -> np.ndarray:
    _require_torch()
    if not isinstance(value, torch.Tensor):
        raise SerializeV11Error("model tensor is not a Torch tensor")
    return value.detach().to(device="cpu", dtype=torch.float32).numpy().astype(dtype, copy=False)


def _quant_i16(value, scale: float) -> np.ndarray:
    raw = np.rint(np.asarray(value, dtype=np.float64) * scale)
    if np.any(raw < -32768) or np.any(raw > 32767):
        raise SerializeV11Error("int16 quantization overflow")
    return raw.astype("<i2")


def _quant_i32(value, scale: float) -> np.ndarray:
    raw = np.rint(np.asarray(value, dtype=np.float64) * scale)
    if np.any(raw < -(1 << 31)) or np.any(raw > (1 << 31) - 1):
        raise SerializeV11Error("int32 quantization overflow")
    return raw.astype("<i4")


def _quant_fc(layer, *, first: bool = False, output: bool = False) -> tuple[np.ndarray, np.ndarray]:
    scale_bias = 16384.0 if (first or output) else 8192.0
    scale_weight = scale_bias / 128.0
    max_weight = 127.0 / scale_weight
    bias = _quant_i32(_tensor_numpy(layer.bias, np.float32), scale_bias)
    weights = np.asarray(_tensor_numpy(layer.weight, np.float32), dtype=np.float64)
    weights = np.clip(weights, -max_weight, max_weight)
    quantized = np.rint(weights * scale_weight).astype(np.int8)
    return bias, quantized


def _blocked_weights(weights: np.ndarray, input_width: int) -> bytes:
    """Pack [outputs, inputs] as [input groups][outputs][4 lanes]."""

    outputs, inputs = weights.shape
    if inputs > input_width or input_width % 4:
        raise SerializeV11Error("invalid native affine input width")
    padded = np.zeros((outputs, input_width), dtype=np.int8)
    padded[:, :inputs] = weights
    return padded.reshape(outputs, input_width // 4, 4).transpose(1, 0, 2).tobytes()


def pack_eval_head(l1, l2, output) -> bytes:
    """Quantize one native 2064 -> 32 -> 64 -> 32 -> 128 -> 1 head."""

    _require_torch()
    if (l1.in_features, l1.out_features) != (2064, 32):
        raise SerializeV11Error("V11 L1 head must be Linear(2064, 32)")
    if (l2.in_features, l2.out_features) != (64, 32):
        raise SerializeV11Error("V11 L2 head must be Linear(64, 32)")
    if (output.in_features, output.out_features) != (128, 1):
        raise SerializeV11Error("V11 output head must be Linear(128, 1)")
    l1_bias, l1_weights = _quant_fc(l1, first=True)
    l2_bias, l2_weights = _quant_fc(l2)
    final_bias, final_weights = _quant_fc(output, output=True)
    payload = bytearray(EVAL_HEAD_BUCKET_SIZE)
    payload[L1_BIAS_OFFSET:L1_BIAS_OFFSET + l1_bias.nbytes] = l1_bias.tobytes()
    blocked_l1 = _blocked_weights(l1_weights, 2064)
    payload[L1_WEIGHT_OFFSET:L1_WEIGHT_OFFSET + len(blocked_l1)] = blocked_l1
    payload[L2_BIAS_OFFSET:L2_BIAS_OFFSET + l2_bias.nbytes] = l2_bias.tobytes()
    blocked_l2 = _blocked_weights(l2_weights, 64)
    payload[L2_WEIGHT_OFFSET:L2_WEIGHT_OFFSET + len(blocked_l2)] = blocked_l2
    payload[FINAL_BIAS_OFFSET:FINAL_BIAS_OFFSET + final_bias.nbytes] = final_bias.tobytes()
    payload[FINAL_WEIGHT_OFFSET:FINAL_WEIGHT_OFFSET + final_weights.nbytes] = final_weights.reshape(-1).tobytes()
    return bytes(payload)


def pack_primary(model: model_v11.NNUE) -> bytes:
    _require_torch()
    feature_set = model.feature_set
    if feature_set.num_real_features != REAL_FEATURES:
        raise SerializeV11Error(f"V11 runtime expects {REAL_FEATURES} real features, got {feature_set.num_real_features}")
    if model.ft_dim != ACCUMULATOR_WIDTH:
        raise SerializeV11Error("V11 transformer dimensions are not 2048")
    bias = _quant_i16(_tensor_numpy(model.input.bias[:ACCUMULATOR_WIDTH], np.float32), 127.0)
    coalesced = model_v11.coalesce_ft_weights(model, model.input)
    if tuple(coalesced.shape) != (REAL_FEATURES, ACCUMULATOR_WIDTH):
        raise SerializeV11Error(f"coalesced transformer shape is {tuple(coalesced.shape)}")
    ft = _quant_i16(_tensor_numpy(coalesced[:, :ACCUMULATOR_WIDTH], np.float32), 127.0)
    context_bias, context_weights = _quant_fc(model.context_tower, first=True)
    payload = bytearray(PRIMARY_SIZE)
    cursor = 0
    payload[cursor:cursor + bias.nbytes] = bias.tobytes(); cursor += bias.nbytes
    payload[cursor:cursor + ft.nbytes] = ft.tobytes(); cursor += ft.nbytes
    payload[cursor:cursor + context_bias.nbytes] = context_bias.tobytes(); cursor += context_bias.nbytes
    payload[cursor:cursor + context_weights.nbytes] = context_weights.reshape(-1).tobytes(); cursor += context_weights.nbytes
    if cursor != PRIMARY_SIZE:
        raise AssertionError("V11 primary payload size calculation is inconsistent")
    return bytes(payload)


def pack_eval_heads(model: model_v11.NNUE) -> bytes:
    payload = bytearray()
    for l1, l2, output in model.layer_stacks.get_coalesced_layer_stacks():
        payload.extend(pack_eval_head(l1, l2, output))
    if len(payload) != EVAL_HEADS_SIZE:
        raise SerializeV11Error("V11 eval-head payload has the wrong size")
    return bytes(payload)


def _optional_payload(value: bytes | bytearray | str | Path | None, size: int, name: str) -> bytes:
    if value is None:
        return bytes(size)
    if isinstance(value, (str, Path)):
        try:
            value = Path(value).read_bytes()
        except OSError as exc:
            raise SerializeV11Error(f"cannot read {name}: {exc}") from exc
    value = bytes(value)
    if len(value) != size:
        raise SerializeV11Error(f"{name} must contain exactly {size} bytes")
    return value


def _runtime_table_payloads(*, probability_score_to_mass=None, probability_mass_to_score=None,
                            smoke_only: bool = False) -> dict[str, bytes]:
    """Resolve probability chunks without silently manufacturing tables.

    A normal runtime package needs both probability lookup tables because the
    engine uses them for hidden-position aggregation.
    """

    values = {
        CHUNK_NAMES[2]: probability_score_to_mass,
        CHUNK_NAMES[3]: probability_mass_to_score,
    }
    sizes = {
        CHUNK_NAMES[2]: PROBABILITY_SCORE_TO_MASS_SIZE,
        CHUNK_NAMES[3]: PROBABILITY_MASS_TO_SCORE_SIZE,
    }
    probability_names = (CHUNK_NAMES[2], CHUNK_NAMES[3])
    missing = [name for name in probability_names if values[name] is None]
    if missing:
        if not smoke_only:
            names = ", ".join(missing)
            raise SerializeV11Error(
                "V11 export requires runtime tables (missing: " + names
                + "); provide both probability table paths, or use "
                  "--smoke-only for a structural zero-table package"
            )
        if len(missing) != len(probability_names):
            raise SerializeV11Error(
                "--smoke-only cannot mix supplied and missing probability tables"
            )
        for name in probability_names:
            values[name] = bytes(sizes[name])

    return {
        name: _optional_payload(values[name], sizes[name], name)
        for name in values
    }


def build_chunks(model: model_v11.NNUE, *, probability_score_to_mass=None,
                 probability_mass_to_score=None,
                 smoke_only: bool = False) -> dict[str, bytes]:
    tables = _runtime_table_payloads(
        probability_score_to_mass=probability_score_to_mass,
        probability_mass_to_score=probability_mass_to_score,
        smoke_only=smoke_only,
    )
    return {
        CHUNK_NAMES[0]: pack_primary(model),
        CHUNK_NAMES[1]: pack_eval_heads(model),
        CHUNK_NAMES[2]: tables[CHUNK_NAMES[2]],
        CHUNK_NAMES[3]: tables[CHUNK_NAMES[3]],
    }


def _chunk_metadata(chunks: Mapping[str, bytes]) -> list[dict[str, object]]:
    offset = 0
    output = []
    for name in CHUNK_NAMES:
        payload = chunks[name]
        output.append({
            "name": name,
            "role": "runtime",
            "dtype": "u8" if name.endswith(".bin") else "i32le",
            "data_offset": offset,
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        })
        offset += len(payload)
    return output


def build_package_bytes(model: model_v11.NNUE, *, description: str | None = None,
                        probability_score_to_mass=None, probability_mass_to_score=None,
                        smoke_only: bool = False) -> bytes:
    chunks = build_chunks(model, probability_score_to_mass=probability_score_to_mass,
                          probability_mass_to_score=probability_mass_to_score,
                          smoke_only=smoke_only)
    feature_name = model.feature_set.name
    metadata = {
        "architecture_sha256": ARCHITECTURE_SHA256,
        "format_version": PACKAGE_VERSION,
        "schema": PACKAGE_SCHEMA,
        # Runtime packages have one closed feature ABI.  The training side
        # may use either the plain or factorized name, but both are coalesced
        # to this exact native identity before export.
        "feature_identity": "HalfKAv2_hm_jieqi_v11_sfnn_pool_owner_exact_loss_visible_threat_summary_no_relation_threat_input",
        "feature_identity_sha256": features_v11.feature_identity_sha256("HalfKAv2_hm_jieqi_v11"),
        "feature_dimensions": REAL_FEATURES,
        "accumulator_width": ACCUMULATOR_WIDTH,
        "inventory_context_inputs": model_v11.CONTEXT_INPUTS,
        "inventory_context_hidden": model_v11.CONTEXT_HIDDEN,
        "psqt_buckets": PSQT_BUCKETS,
        "layer_stacks": LAYER_STACKS,
        "interpolation": {"format": "q0.8", "rounding": "+127/255", "version": "v1"},
        "interpolation_format": "Q0.8",
        "interpolation_formula": "continuous-q0.8-v1",
        "description": description or f"AB-JChess V11 NNUE ({feature_name})",
        "smoke_only": bool(smoke_only),
        "architecture": model.architecture_contract(),
        "payload_bytes": sum(len(chunks[name]) for name in CHUNK_NAMES),
        "chunks": _chunk_metadata(chunks),
    }
    metadata_bytes = json.dumps(metadata, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload = b"".join(chunks[name] for name in CHUNK_NAMES)
    return MAGIC + struct.pack("<II", PACKAGE_VERSION, len(metadata_bytes)) + metadata_bytes + payload


def export_model(model: model_v11.NNUE, target: str | Path, **kwargs) -> Path:
    target = Path(target).resolve()
    if target.suffix.lower() != ".nnue":
        raise SerializeV11Error("V11 runtime output must use the .nnue extension")
    data = build_package_bytes(model, **kwargs)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(target)
    return target


def load_source(source: str | Path, feature_set: features_v11.FeatureSetV11) -> model_v11.NNUE:
    validate_source_suffix(source)
    try:
        provenance = validate_before_torch_load(source, expected_feature_name=feature_set.name)
    except WeightPolicyError as exc:
        raise SerializeV11Error(f"V11 weight policy rejected source: {exc}") from exc
    return model_v11.load_v11_checkpoint(str(source), feature_set)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Export authenticated PyTorch V11 weights to an ABJNNUE V11 package")
    parser.add_argument("source", help="V11 .pt or .ckpt checkpoint with .abjv11.json sidecar")
    parser.add_argument("target", help="V11 runtime package (.nnue)")
    features_v11.add_argparse_args(parser)
    parser.add_argument("--description", default=None)
    parser.add_argument("--probability-score-to-mass", default=None)
    parser.add_argument("--probability-mass-to-score", default=None)
    parser.add_argument(
        "--smoke-only", action="store_true",
        help="mark as smoke-only; permits zero tables when neither table is supplied",
    )
    args = parser.parse_args(argv)
    feature_set = features_v11.get_feature_set_from_name(args.features)
    model = load_source(args.source, feature_set)
    model.eval()
    export_model(model, args.target, description=args.description,
                 probability_score_to_mass=args.probability_score_to_mass,
                 probability_mass_to_score=args.probability_mass_to_score,
                 smoke_only=args.smoke_only)
    print(f"wrote V11 package {Path(args.target).resolve()}")


if __name__ == "__main__":
    main()
