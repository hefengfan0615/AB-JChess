"""Independent NumPy integer oracle, CUDA model, scalar C++, and BMI2 comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import features_v11 as features
from model_v11 import inventory_context_from_sparse
from serialize_v11 import load_source

FEATURES = features.REAL_INPUTS
HEAD_STRIDE = 68544
OUTPUT_RECORD_SIZE = 10264


def trunc_div(n, d):
    return n // d if n >= 0 else -((-n) // d)


def interpolate(a, b, head, q):
    if head == 15 or q == 0:
        return a
    if q == 255:
        return b
    return trunc_div(a * (255 - q) + b * q + 127, 255)


def integer_head(payload, index, x):
    raw = memoryview(payload)[index * HEAD_STRIDE:(index + 1) * HEAD_STRIDE]
    def affine(offset, bias, inputs, width):
        # Read the documented wire order, independent of serializer helpers.
        weights = np.frombuffer(raw, dtype=np.int8, count=width * 32, offset=offset)
        weights = weights.reshape(width // 4, 32, 4).transpose(1, 0, 2).reshape(32, width)
        return weights.astype(np.int64) @ inputs + np.frombuffer(raw, '<i4', 32, bias)
    def activate(value, shift):
        square = np.minimum((np.clip(value, -32768, 32767) ** 2) >> (2 * shift + 7), 127)
        linear = np.clip(value >> shift, 0, 127)
        return np.concatenate((square, linear))
    first = affine(128, 0, x, 2064)
    ac0 = activate(first, 7)
    second = affine(0x10300, 0x10280, ac0, 64)
    ac1 = activate(second, 6)
    last = np.frombuffer(raw, np.int8, 128, 0x10B40).astype(np.int64) @ np.concatenate((ac0, ac1))
    last += int(np.frombuffer(raw, '<i4', 1, 0x10B00)[0]) + int(first[30] - first[31])
    return trunc_div(int(last) * 9600, 16384)


def context_q07(rows, us):
    """Return the runtime inventory context for the side to move."""
    white = torch.tensor([rows[0]], dtype=torch.int64)
    black = torch.tensor([rows[1]], dtype=torch.int64)
    white_to_move = torch.tensor([1.0 if us == 0 else 0.0])
    values = inventory_context_from_sparse(white, black, white_to_move)[0].numpy()
    return np.clip(np.rint(values * 127.0), 0, 127).astype(np.uint8)


def golden_rows():
    path = Path(__file__).resolve().parents[1] / 'tests_v11/fixtures/v11_feature_golden.json'
    symbols = {'R': 0, 'A': 1, 'C': 2, 'P': 3, 'N': 4, 'B': 5, 'K': 6,
               'r': 7, 'a': 8, 'c': 9, 'p': 10, 'n': 11, 'b': 12, 'k': 13, 'X': 14, 'x': 15}
    for case in json.loads(path.read_text())['cases']:
        board = [255] * 90
        for rank, row in enumerate(case['board'].split('/')):
            file = 0
            for ch in row:
                if ch.isdigit():
                    file += int(ch)
                else:
                    board[(9 - rank) * 9 + file] = symbols[ch]
                    file += 1
        position = features.Position(board, case['rest'], case.get('side_to_move', 0))
        rows = [features.active_features(position, c) for c in (0, 1)]
        yield (case['selection']['floor'], case['selection']['blend_q8'], position.side_to_move,
               rows, context_q07(rows, position.side_to_move), 'golden:' + case['name'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--bridge', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    model = load_source(args.checkpoint, features.get_feature_set_from_name('HalfKAv2_hm_jieqi_v11^')).cuda().eval()
    data = args.package.read_bytes()
    length = struct.unpack_from('<I', data, 20)[0]
    metadata = json.loads(data[24:24 + length])
    payload = memoryview(data)[24 + length:]
    primary, head_chunk = metadata['chunks'][:2]
    primary_payload = payload[primary['data_offset']:primary['data_offset'] + primary['size']]
    transformer_biases = np.frombuffer(primary_payload, '<i2', 2048).astype(np.int64)
    weights = np.frombuffer(primary_payload, '<i2', FEATURES * 2048, 4096).reshape(FEATURES, 2048)
    context_offset = 4096 + FEATURES * 2048 * 2
    context_biases = np.frombuffer(primary_payload, '<i4', 16, context_offset).astype(np.int64)
    context_weights = np.frombuffer(primary_payload, np.int8, 256, context_offset + 64).reshape(16, 16).astype(np.int64)
    heads = payload[head_chunk['data_offset']:head_chunk['data_offset'] + head_chunk['size']]
    samples = list(golden_rows())
    rng = np.random.default_rng(20261002)
    for h in range(16):
        for q in (0, 128, 255):
            rows = [rng.integers(0, FEATURES, size=40).tolist() for _ in range(2)]
            us = h % 2
            samples.append((h, q, us, rows, context_q07(rows, us), 'random'))
    import nnue_dataset
    provider = nnue_dataset.SparseBatchProvider('HalfKAv2_hm_jieqi_v11^', str(args.manifest),
                                               64, cyclic=False, device='cpu')
    try:
        batch = next(provider)
    finally:
        provider.close()
    assert len(batch) == 11
    for i in range(64):
        rows = []
        for idx, val in ((batch[2], batch[3]), (batch[4], batch[5])):
            rows.append([int(k) for k, v in zip(idx[i].tolist(), val[i].tolist())
                         if 0 <= k < FEATURES for _ in range(int(v))])
        us = 0 if float(batch[0][i]) else 1
        samples.append((int(batch[8][i]), round(float(batch[9][i]) * 255),
                        us, rows, context_q07(rows, us), 'native-jqv4'))
    args.output.mkdir(parents=True, exist_ok=True)
    input_path = args.output / 'parity-input.bin'
    output_path = args.output / 'parity-output.bin'
    with input_path.open('wb') as stream:
        stream.write(struct.pack('<I', len(samples)))
        for head, q, us, rows, context, _ in samples:
            stream.write(struct.pack('<III', head, q, us))
            for indices in rows:
                stream.write(struct.pack('<I', len(indices)))
                stream.write(np.array(indices, dtype='<u4').tobytes())
            stream.write(context.tobytes())
    run = subprocess.run([str(args.bridge), str(args.package), str(input_path), str(output_path)],
                         capture_output=True, text=True)
    if run.returncode:
        raise RuntimeError(f"native parity bridge failed: {run.stderr.strip() or run.stdout.strip()}")
    native = output_path.read_bytes()
    assert len(native) == len(samples) * OUTPUT_RECORD_SIZE
    details = []
    for i, (head, q, us, rows, context, label) in enumerate(samples):
        offset = i * OUTPUT_RECORD_SIZE
        acc = np.stack([transformer_biases + weights[indices].astype(np.int64).sum(axis=0) for indices in rows])
        np.testing.assert_array_equal(acc, np.frombuffer(native, '<i2', 4096, offset).reshape(2, 2048))
        bounded = np.clip(acc, 0, 127)
        transformed = (bounded[:, :1024] * bounded[:, 1024:]) >> 7
        transformed = np.concatenate((transformed[us], transformed[1 - us]))
        hidden_context = np.clip((context_biases + context_weights @ context.astype(np.int64)) >> 7, 0, 127)
        head_input = np.concatenate((transformed, hidden_context.astype(np.uint8)))
        np.testing.assert_array_equal(head_input, np.frombuffer(native, np.uint8, 2064, offset + 8192))
        expected = interpolate(integer_head(heads, head, head_input),
                               integer_head(heads, min(head + 1, 15), head_input), head, q)
        scalar, simd = struct.unpack_from('<ii', native, offset + 10256)
        assert scalar == simd == expected, (i, expected, scalar, simd)
        factored = [[j for row in indices for j in features.real_feature_factors(row)] for indices in rows]
        tensors = []
        for indices in factored:
            idx = torch.tensor([indices], dtype=torch.int32, device='cuda')
            tensors.extend((idx, torch.ones(idx.shape, device='cuda')))
        with torch.no_grad():
            pred = model(torch.tensor([[1 - us]], device='cuda'), torch.tensor([[us]], device='cuda'),
                         *tensors, torch.tensor([head], device='cuda'), torch.tensor([q / 255], device='cuda'))
        error_cp = abs(float(pred) * 600 - expected / 16)
        details.append({'sample': i, 'kind': label, 'head': head, 'q': q, 'raw': expected,
                        'inventory_context_q07': context.tolist(),
                        'float_cp': float(pred) * 600, 'error_cp': error_cp})
    report = {'samples': len(samples), 'integer_mismatches': 0,
              'cpp_backend': run.stdout.strip(),
              # Export uses nearest integer FT/head quantization.  The fixed
              # point runtime is exact; the float trainer may differ by a few
              # centipawns at activation boundaries.
              'float_tolerance_cp': 32, 'max_float_error_cp': max(x['error_cp'] for x in details),
              'mean_float_error_cp': float(np.mean([x['error_cp'] for x in details])),
              'package_sha256': hashlib.sha256(data).hexdigest(), 'details': details}
    (args.output / 'parity-report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'details'}))
    assert report['max_float_error_cp'] <= report['float_tolerance_cp'], 'floating/quantized error exceeds 32 cp'


if __name__ == '__main__':
    main()
