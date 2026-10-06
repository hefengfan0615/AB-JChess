"""Closed runtime architecture and quantization contract (canonical JSON SHA-256)."""
import hashlib
import json

RUNTIME_ARCHITECTURE = {
    'schema': 'abjchess-v11-sfnn-inventory-context-v1',
    'real_features': 34800,
    'ft': 2048,
    'heads': 16,
    'context_input': 16,
    'context_hidden': 16,
    'context_shape': [16, 16],
    'context_input_order': [
        'us_pool_rook', 'us_pool_advisor', 'us_pool_cannon',
        'us_pool_pawn', 'us_pool_knight', 'us_pool_bishop',
        'them_pool_rook', 'them_pool_advisor', 'them_pool_cannon',
        'them_pool_pawn', 'them_pool_knight', 'them_pool_bishop',
        'us_dark_count', 'them_dark_count',
        'us_unknown_loss', 'them_unknown_loss',
    ],
    'context_pool_maxima': [2, 2, 2, 5, 2, 2],
    'context_input_quantization': 'unsigned-q0.7-round-half-up',
    'context_activation': 'clipped-q0.7-appended-after-transformer',
    'affine_shapes': [[2064, 32], [64, 32], [128, 1]],
    'activation_order': ['squared', 'clipped'],
    'activation_shifts': [7, 6],
    'squared_shifts': [21, 19],
    'skip': [30, 31],
    'ft_scale': 127,
    'hidden_one': 128,
    'weight_scales': [128, 64, 128],
    'bias_scales': [16384, 8192, 16384],
    'output_multiplier': 9600,
    'output_denominator': 16384,
    'output_scale': 16,
    'interpolation': 'continuous-q0.8-v1',
    'psqt': False,
    'head_stride': 68544,
    'offsets': [0, 128, 66176, 66304, 68352, 68416],
}
ARCHITECTURE_SHA256 = hashlib.sha256(json.dumps(RUNTIME_ARCHITECTURE, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()
