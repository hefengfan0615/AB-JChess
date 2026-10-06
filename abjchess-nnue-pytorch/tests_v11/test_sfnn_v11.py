"""Behavioral SFNN checks with optional checkpoint integration."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
import features_v11 as features
from model_v11 import HEAD_INPUTS, LayerStacks, NNUE, inventory_context_from_sparse
import serialize_v11 as serializer
import weight_policy_v11 as policy


class HeadTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        self.heads = LayerStacks(16, 1024, 32, 32)
        self.heads.reset_independent_heads(42)

    def test_phase_endpoints_and_head_independence(self):
        x = torch.rand(16, HEAD_INPUTS)
        indices = torch.arange(16)
        values, _ = self.heads._all_heads(x)
        for q in (0, 1 / 255, 128 / 255, 1):
            alpha = torch.full((16,), q)
            expected = values[indices, indices] * (1 - alpha) + values[indices, (indices + 1).clamp_max(15)] * alpha
            torch.testing.assert_close(self.heads(x, indices, alpha)[:, 0], expected)
        result = self.heads(x[:1], torch.tensor([5]), torch.tensor([0.0]))
        result.sum().backward()
        self.assertGreater(self.heads.l1.weight.grad[160:192].abs().sum().item(), 0)
        self.assertEqual(self.heads.l1.weight.grad[:160].abs().sum().item(), 0)
        self.assertEqual(self.heads.l1.weight.grad[192:].abs().sum().item(), 0)
        self.assertFalse(torch.equal(self.heads.l1.weight[:32], self.heads.l1.weight[32:64]))

    def test_skip_is_signed_and_bypasses_clipping(self):
        with torch.no_grad():
            for parameter in self.heads.parameters():
                parameter.zero_()
            self.heads.l1.bias[30] = -2.0
            self.heads.l1.bias[31] = 3.0
        self.assertEqual(self.heads(torch.zeros(1, HEAD_INPUTS), torch.tensor([0])).item(), -5.0)

    def test_second_squared_activation_accepts_negative_values(self):
        with torch.no_grad():
            for parameter in self.heads.parameters():
                parameter.zero_()
            self.heads.l2.bias[0] = -0.5
            self.heads.output.weight[0, 64] = 1.0
        self.assertEqual(self.heads(torch.zeros(1, HEAD_INPUTS), torch.tensor([0])).item(), 0.25)

    def test_invalid_phase_and_shape_rejected(self):
        with self.assertRaises(ValueError):
            self.heads(torch.zeros(1, 1024), torch.tensor([0]))
        with self.assertRaises(ValueError):
            self.heads(torch.zeros(1, HEAD_INPUTS), torch.tensor([16]))
        with self.assertRaises(ValueError):
            self.heads(torch.zeros(1, HEAD_INPUTS), torch.tensor([0]), torch.tensor([float('nan')]))

    def test_batch_has_no_psqt(self):
        values = tuple(range(10))
        self.assertEqual(NNUE._unpack_batch(values), values + (None,))
        with self.assertRaises(ValueError):
            NNUE._unpack_batch(tuple(range(12)))

    def test_inventory_context_order_and_q07_match_public_sparse_features(self):
        board = [features.EMPTY] * features.NUM_SQ
        board[features.square(4, 0)] = features.W_KING
        board[features.square(4, 9)] = features.B_KING
        board[features.square(0, 2)] = features.DARK_WHITE
        board[features.square(8, 7)] = features.DARK_BLACK
        position = features.Position(
            board=board,
            rest=((1, 0, 1, 2, 0, 0), (0, 1, 0, 3, 0, 1)),
            unknown_loss=(3, 4),
        )
        white = torch.tensor([features.active_features(position, 0)])
        black = torch.tensor([features.active_features(position, 1)])
        context = inventory_context_from_sparse(white, black, torch.tensor([1.0]))
        expected = torch.tensor([1 / 2, 0, 1 / 2, 2 / 5, 0, 0,
                                 0, 1 / 2, 0, 3 / 5, 0, 1 / 2,
                                 1 / 16, 1 / 16, 3 / 15, 4 / 15])
        expected = torch.floor(expected * 127 + 0.5) / 127
        torch.testing.assert_close(context[0], expected)


class ProvenanceTests(unittest.TestCase):
    def test_old_checkpoints_rejected_before_deserialization(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'old.ckpt'
            p.write_bytes(b'not a pickle')
            for version in ('v8', 'v9', 'v10'):
                policy.sidecar_path(p).write_text(json.dumps({'schema': f'abjchess-{version}-training-weight-v1'}))
                with patch.object(serializer.model_v11, 'load_v11_checkpoint') as loader:
                    with self.assertRaises(serializer.SerializeV11Error):
                        serializer.load_source(p, features.get_feature_set_from_name('HalfKAv2_hm_jieqi_v11^'))
                    loader.assert_not_called()

    def test_sidecar_roundtrip_and_hash_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'test.pt'
            p.write_bytes(b'contract validation fixture')
            name = 'HalfKAv2_hm_jieqi_v11^'
            policy.write_v11_sidecar(p, feature_name=name, feature_identity_sha256=features.feature_identity_sha256(name),
                                    architecture=policy.expected_architecture(name), teacher_dataset_sha256='a' * 64)
            self.assertEqual(policy.validate_v11_weight(p).feature_name, name)
            p.write_bytes(b'tampered')
            with self.assertRaises(policy.WeightPolicyError):
                policy.validate_v11_weight(p)

    @unittest.skipUnless(os.environ.get('V11_RUN_ROOT'), 'requires completed smoke run')
    def test_smoke_checkpoint_export_roundtrip(self):
        root = Path(os.environ['V11_RUN_ROOT'])
        checkpoint = root / 'smoke/lightning_logs/version_0/checkpoints/last.ckpt'
        model = serializer.load_source(checkpoint, features.get_feature_set_from_name('HalfKAv2_hm_jieqi_v11^'))
        self.assertEqual(tuple(model.input.weight.shape), (features.TOTAL_INPUTS, 2048))
        self.assertFalse(any('psqt' in key.lower() for key in model.state_dict()))
        from train_v11 import checkpoint_architecture
        self.assertEqual(checkpoint_architecture(model), policy.expected_architecture(model.feature_set.name))
        regenerated = serializer.build_package_bytes(model, smoke_only=True,
            description='AB-JChess V11 GPU smoke; 8 updates; authenticated formal probability tables',
            probability_score_to_mass=root / 'smoke/probability_score_to_mass.i32le',
            probability_mass_to_score=root / 'smoke/probability_mass_to_score.i32le')
        self.assertEqual(regenerated, (root / 'smoke/abjchess-v11-smoke.nnue').read_bytes())


if __name__ == '__main__':
    unittest.main()
