import json
from pathlib import Path
import unittest

import architecture_v11
import features_v11 as f
import weight_policy_v11 as policy


class V11ContractTests(unittest.TestCase):
    def test_identity_and_no_psqt(self):
        self.assertEqual(f.REAL_INPUTS, 34800)
        self.assertEqual(f.PS_NB, 1450)
        self.assertEqual(f.NUM_PSQT_BUCKETS, 0)
        self.assertFalse(f.feature_identity_document()['psqt_accumulator'])
        self.assertEqual(len(f.feature_identity_sha256('HalfKAv2_hm_jieqi_v11')), 64)

    def test_architecture_hash_is_stable(self):
        self.assertEqual(len(architecture_v11.ARCHITECTURE_SHA256), 64)
        self.assertEqual(architecture_v11.RUNTIME_ARCHITECTURE['affine_shapes'],
                         [[2064, 32], [64, 32], [128, 1]])
        self.assertEqual(architecture_v11.RUNTIME_ARCHITECTURE['psqt'], False)

    def test_sidecar_closes_nested_inventory_context_contract(self):
        name = 'HalfKAv2_hm_jieqi_v11^'
        expected = policy.expected_architecture(name)
        self.assertEqual(expected['inventory_context'], {
            'inputs': 16,
            'hidden': 16,
            'normalization': 'per-side-inventory-maxima-v1',
        })
        self.assertTrue(policy._architecture_matches(expected, name))
        changed = dict(expected)
        changed['inventory_context'] = dict(expected['inventory_context'], hidden=8)
        self.assertFalse(policy._architecture_matches(changed, name))

    def test_golden_identity_matches(self):
        fixture = Path(__file__).parents[1] / 'tests_v11/fixtures/v11_feature_golden.json'
        payload = json.loads(fixture.read_text())
        self.assertEqual(payload['feature_identity_sha256'],
                         f.feature_identity_sha256('HalfKAv2_hm_jieqi_v11'))


if __name__ == '__main__':
    unittest.main()
