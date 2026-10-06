import json
from pathlib import Path
import unittest


class V11FeatureContractTests(unittest.TestCase):
    def test_real_and_virtual_dimensions_are_frozen(self):
        import features_v11 as f

        self.assertEqual(f.BASE_PS_NB, 15 * 90)
        self.assertEqual(f.META_NB, 100)
        self.assertEqual(f.PS_NB, 1450)
        self.assertEqual(f.NUM_BUCKETS, 24)
        self.assertEqual(f.REAL_INPUTS, 34800)
        self.assertEqual(f.VIRTUAL_INPUTS, 1474)
        self.assertEqual(f.TOTAL_INPUTS, 36274)

    def test_factor_mapping_has_only_two_virtual_factors(self):
        import features_v11 as f

        factors = f.real_feature_factors(5 * f.PS_NB + 17)
        self.assertEqual(factors, [5 * f.PS_NB + 17, f.REAL_INPUTS + 17, f.REAL_INPUTS + f.PS_NB + 5])

    def test_feature_names_and_architecture(self):
        import features_v11 as f

        plain = f.get_feature_set_from_name("HalfKAv2_hm_jieqi_v11")
        factored = f.get_feature_set_from_name("HalfKAv2_hm_jieqi_v11^")
        self.assertEqual(plain.num_real_features, f.REAL_INPUTS)
        self.assertEqual(plain.num_features, f.REAL_INPUTS)
        self.assertEqual(factored.num_features, f.TOTAL_INPUTS)
        self.assertEqual(plain.num_psqt_buckets, 0)
        self.assertEqual(plain.num_ls_buckets, 16)
        self.assertEqual((plain.ft_dim, plain.l1, plain.l2, plain.l3), (2048, 1024, 32, 32))

    def test_piece_square_plane_order(self):
        import features_v11 as f

        # This is the order used by training_data_loader.cpp and the V11 C++
        # encoder, rather than the numeric Jieqi piece-type order.
        self.assertEqual(
            [f.PS_W_ROOK, f.PS_W_CANNON, f.PS_W_KNIGHT, f.PS_W_PAWN,
             f.PS_W_ADVISOR, f.PS_W_BISHOP, f.PS_WB_KING,
             f.PS_DARK_US, f.PS_DARK_THEM],
            [0, 2 * f.NUM_SQ, 4 * f.NUM_SQ, 6 * f.NUM_SQ,
             8 * f.NUM_SQ, 10 * f.NUM_SQ, 12 * f.NUM_SQ,
             13 * f.NUM_SQ, 14 * f.NUM_SQ],
        )
    def test_mid_mirror_and_attack_bucket_are_deterministic(self):
        import features_v11 as f

        board = [f.EMPTY] * f.NUM_SQ
        board[f.square(4, 0)] = f.W_KING
        board[f.square(4, 9)] = f.B_KING
        board[f.square(0, 0)] = f.W_ROOK
        board[f.square(1, 0)] = f.W_CANNON
        board[f.square(8, 9)] = f.B_KNIGHT
        position = f.Position(board=board, rest=((0,) * 6, (0,) * 6), side_to_move=0)
        self.assertEqual(f.attack_bucket(position, 0), 3)
        self.assertEqual(f.king_bucket(4, 85, False), (1, False))
        self.assertIsInstance(f.requires_mid_mirror(position, 0), bool)

    def test_dark_planes_encode_owner_and_reject_uncolored_markers(self):
        import features_v11 as f

        board = [f.EMPTY] * f.NUM_SQ
        board[f.square(4, 0)] = f.W_KING
        board[f.square(4, 9)] = f.B_KING
        board[f.square(0, 1)] = f.DARK_WHITE
        board[f.square(8, 8)] = f.DARK_BLACK
        position = f.Position(
            board=board, rest=((1, 0, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0)))
        white_rows = f.active_features(position, 0)
        self.assertIn(f.PS_DARK_US + f.square(0, 1),
                      [row % f.PS_NB for row in white_rows])
        self.assertIn(f.PS_DARK_THEM + f.square(8, 8),
                      [row % f.PS_NB for row in white_rows])
        with self.assertRaisesRegex(ValueError, "owner-colored"):
            f.Position(board=[16] + [f.EMPTY] * 89, rest=((0,) * 6, (0,) * 6))

    def test_unknown_loss_uses_exact_metadata_rows(self):
        import features_v11 as f

        board = [f.EMPTY] * f.NUM_SQ
        board[f.square(4, 0)] = f.W_KING
        board[f.square(4, 9)] = f.B_KING
        board[f.square(0, 3)] = f.DARK_WHITE
        position = f.Position(board=board, rest=((0, 0, 0, 3, 0, 0), (0,) * 6))
        self.assertEqual(f.meta_feature_offsets(position, 0)[5:7], (42, 56))
        rows = [row % f.PS_NB for row in f.active_features(position, 0)]
        self.assertIn(f.BASE_PS_NB + 42, rows)

    def test_wire_king_codes_win_over_normalized_code_collisions(self):
        import features_v11 as f

        board = [f.EMPTY] * f.NUM_SQ
        board[f.square(4, 0)] = f.W_KING
        board[f.square(0, 6)] = f.DARK_BLACK
        board[f.square(4, 9)] = f.B_KING
        rest = ((0,) * 6, (1, 0, 0, 0, 0, 0))
        self.assertEqual(f._king_square(f.Position(board, rest), 0), f.square(4, 0))
        self.assertEqual(f._king_square(f.Position(board, rest), 1), f.square(4, 9))


class V11FeatureSemanticsTests(unittest.TestCase):
    @staticmethod
    def _kings():
        import features_v11 as f

        board = [f.EMPTY] * f.NUM_SQ
        board[f.square(4, 0)] = f.W_KING
        board[f.square(4, 9)] = f.B_KING
        return board

    def test_unknown_loss_exact_rows_for_both_perspectives(self):
        import features_v11 as f

        board = self._kings()
        # The pawn inventory is used so every test value remains a legal
        # compact rest count while exercising the 3+ bucket.
        pos = f.Position(
            board=board,
            rest=((0, 0, 0, 0, 0, 0), (0, 0, 0, 4, 0, 0)),
            unknown_loss=(0, 4),
        )
        self.assertEqual(f.meta_feature_offsets(pos, 0)[5:7], (40, 60))
        self.assertEqual(f.meta_feature_offsets(pos, 1)[5:7], (44, 56))

    def test_unknown_loss_requires_two_nonnegative_integer_sides(self):
        import features_v11 as f

        board = self._kings()
        rest = ((0,) * 6, (0,) * 6)
        for malformed in ((0,), (0, 1, 2), (-1, 0), (0, -1), (0.5, 0), ("1", 0)):
            with self.assertRaises(ValueError):
                f.Position(board=board, rest=rest, unknown_loss=malformed)

    def test_unknown_loss_inventory_must_match_colored_dark_squares(self):
        import features_v11 as f

        board = self._kings()
        board[f.square(0, 3)] = f.DARK_WHITE
        with self.assertRaisesRegex(ValueError, "unknown_loss"):
            f.Position(
                board=board,
                rest=((0, 0, 0, 2, 0, 0), (0,) * 6),
                unknown_loss=(0, 0),
            )
        valid = f.Position(
            board=board,
            rest=((0, 0, 0, 2, 0, 0), (0,) * 6),
            unknown_loss=(1, 0),
        )
        self.assertEqual(f.meta_feature_offsets(valid, 0)[5], 41)

    def test_threat_summary_counts_distinct_visible_targets(self):
        import features_v11 as f

        board = self._kings()
        board[f.square(0, 0)] = f.W_ROOK
        board[f.square(1, 3)] = f.W_ROOK
        board[f.square(0, 3)] = f.B_ROOK
        pos = f.Position(board=board, rest=((0,) * 6, (0,) * 6))
        self.assertEqual(f.visible_threat_summary(pos, 0), (1, 2))

    def test_rook_and_cannon_lines_respect_dark_blockers_and_screens(self):
        import features_v11 as f

        board = self._kings()
        board[f.square(0, 0)] = f.W_ROOK
        board[f.square(0, 3)] = f.B_ROOK
        board[f.square(0, 2)] = f.DARK_WHITE
        blocked = f.Position(board=board, rest=((1, 0, 0, 0, 0, 0), (0,) * 6))
        self.assertEqual(f.visible_threat_summary(blocked, 0), (0, 0))

        board = self._kings()
        board[f.square(0, 0)] = f.W_CANNON
        board[f.square(0, 1)] = f.DARK_WHITE
        board[f.square(0, 2)] = f.B_ROOK
        screened = f.Position(board=board, rest=((1, 0, 0, 0, 0, 0), (0,) * 6))
        self.assertEqual(f.visible_threat_summary(screened, 0), (1, 0))

        board[f.square(0, 2)] = f.DARK_BLACK
        board[f.square(0, 3)] = f.DARK_BLACK
        board[f.square(0, 4)] = f.B_ROOK
        double_screened = f.Position(
            board=board, rest=((1, 0, 0, 0, 0, 0), (2, 0, 0, 0, 0, 0)))
        self.assertEqual(f.visible_threat_summary(double_screened, 0), (0, 0))

    def test_knight_leg_and_bishop_eye_blockers_are_occupied(self):
        import features_v11 as f

        board = self._kings()
        board[f.square(4, 4)] = f.W_KNIGHT
        board[f.square(6, 5)] = f.B_ROOK
        board[f.square(5, 4)] = f.DARK_BLACK
        blocked_knight = f.Position(board=board, rest=((0,) * 6, (1, 0, 0, 0, 0, 0)))
        self.assertEqual(f.visible_threat_summary(blocked_knight, 0), (0, 0))
        board[f.square(5, 4)] = f.EMPTY
        open_knight = f.Position(board=board, rest=((0,) * 6, (0,) * 6))
        self.assertEqual(f.visible_threat_summary(open_knight, 0), (1, 0))

        board = self._kings()
        board[f.square(2, 2)] = f.W_BISHOP
        board[f.square(4, 4)] = f.B_BISHOP
        board[f.square(3, 3)] = f.DARK_WHITE
        blocked_bishop = f.Position(board=board, rest=((1, 0, 0, 0, 0, 0), (0,) * 6))
        self.assertEqual(f.visible_threat_summary(blocked_bishop, 0), (0, 0))
        board[f.square(3, 3)] = f.EMPTY
        open_bishop = f.Position(board=board, rest=((0,) * 6, (0,) * 6))
        self.assertEqual(f.visible_threat_summary(open_bishop, 0), (1, 1))

    def test_pawn_river_direction_controls_forward_and_side_attacks(self):
        import features_v11 as f

        board = self._kings()
        board[f.square(4, 4)] = f.W_PAWN
        board[f.square(4, 5)] = f.B_BISHOP
        board[f.square(3, 4)] = f.B_CANNON
        before_river = f.Position(board=board, rest=((0,) * 6, (0,) * 6))
        self.assertEqual(f.visible_threat_summary(before_river, 0), (1, 0))

        board = self._kings()
        board[f.square(4, 5)] = f.W_PAWN
        board[f.square(4, 6)] = f.B_BISHOP
        board[f.square(3, 5)] = f.B_CANNON
        after_river = f.Position(board=board, rest=((0,) * 6, (0,) * 6))
        self.assertEqual(f.visible_threat_summary(after_river, 0), (2, 0))

    def test_dark_attackers_and_dark_targets_are_not_counted(self):
        import features_v11 as f

        board = self._kings()
        board[f.square(0, 0)] = f.DARK_WHITE
        board[f.square(0, 3)] = f.B_ROOK
        board[f.square(1, 0)] = f.W_ROOK
        board[f.square(1, 3)] = f.DARK_BLACK
        pos = f.Position(board=board, rest=((1, 0, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0)))
        self.assertEqual(f.visible_threat_summary(pos, 0), (0, 0))

    def test_metadata_uses_unknown_loss_and_threat_groups(self):
        import features_v11 as f

        board = self._kings()
        board[f.square(0, 0)] = f.W_ROOK
        board[f.square(0, 3)] = f.B_ROOK
        pos = f.Position(
            board=board,
            rest=((0, 0, 0, 2, 0, 0), (0, 0, 0, 3, 0, 0)),
            unknown_loss=(2, 3),
        )
        self.assertEqual(
            f.meta_feature_offsets(pos, 0),
            (0, 8, 18, 27, 32, 42, 59, 73, 81),
        )
        self.assertEqual(
            f.meta_feature_offsets(pos, 1),
            (0, 8, 19, 26, 32, 43, 58, 73, 81),
        )

    def test_layer_stack_selection_returns_q08_and_endpoint_zero(self):
        import features_v11 as f

        base = f.Position(board=self._kings(), rest=((0,) * 6, (0,) * 6))
        self.assertEqual(f.layer_stack_selection(base), (0, 0))

        board = self._kings()
        board[f.square(0, 0)] = f.W_ROOK
        stepped = f.Position(board=board, rest=((0,) * 6, (0,) * 6))
        floor, blend_q8 = f.layer_stack_selection(stepped)
        self.assertEqual(floor, 0)
        self.assertEqual(blend_q8, 17)

        board = self._kings()
        for sq in range(30):
            if board[sq] == f.EMPTY:
                board[sq] = f.DARK_WHITE if sq % 2 else f.DARK_BLACK
        endpoint = f.Position(
            board=board, rest=((0, 0, 0, 15, 0, 0), (0, 0, 0, 15, 0, 0)))
        self.assertEqual(f.layer_stack_selection(endpoint), (15, 0))

    def test_v11_identity_documents_new_semantics(self):
        import features_v11 as f

        document = f.feature_identity_document()
        self.assertEqual(document["schema"], "abjchess-v11-feature-identity-v1")
        self.assertEqual(document["metadata_layout"], "pool-dark-owner-exact-loss-context-v1")
        self.assertEqual(document["layer_stack_selection"], "continuous-q0.8-v1")
        self.assertEqual(document["real_features"], 34800)
        self.assertEqual(document["unknown_loss_offsets"], [40, 56])
        self.assertEqual(document["pool_count_offsets"], [88, 94])
        self.assertEqual(len(f.feature_identity_sha256("HalfKAv2_hm_jieqi_v11")), 64)


class V11FeatureGoldenFixtureTests(unittest.TestCase):
    @staticmethod
    def _parse_board(board: str):
        import features_v11 as f

        pieces = {
            "R": f.W_ROOK, "A": f.W_ADVISOR, "C": f.W_CANNON,
            "P": f.W_PAWN, "N": f.W_KNIGHT, "B": f.W_BISHOP,
            "K": f.W_KING, "r": f.B_ROOK, "a": f.B_ADVISOR,
            "c": f.B_CANNON, "p": f.B_PAWN, "n": f.B_KNIGHT,
            "b": f.B_BISHOP, "k": f.B_KING,
            "X": f.DARK_WHITE, "x": f.DARK_BLACK,
        }
        ranks = []
        for rank in board.split("/"):
            row = []
            for token in rank:
                if token.isdigit():
                    row.extend([f.EMPTY] * int(token))
                else:
                    row.append(pieces[token])
            if len(row) != 9:
                raise ValueError("golden fixture rank must contain nine files")
            ranks.append(row)
        if len(ranks) != 10:
            raise ValueError("golden fixture board must contain ten ranks")
        return [piece for row in reversed(ranks) for piece in row]

    def test_python_helpers_match_checked_in_v11_golden_fixture(self):
        import features_v11 as f

        path = Path(__file__).parent / "fixtures" / "v11_feature_golden.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema"], "abjchess-v11-feature-golden-v1")
        self.assertEqual(
            payload["feature_identity_sha256"],
            f.feature_identity_sha256("HalfKAv2_hm_jieqi_v11"),
        )
        cases = payload["cases"]
        self.assertGreaterEqual(len(cases), 8)
        for case in cases:
            rest = tuple(tuple(int(value) for value in side) for side in case["rest"])
            unknown_loss = case.get("unknown_loss")
            position = f.Position(
                board=self._parse_board(case["board"]),
                rest=rest,
                side_to_move=int(case.get("side_to_move", 0)),
                unknown_loss=None if unknown_loss is None else tuple(unknown_loss),
            )
            self.assertEqual(
                list(f.layer_stack_selection(position)),
                [case["selection"]["floor"], case["selection"]["blend_q8"]],
                case["name"],
            )
            self.assertEqual(
                list(f.meta_feature_offsets(position, 0)),
                case["perspectives"]["0"]["meta_feature_offsets"],
                case["name"],
            )
            self.assertEqual(
                list(f.meta_feature_offsets(position, 1)),
                case["perspectives"]["1"]["meta_feature_offsets"],
                case["name"],
            )
            for perspective in (0, 1):
                expected = case["perspectives"][str(perspective)]
                self.assertEqual(
                    f.unknown_loss_count(position, perspective),
                    expected["unknown_loss_count"],
                    case["name"],
                )
                self.assertEqual(
                    list(f.visible_threat_summary(position, perspective)),
                    expected["threat_summary"],
                    case["name"],
                )

if __name__ == "__main__":
    unittest.main()
