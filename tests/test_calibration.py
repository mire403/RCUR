from fractions import Fraction as F
from math import inf, nextafter
from pathlib import Path
import json
import sys
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiment_code'))
from experiment_code.rcur_core import calibrate_radius, build_joint_lp, occurrence_set, revise
from experiment_code.rational_lp import RationalSimplex
from calibration.exact_calibration import QuestionCounts, exact_f1, exact_prediction, calibrate_from_counts, upward_float, audit_stored_radius, read_trace

class ExactCalibrationTests(unittest.TestCase):

    def boundary_question(self):
        return QuestionCounts((6, 6), 6, (2, 0), 1, (0.25, 0.0))

    def test_rounding_boundary_and_lp_membership(self):
        a = ['one', 'two', 'three', 'four', 'five', 'six']
        d = ['anchor0', 'anchor1', 'anchor2', 'anchor3', 'anchor4', 'anchor5']
        reference = ['one', 'two', 'new1', 'new2', 'new3', 'new4']
        Y = occurrence_set(reference)
        old = calibrate_radius([[F(1, 3), 0]] * 19, [[0.25, 0]] * 19)
        corrected = calibrate_from_counts([[self.boundary_question()]] * 19)
        self.assertEqual(corrected.radius, F(1, 12))
        self.assertLess(F.from_float(old), corrected.radius)
        self.assertFalse(audit_stored_radius(corrected, old)['encloses_exact_quantile'])
        for radius, expected in [(old, False), (corrected.radius, True), (upward_float(corrected.radius), True)]:
            lp = build_joint_lp([a, d], 1, [0.25, 0.0], radius)
            witness = [F(0)] * len(lp.lo)
            witness[0] = F(1, 3)
            for (ell, occurrence), j in lp.z_index.items():
                witness[j] = F(2 * (occurrence in Y), ell + len(Y))
            self.assertEqual(lp.feasible(witness), expected)
        decision = revise([a, d], ['a', 'd'], 1, [0.25, 0.0], corrected.radius, RationalSimplex())
        self.assertEqual(decision.index, 0)
        self.assertLessEqual(decision.bounds[0], F(1, 3))
        self.assertGreater(decision.bounds[0], 0)

    def test_packet_maximum_includes_all_questions(self):
        low = self.boundary_question()
        high = QuestionCounts((1, 1), 1, (1, 0), 1, (0.5, 0))
        result = calibrate_from_counts([[low, high], [low]], alpha=F(1, 2))
        self.assertEqual(result.scores, (F(1, 2), F(1, 12)))
        self.assertEqual(result.rank, 2)
        self.assertEqual(result.radius, F(1, 2))

    def test_finite_sample_cutoff(self):
        q = self.boundary_question()
        self.assertEqual(calibrate_from_counts([[q]] * 18).radius, inf)
        self.assertEqual(calibrate_from_counts([[q]] * 19).rank, 19)
        self.assertEqual(calibrate_from_counts([[q]] * 98).rank, 95)
        self.assertEqual(calibrate_from_counts([[q]] * 98, '0.01').radius, inf)

    def test_rank_not_maximum_when_enough_packets(self):
        packets = [[QuestionCounts((1, 1), 1, (0, 0), 1, (F(i, 100), 0))] for i in range(1, 99)]
        result = calibrate_from_counts(packets)
        self.assertEqual(result.radius, F(95, 100))

    def test_empty_answers_and_large_counts(self):
        self.assertEqual(exact_f1(0, 0, 0), 1)
        self.assertEqual(exact_f1(1, 0, 0), 0)
        self.assertEqual(exact_f1(0, 7, 0), 0)
        large = 10 ** 50 + 3
        self.assertEqual(exact_f1(large, large, large), 1)
        q = QuestionCounts((0, 1), 0, (0, 0), 0, (0, 0))
        self.assertEqual(q.residuals(), (0, 1))

    def test_preserves_binary_prediction(self):
        self.assertEqual(exact_prediction(0.1), F.from_float(0.1))
        self.assertNotEqual(exact_prediction(0.1), F(1, 10))
        self.assertEqual(exact_prediction('1/10'), F(1, 10))

    def test_outward_float_enclosure_and_minimality(self):
        radii = [F(0), F(1, 12), F(1, 10), F(2), F(1, 10 ** 400)]
        radii += [F(j * 123456791 % 10 ** 10, 10 ** 10 + 19) for j in range(200)]
        for r in radii:
            f = upward_float(r)
            self.assertGreaterEqual(F.from_float(f), r)
            self.assertLess(F.from_float(nextafter(f, -inf)), r)

    def test_exact_json_roundtrip_and_radius_audit(self):
        result = calibrate_from_counts([[self.boundary_question()]] * 19)
        stored = json.loads(json.dumps(result.as_dict(), allow_nan=False))
        self.assertEqual(F(stored['radius']), F(1, 12))
        self.assertTrue(audit_stored_radius(result, F(1, 12))['encloses_exact_quantile'])
        self.assertTrue(audit_stored_radius(result, inf)['encloses_exact_quantile'])
        infinite = calibrate_from_counts([[self.boundary_question()]])
        self.assertFalse(audit_stored_radius(infinite, F(2))['encloses_exact_quantile'])
        json.dumps(infinite.as_dict(), allow_nan=False)

    def test_malformed_counts_and_predictions_fail(self):
        for values in [(1, 1, 2), (-1, 1, 0), (1.0, 1, 0), (True, 1, 0)]:
            with self.assertRaises((ValueError, TypeError)):
                exact_f1(*values)
        for pred in [float('nan'), inf, -inf, 1.01, True]:
            with self.assertRaises((ValueError, TypeError)):
                exact_prediction(pred)
        with self.assertRaises(ValueError):
            QuestionCounts((1,), 1, (1,), 0, (0.1,)).residuals()

    def test_empty_packets_and_ambiguous_alpha_fail(self):
        for packets in [[], [[]]]:
            with self.assertRaises(ValueError):
                calibrate_from_counts(packets)
        for alpha in [0.05, 0, 1, 'nan']:
            with self.assertRaises((ValueError, TypeError, ZeroDivisionError)):
                calibrate_from_counts([[self.boundary_question()]], alpha)

    def test_trace_schema_and_duplicate_ids(self):
        packet = {'image_id': 'packet-1', 'questions': [{'question_id': 'question-1', 'reference_length': 6, 'anchor_index': 1, 'candidates': [{'length': 6, 'overlap': 2, 'prediction': 0.25}, {'length': 6, 'overlap': 0, 'prediction': 0.0}]}]}
        packets = read_trace({'packets': [packet]})
        self.assertEqual(packets[0][0].residuals(), (F(1, 12), 0))
        with self.assertRaises(ValueError):
            read_trace({'packets': [packet, packet]})
if __name__ == '__main__':
    unittest.main(verbosity=2)
