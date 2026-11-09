from fractions import Fraction as F
from itertools import combinations, product
from math import inf
import unittest

import numpy as np

from experiment_code.rcur_core import (
    BoundedLP, PivotBudget, build_joint_lp, calibrate_radius,
    difference, dot, fit_gain_model, occurrence_set, predict_gains,
    revise, verified_lower_bound,
)
from experiment_code.rational_lp import RationalSimplex


def action_bounds(lp, count, anchor):
    solver, budget = RationalSimplex(), PivotBudget()
    witness = solver.find_feasible(lp, budget)
    values = []
    for a in range(count):
        objective = difference(a, anchor)
        eta, nu = solver.minimize(lp, objective, budget)
        values.append(verified_lower_bound(lp, objective, witness, eta, nu))
    return values


def vertices(lp):
    edges = list(zip(lp.A, lp.b)) + list(zip(lp.E, lp.f))
    edges += [({j: F(1)}, end) for j in range(2) for end in (lp.lo[j], lp.hi[j])]
    result = []
    for (row, b), (other, f) in combinations(edges, 2):
        a, c, d, e = row.get(0, F(0)), row.get(1, F(0)), other.get(0, F(0)), other.get(1, F(0))
        determinant = a * e - c * d
        if determinant:
            point = [(b * e - c * f) / determinant, (a * f - b * d) / determinant]
            if lp.feasible(point):
                result.append(point)
    return result


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.tokens = [s.split() for s in ('red polyp', 'red lesion', 'blue polyp', 'blue lesion')]
        self.predictions = [F(24, 100), F(25, 100), F(25, 100), F(0)]
        self.radius = F(12, 100)
        self.identities = [' '.join(t) for t in self.tokens]

    def test_shared_reference_relation_changes_selection(self):
        coupled = action_bounds(build_joint_lp(self.tokens, 3, self.predictions, self.radius), 4, 3)
        box = action_bounds(build_joint_lp(self.tokens, 3, self.predictions, self.radius, coupled=False), 4, 3)
        self.assertEqual(coupled, [F(26, 100), F(13, 100), F(13, 100), F(0)])
        self.assertEqual(box, [F(12, 100), F(13, 100), F(13, 100), F(0)])
        decision = revise(self.tokens, self.identities, 3, self.predictions, self.radius, RationalSimplex())
        self.assertEqual(decision.index, 0)
        self.assertEqual(min(range(4), key=lambda a: (-box[a], a != 3, self.identities[a])), 2)

    def test_reference_utilities_satisfy_envelope(self):
        pool = [[], ['a'], ['a', 'a'], ['a', 'b'], ['b', 'a'], ['b', 'b', 'a']]
        lp = build_joint_lp(pool, 0, [F(0)] * len(pool), F(1))
        checked = 0
        for length in range(5):
            for reference in product(('a', 'b', 'outside'), repeat=length):
                y = occurrence_set(reference)
                witness = [F(0)] * len(lp.lo)
                for a, candidate in enumerate(pool):
                    tokens = occurrence_set(candidate)
                    witness[a] = F(2 * len(tokens & y), len(tokens) + len(y)) if tokens or y else F(1)
                for (ell, token), j in lp.z_index.items():
                    witness[j] = F(2 * (token in y), ell + len(y))
                self.assertTrue(lp.feasible(witness))
                checked += 1
        self.assertEqual(checked, 121)

    def test_lp_matches_independent_vertex_enumeration(self):
        for trial in range(24):
            lp = BoundedLP()
            lp.variable(-1, 2)
            lp.variable(0, 3)
            feasible = [F(1, 3), F(2, 3)]
            for k in range(4):
                row = {0: F((trial + k) % 7 - 3), 1: F((2 * trial + 3 * k) % 7 - 3)}
                lp.le(row, dot(row, feasible) + F((trial + k) % 5, 3))
            if trial % 3 == 0:
                lp.eq({0: F(1), 1: F(1)}, 1)
                if trial % 6 == 0:
                    lp.eq({0: F(2), 1: F(2)}, 2)
            objective = {0: F(trial % 7 - 3), 1: F((trial + 3) % 7 - 3)}
            solver, budget = RationalSimplex(), PivotBudget()
            witness = solver.find_feasible(lp, budget)
            eta, nu = solver.minimize(lp, objective, budget)
            lower = verified_lower_bound(lp, objective, witness, eta, nu)
            self.assertEqual(lower, min(dot(objective, point) for point in vertices(lp)))

    def test_image_balanced_ridge_and_clipping(self):
        features = np.zeros((2, 83))
        features[1, 0] = 1.0
        positive = (features, 0, np.array([0.0, 1.0]))
        negative = (features, 0, np.array([0.0, -1.0]))
        theta = fit_gain_model([[positive], [negative, negative, negative]], 0.1)
        self.assertLess(np.max(np.abs(theta)), 1e-14)
        theta = fit_gain_model([[positive]], 0.1)
        self.assertLess(abs(theta[0] - 5 / 6), 1e-14)
        self.assertEqual(np.count_nonzero(theta[1:]), 0)
        self.assertEqual(list(predict_gains(features, 0, theta * 100)), [0.0, 1.0])

    def test_packet_maximum_and_finite_sample_rank(self):
        truth = [np.array([j / 100, 0.0]) for j in range(1, 99)]
        prediction = [np.zeros(2) for _ in truth]
        self.assertEqual(calibrate_radius(truth, prediction, 0.05), 0.95)
        self.assertEqual(calibrate_radius(truth, prediction, 0.01), inf)
        self.assertEqual(calibrate_radius(truth[:18], prediction[:18], 0.05), inf)
        self.assertEqual(calibrate_radius(truth[:19], prediction[:19], 0.05), 0.19)
        self.assertEqual(calibrate_radius([[0.1, -0.7, 0.4]], [[0, 0, 0]], F(1, 2)), 0.7)

    def test_verified_bound_direction_and_feasible_witness(self):
        lp = BoundedLP()
        lp.variable(0, 1)
        lp.le({0: -1}, F(-2, 5))
        self.assertEqual(verified_lower_bound(lp, {0: 1}, [F(1)], [F(0)], []), 0)
        self.assertEqual(verified_lower_bound(lp, {0: 1}, [F(1)], [F(1)], []), F(2, 5))
        self.assertEqual(verified_lower_bound(lp, {0: 1}, [F(1)], [F(1, 2)], []), F(1, 5))
        self.assertEqual(verified_lower_bound(lp, {0: 1}, [F(1)], [F(-1)], []), 0)
        self.assertIsNone(verified_lower_bound(lp, {0: 1}, [F(0)], [F(1)], []))

    def test_anchor_retention_budget_ties_and_threshold(self):
        class UnusedSolver:
            def find_feasible(self, *args):
                raise AssertionError('The radius check must precede LP construction')
        self.assertEqual(revise(self.tokens, self.identities, 3, self.predictions, 1, UnusedSolver()).index, 3)
        self.assertEqual(revise(self.tokens, self.identities, 3, self.predictions, inf, UnusedSolver()).index, 3)
        failed = revise(self.tokens, self.identities, 3, self.predictions, self.radius, RationalSimplex(), pivot_limit=0)
        self.assertEqual((failed.index, failed.reason), (3, 'pivot_budget'))
        bad = revise([['x'], ['x']], ['x!', 'x?'], 1, [F(1, 2), F(0)], F(1, 10), RationalSimplex())
        self.assertEqual((bad.index, bad.reason), (1, 'infeasible_lp'))
        tied = revise([['x'], ['x'], ['y']], ['z', 'a', 'y'], 2, [F(1, 2), F(1, 2), F(0)], F(1, 10), RationalSimplex())
        self.assertEqual(tied.index, 1)
        self.assertEqual(tied.bounds[0], tied.bounds[1])
        threshold = revise(self.tokens, self.identities, 3, self.predictions, self.radius, RationalSimplex(), threshold=F(26, 100))
        self.assertEqual((threshold.index, threshold.reason), (3, 'threshold'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
