from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction as F
from itertools import combinations
from math import ceil, inf, isfinite

import numpy as np


def rational(value):

    if isinstance(value, F):
        return value
    if isinstance(value, (float, np.floating)):
        return F.from_float(float(value))
    if isinstance(value, np.integer):
        return F(int(value))
    return F(value)


def dot(row, vector):
    return sum((value * vector[j] for j, value in row.items()), F(0))


def difference(a, d):
    return {} if a == d else {a: F(1), d: F(-1)}


class LPFailure(RuntimeError):
    pass


@dataclass
class PivotBudget:
    limit: int = 50000
    used: int = 0

    def consume(self):
        if self.used >= self.limit:
            raise LPFailure("pivot_budget")
        self.used += 1


@dataclass
class BoundedLP:
    lo: list = field(default_factory=list)
    hi: list = field(default_factory=list)
    A: list = field(default_factory=list)
    b: list = field(default_factory=list)
    E: list = field(default_factory=list)
    f: list = field(default_factory=list)
    z_index: dict = field(default_factory=dict)

    def variable(self, lower, upper):
        lower, upper = rational(lower), rational(upper)
        if lower > upper:
            raise ValueError("Reversed variable bounds")
        self.lo.append(lower)
        self.hi.append(upper)
        return len(self.lo) - 1

    def _row(self, coefficients):
        if any(j < 0 or j >= len(self.lo) for j in coefficients):
            raise ValueError("Unknown variable index")
        return {j: rational(v) for j, v in coefficients.items() if v}

    def le(self, coefficients, rhs):
        self.A.append(self._row(coefficients))
        self.b.append(rational(rhs))

    def eq(self, coefficients, rhs):
        self.E.append(self._row(coefficients))
        self.f.append(rational(rhs))

    def feasible(self, point):
        if point is None or len(point) != len(self.lo):
            return False
        try:
            v = list(map(rational, point))
        except (ValueError, TypeError, OverflowError):
            return False
        return (
            all(l <= x <= h for l, x, h in zip(self.lo, v, self.hi))
            and all(dot(row, v) <= rhs for row, rhs in zip(self.A, self.b))
            and all(dot(row, v) == rhs for row, rhs in zip(self.E, self.f))
        )


def _features(phi, anchor):
    phi = np.asarray(phi, dtype=float)
    if phi.ndim != 2 or phi.shape[1] != 83 or not 0 <= anchor < len(phi):
        raise ValueError("Expected K by 83 normalized features and an anchor")
    if not np.isfinite(phi).all():
        raise ValueError("Nonfinite feature")
    return phi


def _fitting_packets(packets):
    packets = tuple(tuple(packet) for packet in packets)
    if not packets or any(not packet for packet in packets):
        raise ValueError("Fitting needs nonempty image packets")
    checked = []
    for packet in packets:
        group = []
        for phi, anchor, gain in packet:
            phi = _features(phi, anchor)
            gain = np.asarray(gain, dtype=float)
            if gain.shape != (len(phi),) or not np.isfinite(gain).all():
                raise ValueError("One finite target gain per candidate is required")
            if gain[anchor] != 0 or np.any(np.abs(gain) > 1):
                raise ValueError("Targets must be anchor-relative F1 gains")
            group.append((phi, anchor, gain))
        checked.append(group)
    return checked


def fit_gain_model(packets, lam):
    packets = _fitting_packets(packets)
    if not isfinite(lam) or lam <= 0:
        raise ValueError("Positive ridge penalty required")
    H, b = lam * np.eye(83), np.zeros(83)
    for packet in packets:
        for phi, d, gain in packet:
            X = phi - phi[d]
            w = 1.0 / (len(packets) * len(packet) * len(phi))
            H += w * (X.T @ X)
            b += w * (X.T @ gain)
    tolerance = 1e-10 * (1 + np.linalg.norm(b, np.inf))
    for method in ("cholesky", "qr"):
        try:
            if method == "cholesky":
                L = np.linalg.cholesky(H)
                theta = np.linalg.solve(L.T, np.linalg.solve(L, b))
            else:
                Q, R = np.linalg.qr(H)
                theta = np.linalg.solve(R, Q.T @ b)
            if np.linalg.norm(H @ theta - b, np.inf) <= tolerance:
                return theta
        except np.linalg.LinAlgError:
            pass
    return None


def predict_gains(phi, anchor, theta):
    phi = _features(phi, anchor)
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (83,) or not np.isfinite(theta).all():
        raise ValueError("Expected 83 finite fitted coefficients")
    gains = np.clip((phi - phi[anchor]) @ theta, -1.0, 1.0)
    gains[anchor] = 0.0
    return gains


def occurrence_set(tokens):
    counts = Counter(tokens)
    return {(token, j) for token, n in counts.items() for j in range(1, n + 1)}


def build_joint_lp(tokens, anchor, predictions, radius, coupled=True):
    K = len(tokens)
    pred, r = list(map(rational, predictions)), rational(radius)
    if not K or len(pred) != K or not 0 <= anchor < K or r < 0:
        raise ValueError("Invalid candidate pool or radius")
    if pred[anchor] != 0 or any(abs(p) > 1 for p in pred):
        raise ValueError("Predictions must be clipped anchor-relative gains")
    A = [occurrence_set(t) for t in tokens]
    lp = BoundedLP()
    for _ in range(K):
        lp.variable(0, 1)
    if coupled:
        for length in sorted({len(s) for s in A} - {0}):
            group = [a for a in range(K) if len(A[a]) == length]
            vocabulary = sorted(set().union(*(A[a] for a in group)))
            z = {v: lp.variable(0, F(2, length + 1)) for v in vocabulary}
            lp.z_index.update({(length, v): j for v, j in z.items()})
            for a in group:
                row = {a: F(1)}
                row.update({z[v]: F(-1) for v in A[a]})
                lp.eq(row, 0)
            lp.le({j: F(1) for j in z.values()}, 2)
        for a, b in combinations(range(K), 2):
            union = A[a] | A[b]
            distance = F(len(A[a] ^ A[b]), len(union)) if union else F(0)
            bound = min(F(1), 2 * distance)
            lp.le(difference(a, b), bound)
            lp.le(difference(b, a), bound)
    for a in range(K):
        if a != anchor:
            lp.le(difference(a, anchor), pred[a] + r)
            lp.le(difference(anchor, a), r - pred[a])
    return lp


def calibrate_radius(gain_packets, prediction_packets, alpha=F(1, 20)):
    if len(gain_packets) != len(prediction_packets) or not gain_packets:
        raise ValueError("Supply matching nonempty image-packet lists")
    alpha = F(str(alpha))
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    residuals = []
    for truth, predicted in zip(gain_packets, prediction_packets):
        truth, predicted = np.asarray(truth, float), np.asarray(predicted, float)
        if truth.ndim != 1 or truth.shape != predicted.shape or not truth.size:
            raise ValueError("Flatten all registered candidate gains per image")
        if not (np.isfinite(truth).all() and np.isfinite(predicted).all()):
            raise ValueError("Nonfinite calibration gain")
        residuals.append(float(np.max(np.abs(truth - predicted))))
    k = ceil((len(residuals) + 1) * (1 - alpha))
    return sorted(residuals)[k - 1] if k <= len(residuals) else inf


def verified_lower_bound(lp, objective, witness, eta, nu):
    if not lp.feasible(witness):
        return None
    if len(eta) != len(lp.A) or len(nu) != len(lp.E):
        return None
    try:
        eta = [max(F(0), rational(t)) for t in eta]
        nu = list(map(rational, nu))
        c = lp._row(objective)
    except (ValueError, TypeError, OverflowError):
        return None
    s = [c.get(j, F(0)) for j in range(len(lp.lo))]
    for multiplier, row in zip(eta, lp.A):
        for j, value in row.items():
            s[j] += multiplier * value
    for multiplier, row in zip(nu, lp.E):
        for j, value in row.items():
            s[j] += multiplier * value
    offset = -sum((t * b for t, b in zip(eta, lp.b)), F(0))
    offset -= sum((t * f for t, f in zip(nu, lp.f)), F(0))
    return offset + sum(
        (min(t * l, t * h) for t, l, h in zip(s, lp.lo, lp.hi)), F(0)
    )


@dataclass
class Decision:
    index: int
    bounds: dict
    reason: str
    pivots: int


def revise(tokens, identities, anchor, predictions, radius, solver,
           pivot_limit=50000, threshold=F(1, 100000000)):
    K = len(tokens)
    if len(identities) != K or len(set(identities)) != K or not 0 <= anchor < K:
        raise ValueError("Supply unique normalized identities and a valid anchor")
    if rational(threshold) < 0:
        raise ValueError("Revision threshold must be nonnegative")
    budget, bounds = PivotBudget(pivot_limit), {anchor: F(0)}
    def retain(reason):
        return Decision(anchor, dict(bounds), reason, budget.used)
    if K == 1 or radius == inf:
        return retain("anchor_only")
    pred, r = list(map(rational, predictions)), rational(radius)
    if len(pred) != K or pred[anchor] != 0 or r < 0 or any(abs(p) > 1 for p in pred):
        raise ValueError("Invalid gain predictions or calibration radius")
    if r >= max(map(abs, pred)):
        return retain("zero_vector_feasible")
    lp = build_joint_lp(tokens, anchor, pred, r)
    try:
        witness = solver.find_feasible(lp, budget)
        if not lp.feasible(witness):
            return retain("feasibility_failed")
        for a in range(K):
            if a == anchor:
                continue
            c = difference(a, anchor)
            eta, nu = solver.minimize(lp, c, budget)
            bound = verified_lower_bound(lp, c, witness, eta, nu)
            if bound is None:
                return retain("bound_verification_failed")
            bounds[a] = bound
    except LPFailure as failure:
        return retain(str(failure))
    best = min(range(K), key=lambda a: (-bounds[a], a != anchor, identities[a]))
    if bounds[best] <= rational(threshold):
        return retain("threshold")
    return Decision(best, bounds, "revised", budget.used)
