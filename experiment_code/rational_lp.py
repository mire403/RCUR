from copy import deepcopy
from fractions import Fraction as F

if __package__:
    from .rcur_core import LPFailure, dot, rational
else:
    from rcur_core import LPFailure, dot, rational


class RationalSimplex:
    def __init__(self):
        self.lp = None

    @staticmethod
    def _pivot(state, row, column, budget):
        budget.consume()
        T, rhs, transform, basis = state
        scale = T[row][column]
        T[row] = [v / scale for v in T[row]]
        rhs[row] /= scale
        transform[row] = [v / scale for v in transform[row]]
        for i in range(len(T)):
            factor = T[i][column]
            if i == row or not factor:
                continue
            T[i] = [v - factor * w for v, w in zip(T[i], T[row])]
            rhs[i] -= factor * rhs[row]
            transform[i] = [v - factor * w for v, w in zip(transform[i], transform[row])]
        basis[row] = column

    @classmethod
    def _optimize(cls, state, cost, budget):
        T, rhs, transform, basis = state
        while True:
            reduced = list(cost)
            for i, basic in enumerate(basis):
                coefficient = cost[basic]
                if coefficient:
                    reduced = [v - coefficient * w for v, w in zip(reduced, T[i])]
            entering = next((j for j, v in enumerate(reduced) if v < 0), None)
            if entering is None:
                return sum((cost[b] * v for b, v in zip(basis, rhs)), F(0))
            eligible = [i for i in range(len(T)) if T[i][entering] > 0]
            if not eligible:
                raise LPFailure("unbounded_lp")
            leaving = min(eligible, key=lambda i: (rhs[i] / T[i][entering], basis[i]))
            cls._pivot(state, leaving, entering, budget)

    def find_feasible(self, lp, budget):
        self.lp = None
        n = len(lp.lo)
        rows = [(row, rhs, "A", i) for i, (row, rhs) in enumerate(zip(lp.A, lp.b))]
        rows += [(row, rhs, "E", i) for i, (row, rhs) in enumerate(zip(lp.E, lp.f))]
        rows += [({j: F(1)}, lp.hi[j], "box", j) for j in range(n)]
        specs, metadata, width = [], [], n
        artificial = set()
        for row, rhs, kind, index in rows:
            rhs -= dot(row, lp.lo)
            sign = F(1) if rhs >= 0 else F(-1)
            coeff = {j: sign * v for j, v in row.items()}
            basic = None
            if kind != "E":
                coeff[width] = sign
                if sign == 1:
                    basic = width
                width += 1
            if basic is None:
                basic = width
                artificial.add(width)
                coeff[width] = F(1)
                width += 1
            specs.append((coeff, sign * rhs, basic))
            metadata.append((kind, index, sign))
        m = len(specs)
        T = [[row.get(j, F(0)) for j in range(width)] for row, _, _ in specs]
        rhs = [v for _, v, _ in specs]
        basis = [j for _, _, j in specs]
        transform = [[F(i == j) for j in range(m)] for i in range(m)]
        state = (T, rhs, transform, basis)
        phase_one = [F(j in artificial) for j in range(width)]
        if self._optimize(state, phase_one, budget) != 0:
            raise LPFailure("infeasible_lp")
        i = 0
        while i < len(basis):
            if basis[i] not in artificial:
                i += 1
                continue
            entering = next((j for j in range(width)
                             if j not in artificial and j not in basis and T[i][j]), None)
            if entering is None:
                del T[i], rhs[i], transform[i], basis[i]
            else:
                self._pivot(state, i, entering, budget)
                i += 1
        keep = [j for j in range(width) if j not in artificial]
        new_index = {old: new for new, old in enumerate(keep)}
        T = [[row[j] for j in keep] for row in T]
        basis = [new_index[j] for j in basis]
        self.base = (T, rhs, transform, basis)
        self.metadata, self.width = metadata, len(keep)
        self.lp = lp
        witness = self._witness(self.base)
        if not lp.feasible(witness):
            self.lp = None
            raise LPFailure("feasibility_failed")
        return witness

    def _witness(self, state):
        point = list(self.lp.lo)
        _, rhs, _, basis = state
        for value, j in zip(rhs, basis):
            if j < len(point):
                point[j] += value
        return point

    def minimize(self, lp, objective, budget):
        if lp is not self.lp:
            raise LPFailure("missing_feasible_basis")
        cost = [rational(objective.get(j, 0)) for j in range(len(lp.lo))]
        cost += [F(0)] * (self.width - len(cost))
        state = deepcopy(self.base)
        self._optimize(state, cost, budget)
        if not lp.feasible(self._witness(state)):
            raise LPFailure("feasibility_failed")
        _, _, transform, basis = state
        dual = [sum((cost[b] * row[j] for b, row in zip(basis, transform)), F(0))
                for j in range(len(self.metadata))]
        eta, nu = [F(0)] * len(lp.A), [F(0)] * len(lp.E)
        for value, (kind, index, sign) in zip(dual, self.metadata):
            if kind == "A":
                eta[index] = -sign * value
            elif kind == "E":
                nu[index] = -sign * value
        return eta, nu
