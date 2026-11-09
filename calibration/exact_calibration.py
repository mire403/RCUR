from dataclasses import dataclass
from fractions import Fraction
from math import ceil, inf, isfinite, nextafter
from numbers import Integral, Real


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer count")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return int(value)


def exact_f1(candidate_length, reference_length, overlap):

    m = _integer(candidate_length, "candidate_length")
    n = _integer(reference_length, "reference_length")
    c = _integer(overlap, "overlap")
    if c > min(m, n):
        raise ValueError("overlap exceeds a sequence length")
    return Fraction(2 * c, m + n) if m + n else Fraction(1)


def exact_prediction(value):

    if isinstance(value, bool):
        raise TypeError("A boolean is not a prediction")
    if isinstance(value, Fraction):
        result = value
    elif isinstance(value, Integral):
        result = Fraction(int(value))
    elif isinstance(value, Real):
        value = float(value)
        if not isfinite(value):
            raise ValueError("Nonfinite prediction")
        result = Fraction.from_float(value)
    elif isinstance(value, str):
        result = Fraction(value)
    else:
        raise TypeError("Use a stored float, integer, Fraction or exact string")
    if abs(result) > 1:
        raise ValueError("Prediction is outside the clipped F1 gain range")
    return result


@dataclass(frozen=True)
class QuestionCounts:
    candidate_lengths: tuple
    reference_length: int
    overlaps: tuple
    anchor: int
    predictions: tuple

    def residuals(self):

        k = len(self.candidate_lengths)
        if not k or len(self.overlaps) != k or len(self.predictions) != k:
            raise ValueError("One length, overlap and prediction per candidate")
        anchor = _integer(self.anchor, "anchor")
        if anchor >= k:
            raise ValueError("Anchor index is outside the pool")
        scores = tuple(exact_f1(m, self.reference_length, c)
                       for m, c in zip(self.candidate_lengths, self.overlaps))
        predictions = tuple(map(exact_prediction, self.predictions))
        if predictions[anchor] != 0:
            raise ValueError("The anchor prediction must be exactly zero")
        return tuple(abs(score - scores[anchor] - pred)
                     for score, pred in zip(scores, predictions))


@dataclass(frozen=True)
class Calibration:
    scores: tuple
    rank: int
    radius: object
    alpha: Fraction

    def as_dict(self):

        return {"alpha": str(self.alpha), "packet_count": len(self.scores),
                "rank": self.rank, "packet_scores": list(map(str, self.scores)),
                "radius": "inf" if self.radius == inf else str(self.radius),
                "upward_float_radius": (None if self.radius == inf
                                         else upward_float(self.radius))}


def calibrate_from_counts(packets, alpha=Fraction(1, 20)):


    if isinstance(alpha, (bool, float)):
        raise TypeError("Specify alpha exactly, e.g. Fraction(1, 20) or '0.05'")
    alpha = Fraction(alpha)
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one")
    scores = []
    for packet in packets:
        residuals = [residual for question in packet
                     for residual in question.residuals()]
        if not residuals:
            raise ValueError("Calibration packets must contain questions")
        scores.append(max(residuals))
    if not scores:
        raise ValueError("Calibration needs at least one image packet")
    rank = ceil((len(scores) + 1) * (1 - alpha))
    radius = sorted(scores)[rank - 1] if rank <= len(scores) else inf
    return Calibration(tuple(scores), rank, radius, alpha)


def upward_float(radius):


    if not isinstance(radius, Fraction) or radius < 0:
        raise TypeError("Supply an exact nonnegative Fraction radius")
    value = float(radius)
    if not isfinite(value):
        raise ValueError("Radius is outside the finite binary64 range")
    if Fraction.from_float(value) < radius:
        value = nextafter(value, inf)
    assert Fraction.from_float(value) >= radius
    return value


def audit_stored_radius(calibration, stored_radius):

    if isinstance(stored_radius, str) and stored_radius == "inf":
        stored_radius = inf
    if stored_radius == inf:
        return {"encloses_exact_quantile": True, "shortfall": "0"}
    if isinstance(stored_radius, bool):
        raise TypeError("A boolean is not a radius")
    if isinstance(stored_radius, Fraction):
        pass
    elif isinstance(stored_radius, Real) and not isinstance(stored_radius, Integral):
        if not isfinite(stored_radius):
            raise ValueError("Invalid stored radius")
        stored_radius = Fraction.from_float(float(stored_radius))
    else:
        stored_radius = Fraction(stored_radius)
    if stored_radius < 0:
        raise ValueError("Stored radius must be nonnegative")
    if calibration.radius == inf:
        return {"encloses_exact_quantile": False, "shortfall": "inf"}
    deficit = max(Fraction(0), calibration.radius - stored_radius)
    return {"encloses_exact_quantile": deficit == 0, "shortfall": str(deficit)}


def read_trace(data):

    packets, image_ids = [], set()
    for packet in data["packets"]:
        image_id = packet["image_id"]
        if not isinstance(image_id, str) or not image_id or image_id in image_ids:
            raise ValueError("Each image_id must be a unique nonempty string")
        image_ids.add(image_id)
        questions, question_ids = [], set()
        for question in packet["questions"]:
            qid = question["question_id"]
            if not isinstance(qid, str) or not qid or qid in question_ids:
                raise ValueError("Question IDs must be unique within the image")
            question_ids.add(qid)
            candidates = question["candidates"]
            questions.append(QuestionCounts(
                tuple(c["length"] for c in candidates), question["reference_length"],
                tuple(c["overlap"] for c in candidates), question["anchor_index"],
                tuple(c["prediction"] for c in candidates)))
        packets.append(questions)
    return packets


def main():
    import argparse
    import hashlib
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description="Compute an exact packet calibration radius from token counts.")
    parser.add_argument("trace", type=Path)
    parser.add_argument("--alpha", default="1/20")
    parser.add_argument("--stored-radius", help="Exact decimal/fraction or float hex")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raw = args.trace.read_bytes()
    data = json.loads(raw)
    result = calibrate_from_counts(read_trace(data), args.alpha)
    record = result.as_dict()
    record["trace_sha256"] = hashlib.sha256(raw).hexdigest()
    record["implementation"] = "Revision 33 exact-count calibration update"
    if args.stored_radius is not None:
        stored = args.stored_radius
        if "0x" in stored.lower():
            stored = float.fromhex(stored)
        record["stored_radius_check"] = audit_stored_radius(result, stored)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
