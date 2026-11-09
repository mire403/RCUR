import argparse
from fractions import Fraction
import json
from math import inf
from pathlib import Path

from experiment_code.rcur_core import (
    Decision, calibrate_radius, fit_gain_model, predict_gains, revise,
)
from experiment_code.rational_lp import RationalSimplex


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8")


def number(value):
    if isinstance(value, str):
        if value == "inf":
            return inf
        if "0x" in value.lower():
            return float.fromhex(value)
        return Fraction(value)
    return value


def fit(args):
    data = read_json(args.input)
    packets = [[(q["features"], q["anchor_index"], q["gains"])
                for q in packet["questions"]] for packet in data["packets"]]
    theta = fit_gain_model(packets, args.lam)
    write_json(args.output, {
        "feature_dimension": 83,
        "lambda": args.lam,
        "theta": None if theta is None else theta.tolist(),
        "status": "training_failed" if theta is None else "fitted",
    })


def calibrate(args):
    data = read_json(args.input)
    gains, predictions = [], []
    for packet in data["packets"]:
        image_gains, image_predictions = [], []
        for q in packet["questions"]:
            truth, predicted = q["gains"], q["predictions"]
            if not truth or len(truth) != len(predicted):
                raise ValueError("Each question needs matching nonempty gain and prediction arrays")
            image_gains.extend(number(v) for v in truth)
            image_predictions.extend(number(v) for v in predicted)
        gains.append(image_gains)
        predictions.append(image_predictions)
    radius = calibrate_radius(gains, predictions, Fraction(args.alpha))
    write_json(args.output, {
        "implementation": "rcur_core.calibrate_radius",
        "alpha": str(Fraction(args.alpha)),
        "packet_count": len(gains),
        "radius": "inf" if radius == inf else radius,
        "radius_hex": None if radius == inf else radius.hex(),
    })


def infer(args):
    data = read_json(args.input)
    calibration = read_json(args.calibration)
    radius = number(calibration.get("radius_hex") or calibration["radius"])
    model = read_json(args.model) if args.model else None
    outputs = []
    for q in data["questions"]:
        candidates = q["candidates"]
        anchor = q["anchor_index"]
        if isinstance(anchor, bool) or not isinstance(anchor, int):
            raise ValueError("anchor_index must be an integer")
        if not 0 <= anchor < len(candidates):
            raise ValueError("anchor_index is outside the candidate pool")
        tokens, identities, strings = [], [], []
        for candidate in candidates:
            if not isinstance(candidate["text"], str) or not isinstance(candidate["identity"], str):
                raise ValueError("Candidate text and identity must be strings")
            if not isinstance(candidate["tokens"], list) or not all(isinstance(t, str) for t in candidate["tokens"]):
                raise ValueError("Candidate tokens must be a list of strings")
            strings.append(candidate["text"])
            identities.append(candidate["identity"])
            tokens.append(candidate["tokens"])
        if len(set(identities)) != len(identities):
            raise ValueError("Candidate identities must be unique")
        if model is not None and model["theta"] is None:
            decision = Decision(anchor, {anchor: Fraction(0)}, "training_failed", 0)
        else:
            if model is None:
                predictions = [number(v) for v in q["predictions"]]
            else:
                if "predictions" in q:
                    raise ValueError("With --model, supply features instead of stored predictions")
                predictions = predict_gains(q["features"], anchor, model["theta"])
            decision = revise(tokens, identities, anchor, predictions, radius,
                              RationalSimplex(), pivot_limit=args.pivot_limit,
                              threshold=Fraction(args.threshold))
        outputs.append({
            "question_id": q["question_id"],
            "selected_index": decision.index,
            "answer": strings[decision.index],
            "revised": decision.index != anchor,
            "reason": decision.reason,
            "pivots": decision.pivots,
            "bounds": {str(k): str(v) for k, v in decision.bounds.items()},
        })
    write_json(args.output, {"questions": outputs})


def main():
    parser = argparse.ArgumentParser(description="Run RCUR fitting, packet calibration and anchored revision.")
    commands = parser.add_subparsers(dest="command", required=True)
    fitting = commands.add_parser("fit")
    fitting.add_argument("--lambda", dest="lam", type=float, required=True)
    fitting.set_defaults(action=fit)
    calibration = commands.add_parser("calibrate")
    calibration.add_argument("--alpha", default="1/20")
    calibration.set_defaults(action=calibrate)
    revision = commands.add_parser("revise")
    revision.add_argument("--calibration", type=Path, required=True)
    revision.add_argument("--model", type=Path)
    revision.add_argument("--pivot-limit", type=int, default=50000)
    revision.add_argument("--threshold", default="1/100000000")
    revision.set_defaults(action=infer)
    for command in (fitting, calibration, revision):
        command.add_argument("--input", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.action(args)


if __name__ == "__main__":
    main()
