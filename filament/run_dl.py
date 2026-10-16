"""CLI for the deep-learning extension: train, tune, evaluate once, write a candidate CSV.

    python -m filament.run_dl --data MAGFiLO_1.0_Kaggle_2026 --out results/dl
    python -m filament.run_dl --debug     # tiny end-to-end check, a few minutes on CPU

The CSV is written to ``<out>/candidate_submission.csv``; nothing is uploaded to Kaggle.
"""
import argparse
import json
from dataclasses import replace
from pathlib import Path

from .dl import TrainConfig, run_experiment

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--out", type=Path, default=ROOT / "results/dl")
    parser.add_argument("--split", type=Path, default=ROOT / "configs/split.json")
    parser.add_argument("--debug", action="store_true", help="tiny settings for a quick pipeline check")
    for field, kind in (("epochs", int), ("steps_per_epoch", int), ("batch_size", int), ("crop", int),
                        ("lr", float), ("workers", int)):
        parser.add_argument(f"--{field.replace('_', '-')}", type=kind, default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    overrides = {k: v for k, v in vars(args).items()
                 if k in ("epochs", "steps_per_epoch", "batch_size", "crop", "lr", "workers") and v is not None}
    if args.no_pretrained:
        overrides["pretrained"] = False
    cfg = replace(TrainConfig(), **overrides)
    split = json.loads(args.split.read_text())
    run_experiment(args.data, split, args.out, cfg, debug=args.debug)


if __name__ == "__main__":
    main()
