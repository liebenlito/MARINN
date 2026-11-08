"""Check archived homogeneity-matched 1D heat baseline diagnostics.

Run from the archive's ``code`` directory:
  python verify_heat1d_flat_l4.py
  python verify_heat1d_flat_l4.py --retrain
  python verify_heat1d_flat_l4.py --device cpu
  python verify_heat1d_flat_l4.py --device cuda --retrain

The default check recomputes the archived diagnostic metrics from the
saved trained-model checkpoints.  The optional --retrain mode also trains
the five flat-L4-squared models for 1200 epochs each and compares their
validation objective and parameters with the original checkpoints.
This verifier does not modify any archived result file.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch

from gpu_utils import select_device
from marinn_core import (
    Config,
    MLP,
    data_sets,
    evaluate,
    residual_heat,
    set_seed,
    train,
)

ROOT = Path(__file__).resolve().parent.parent
BENCHMARK = ROOT / "checkpoints" / "heat_1d"


def midpoint_grid(n: int, device="cpu"):
    """The archived checkpoint diagnostics use 320x320 cell midpoints."""
    coordinates = (torch.arange(n, dtype=torch.float64, device=device) + 0.5) / n
    tt, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    return xx.reshape(-1, 1), tt.reshape(-1, 1), n, n


def archived_checkpoint(seed: int, device="cpu"):
    filename = BENCHMARK / f"seed{seed}" / "Flat-L4-squared.pt"
    model = MLP()
    model.load_state_dict(torch.load(filename, map_location="cpu", weights_only=True))
    return model.to(device=device, dtype=torch.float64)


def compare_metrics(actual, row, atol=2e-10):
    mapping = {
        "rel_L2_exact": "rel_L2",
        "residual_L2tL4": "eta_MR",
        "E_MR": "E_MR",
        "max_residual": "max_grid_residual",
    }
    maximum = 0.0
    for computed, saved in mapping.items():
        delta = abs(float(actual[computed]) - float(row[saved]))
        maximum = max(maximum, delta)
        if not math.isfinite(delta) or delta > atol:
            raise AssertionError(
                f"{saved}: computed {actual[computed]:.15g}, "
                f"archived {row[saved]:.15g}, absolute difference {delta:.3g}"
            )
    return maximum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrain", action="store_true",
                        help="also train each flat-L4-squared model for 1200 Adam epochs")
    parser.add_argument("--device", default=None,
                        help="auto, cpu, cuda, or cuda:<index>; defaults to MARINN_DEVICE or auto")
    args = parser.parse_args()
    device = select_device(args.device)
    cfg = Config()
    torch.set_num_threads(cfg.threads)
    archived = pd.read_csv(BENCHMARK / "heat1d_homogeneity_matched_raw.csv")
    dense_grid = midpoint_grid(320, device)

    results = []
    for seed in range(2024, 2029):
        rows = archived[(archived["seed"] == seed) &
                        (archived["method"] == "Flat-L4-squared")]
        if len(rows) != 1:
            raise ValueError(f"Expected exactly one archived row for seed {seed}")
        row = rows.iloc[0]
        saved_model = archived_checkpoint(seed, device)
        computed = evaluate(saved_model, dense_grid, cfg, "heat")
        max_metric_error = compare_metrics(computed, row)
        result = {"seed": seed, "checkpoint_metric_max_abs_error": max_metric_error}

        if args.retrain:
            set_seed(seed)
            data = data_sets(seed, cfg)
            # Generate the same seeded CPU samples before transferring to CUDA.
            data = {
                name: (pack[0].to(device), pack[1].to(device), *pack[2:])
                for name, pack in data.items()
            }
            model = MLP(cfg.hidden, cfg.depth).to(device)
            model, _, metadata = train(
                model, data["train"], data["val"], residual_heat,
                cfg, loss_name="flat_l4_squared",
            )
            validation_difference = abs(float(metadata["best_val"]) - float(row["best_val"]))
            state = torch.load(
                BENCHMARK / f"seed{seed}" / "Flat-L4-squared.pt",
                map_location="cpu", weights_only=True,
            )
            parameter_difference = max(
                float((value.detach().cpu() - state[name]).abs().max())
                for name, value in model.state_dict().items()
            )
            result.update({
                "validation_loss_abs_error": validation_difference,
                "parameter_max_abs_error": parameter_difference,
                "best_epoch": metadata["best_epoch"],
            })
            # Platform-level numerical differences are expected; these
            # thresholds are for checking gross departures, not bitwise identity.
            if (not math.isfinite(validation_difference)
                    or not math.isfinite(parameter_difference)
                    or validation_difference > 1e-7 or parameter_difference > 1e-3):
                raise AssertionError(f"Retraining differs materially for seed {seed}: {result}")

        results.append(result)
        print(result, flush=True)

    print(f"PASS: {len(results)} checkpoint diagnostic comparisons" +
          (" and training rechecks" if args.retrain else ""))


if __name__ == "__main__":
    main()
