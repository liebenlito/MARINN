#!/usr/bin/env python3
"""MARINN: operator probes and residual-only model selection.
Usage:
 python marinn_analysis.py operator --out operator_probe.csv
 python marinn_analysis.py select --input model_residuals.csv --out model_selection_summary.csv
Input CSV: family,model_id,E_MR,eta_MR,eta_2,eta_4 (one row per independent model).
"""

import argparse, csv, math
import numpy as np
from scipy.integrate import trapezoid
from scipy.stats import spearmanr


def norm_mixed(a, x, t, q=4, p=2):
    return float(trapezoid(trapezoid(abs(a) ** q, x, axis=1) ** (p / q), t) ** (1 / p))


def probe(out):
    x = np.linspace(0, 1, 1025)
    t = np.linspace(0, 1, 1025)
    X = x[None, :]
    T = t[:, None]
    nu = 0.1
    rows = []
    # Smooth zero-initial, zero-boundary trials; no neural approximations required.
    for n in [1, 2, 3, 4]:
        for k in [1, 2, 4, 8]:
            phi = np.sin(n * np.pi * X)
            lam = nu * (n * np.pi) ** 2
            zt = k * np.exp(-k * T) * phi
            Az = lam * (1 - np.exp(-k * T)) * phi
            residual = zt + Az
            for q in [2, 4]:
                num = norm_mixed(zt, x, t, q) + norm_mixed(Az, x, t, q)
                den = norm_mixed(residual, x, t, q)
                rows.append(dict(mode=n, temporal_rate=k, q=q, ratio=num / den))
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)
    for q in [2, 4]:
        vals = [r["ratio"] for r in rows if r["q"] == q]
        print(
            f"q={q}: maximum tested ratio {max(vals):.8f} across {len(vals)} smooth forcing probes (NOT a certified operator upper bound)"
        )
    print(
        "q=2: rigorously C_MR <= sqrt(2) for the Hilbert-space graph norm with zero initial data."
    )
    print("Written", out)


def select(inp, out):
    with open(inp, newline="") as fh:
        rows = list(csv.DictReader(fh))
    required = {"family", "model_id", "E_MR", "eta_MR", "eta_2", "eta_4"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Required columns: " + ",".join(sorted(required)))
    groups = {}
    for r in rows:
        d = {k: float(r[k]) for k in ["E_MR", "eta_MR", "eta_2", "eta_4"]}
        if any(not math.isfinite(v) or v < 0 for v in d.values()):
            raise ValueError("Non-finite or negative norm")
        groups.setdefault(r["family"], []).append((r["model_id"], d))
    summary = []
    for fam, candidates in sorted(groups.items()):
        if len(candidates) < 3:
            print("Caution: too few independent models for family", fam)
        best = min(candidates, key=lambda o: (o[1]["E_MR"], o[0]))
        for ind in ["eta_MR", "eta_2", "eta_4"]:
            chosen = min(candidates, key=lambda o: (o[1][ind], o[0]))
            rho = (
                spearmanr(
                    [a[1][ind] for a in candidates], [a[1]["E_MR"] for a in candidates]
                ).statistic
                if len(candidates) > 2
                else float("nan")
            )
            reg = chosen[1]["E_MR"] - best[1]["E_MR"]
            summary.append(
                dict(
                    family=fam,
                    n_models=len(candidates),
                    indicator=ind,
                    selected_id=chosen[0],
                    oracle_id=best[0],
                    success=int(np.isclose(reg, 0, rtol=1e-12, atol=1e-14)),
                    absolute_regret=reg,
                    relative_regret=(
                        reg / best[1]["E_MR"] if best[1]["E_MR"] > 0 else float("nan")
                    ),
                    spearman=rho,
                )
            )
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=summary[0])
        w.writeheader()
        w.writerows(summary)
    for s in summary:
        print(s)
    print("Written", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("operator")
    p.add_argument("--out", default="operator_probe.csv")
    p = sp.add_parser("select")
    p.add_argument("--input", required=True)
    p.add_argument("--out", default="model_selection_summary.csv")
    a = ap.parse_args()
    probe(a.out) if a.cmd == "operator" else select(a.input, a.out)
