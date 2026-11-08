from __future__ import annotations
import sys, copy, time, gc, math, json
from pathlib import Path
import numpy as np, pandas as pd, torch

sys.path.insert(0, "/mnt/data/marinn_repro_extract/MARINN_REPRODUCIBILITY_BUNDLE/code")
from marinn_core import *

OUT = Path("/mnt/data/MARINN_FAIRNESS_9_10_11")
OUT.mkdir(exist_ok=True)


def nonlinear_train(model, data, cfg, loss_name, epochs):
    return train(
        model,
        data["train"],
        data["val"],
        residual_burgers,
        cfg,
        loss_name=loss_name,
        epochs=epochs,
        record=False,
    )


def fixedK_picard(init_model, data, cfg, loss_name, K, inner_epochs):
    current = clone_model(init_model, cfg)
    stage_rows = []
    t0 = time.perf_counter()
    for k in range(K):
        prev = clone_model(current, cfg)
        Ftr = frozen_forcing(prev, data["train"], cfg)
        Fva = frozen_forcing(prev, data["val"], cfg)
        cand = clone_model(prev, cfg)
        cand, _, meta = train(
            cand,
            data["train"],
            data["val"],
            residual_burgers,
            cfg,
            loss_name=loss_name,
            epochs=inner_epochs,
            frozen_train=Ftr,
            frozen_val=Fva,
            record=False,
        )
        current = cand
        d = picard_delta(current, prev, data["mon"])
        rho = evaluate(current, data["mon"], cfg, "burgers")["residual_L2tL4"]
        stage_rows.append(
            dict(stage=k + 1, delta=d, rho=rho, stage_wall=meta["wall_seconds"])
        )
    return current, pd.DataFrame(stage_rows), time.perf_counter() - t0


def warmup(cfg, data, init):
    m = clone_model(init, cfg)
    # one full AD forward/backward step, excluded from timings
    opt = torch.optim.Adam(m.parameters(), lr=cfg.lr)
    opt.zero_grad(set_to_none=True)
    r = residual_burgers(m, data["train"], cfg)
    torch.mean(r * r).backward()
    opt.step()
    del m, opt, r
    gc.collect()


def eval64(model, cfg, n=80):
    return evaluate(
        copy.deepcopy(model).double(),
        deterministic_grid(n, torch.float64),
        cfg,
        "burgers",
    )


def run_seed(seed, repeats=3, K=3, inner_epochs=400):
    cfg = Config()
    cfg.epochs = inner_epochs
    cfg.val_every = 50
    cfg.nx_train = 20
    cfg.nt_train = 25
    cfg.nx_eval = 12
    cfg.nt_eval = 15
    cfg.picard_max = K
    cfg.threads = 1
    torch.set_num_threads(cfg.threads)
    data = data_sets(seed, cfg)
    set_seed(seed)
    base = MLP(cfg.hidden, cfg.depth)
    # common refined heat checkpoint, excluded from nonlinear timing
    heat = clone_model(base, cfg)
    heat, _, _ = train(
        heat,
        data["train"],
        data["val"],
        residual_heat,
        cfg,
        "mse",
        epochs=inner_epochs,
        record=False,
    )
    heat, _, _ = train(
        heat,
        data["train"],
        data["val"],
        residual_heat,
        cfg,
        "marinn",
        epochs=inner_epochs,
        record=False,
    )
    warmup(cfg, data, heat)

    methods = ["Direct-MSE", "Direct-MARINN", "PINN-Picard", "MARINN-Picard"]
    rows = []
    traces = []
    # balanced deterministic Latin rotation; avoids always timing Picard first.
    for rep in range(repeats):
        order = methods[rep:] + methods[:rep]
        for orderpos, method in enumerate(order):
            gc.collect()
            if method == "Direct-MSE":
                m = clone_model(heat, cfg)
                t0 = time.perf_counter()
                m, _, meta = nonlinear_train(m, data, cfg, "mse", K * inner_epochs)
                wall = time.perf_counter() - t0
                updates = K * inner_epochs
            elif method == "Direct-MARINN":
                m = clone_model(heat, cfg)
                t0 = time.perf_counter()
                m, _, meta = nonlinear_train(m, data, cfg, "marinn", K * inner_epochs)
                wall = time.perf_counter() - t0
                updates = K * inner_epochs
            elif method == "PINN-Picard":
                m, tr, wall = fixedK_picard(heat, data, cfg, "mse", K, inner_epochs)
                tr["seed"] = seed
                tr["repeat"] = rep
                tr["method"] = method
                traces.append(tr)
                updates = K * inner_epochs
            else:
                m, tr, wall = fixedK_picard(heat, data, cfg, "marinn", K, inner_epochs)
                tr["seed"] = seed
                tr["repeat"] = rep
                tr["method"] = method
                traces.append(tr)
                updates = K * inner_epochs
            ev = eval64(m, cfg, 80)
            rows.append(
                dict(
                    seed=seed,
                    repeat=rep,
                    order_position=orderpos,
                    method=method,
                    wall_seconds=wall,
                    adam_updates=updates,
                    **ev,
                )
            )
            del m
            gc.collect()
    return pd.DataFrame(rows), (
        pd.concat(traces, ignore_index=True) if traces else pd.DataFrame()
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--inner-epochs", type=int, default=400)
    args = ap.parse_args()
    df, tr = run_seed(args.seed, args.repeats, args.K, args.inner_epochs)
    df.to_csv(OUT / f"fair2x2_seed{args.seed}.csv", index=False)
    tr.to_csv(OUT / f"fair2x2_picard_trace_seed{args.seed}.csv", index=False)
    print(df.to_string(index=False))
