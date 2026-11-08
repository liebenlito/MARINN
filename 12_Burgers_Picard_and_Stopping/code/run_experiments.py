from __future__ import annotations
import argparse, copy, math, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from marinn_core import *


def ensure(out):
    Path(out).mkdir(parents=True, exist_ok=True)
    return Path(out)


def init_model(seed, cfg):
    set_seed(seed)
    return MLP(cfg.hidden, cfg.depth)


def heat_checkpoint(seed, cfg, data):
    base = init_model(seed, cfg)
    pinn = clone_model(base, cfg)
    pinn, h1, m1 = train(pinn, data["train"], data["val"], residual_heat, cfg, "mse")
    refined = clone_model(pinn, cfg)
    refined, h2, m2 = train(
        refined, data["train"], data["val"], residual_heat, cfg, "marinn"
    )
    return base, pinn, refined, h1, h2, m1, m2


def run_heat_loss(out, cfg, seeds):
    rows = []
    histories = []
    for s in seeds:
        data = data_sets(s, cfg)
        base = init_model(s, cfg)
        for method, loss in [
            ("PINN-MSE", "mse"),
            ("MARINN", "marinn"),
            ("Flat-L4", "flat_l4"),
        ]:
            m = clone_model(base, cfg)
            m, h, meta = train(m, data["train"], data["val"], residual_heat, cfg, loss)
            dense = deterministic_grid(cfg.dense_n, torch.float64)
            # cast model for float64 evaluation
            md = copy.deepcopy(m).double()
            ev = evaluate(md, dense, copy.deepcopy(cfg) | {} if False else cfg, "heat")
            # evaluate() uses cfg.nu and model dtype; pack is float64, but cfg dtype irrelevant
            row = dict(
                seed=s,
                method=method,
                **ev,
                best_epoch=meta["best_epoch"],
                wall_seconds=meta["wall_seconds"],
            )
            rows.append(row)
            h["seed"] = s
            h["method"] = method
            histories.append(h)
    df = pd.DataFrame(rows)
    df.to_csv(out / "heat_loss_topology_raw.csv", index=False)
    pd.concat(histories, ignore_index=True).to_csv(
        out / "heat_loss_topology_histories.csv", index=False
    )
    return df


def run_q_sensitivity(out, cfg, seeds):
    rows = []
    for s in seeds:
        data = data_sets(s, cfg)
        base = init_model(s, cfg)
        for q in [2, 4, 8]:
            m = clone_model(base, cfg)
            m, h, meta = train(
                m,
                data["train"],
                data["val"],
                residual_heat,
                cfg,
                loss_name=f"q{q}",
                q=q,
            )
            md = copy.deepcopy(m).double()
            ev = evaluate(
                md, deterministic_grid(cfg.dense_n, torch.float64), cfg, "heat"
            )
            rows.append(dict(seed=s, q=q, **ev, best_epoch=meta["best_epoch"]))
    df = pd.DataFrame(rows)
    df.to_csv(out / "q_sensitivity_raw.csv", index=False)
    return df


def run_fair_budget(out, cfg, seeds):
    rows = []
    for s in seeds:
        data = data_sets(s, cfg)
        base = init_model(s, cfg)

        direct = clone_model(base, cfg)
        direct, _, meta = train(
            direct,
            data["train"],
            data["val"],
            residual_heat,
            cfg,
            "marinn",
            epochs=2 * cfg.epochs,
        )
        rows.append(
            dict(
                seed=s,
                route="Direct MARINN",
                **evaluate(
                    copy.deepcopy(direct).double(),
                    deterministic_grid(cfg.dense_n, torch.float64),
                    cfg,
                    "heat",
                ),
            )
        )

        pinn = clone_model(base, cfg)
        pinn, _, _ = train(pinn, data["train"], data["val"], residual_heat, cfg, "mse")
        p2m = clone_model(pinn, cfg)
        p2m, _, _ = train(p2m, data["train"], data["val"], residual_heat, cfg, "marinn")
        rows.append(
            dict(
                seed=s,
                route="PINN->MARINN",
                **evaluate(
                    copy.deepcopy(p2m).double(),
                    deterministic_grid(cfg.dense_n, torch.float64),
                    cfg,
                    "heat",
                ),
            )
        )

        p2p = clone_model(pinn, cfg)
        p2p, _, _ = train(p2p, data["train"], data["val"], residual_heat, cfg, "mse")
        rows.append(
            dict(
                seed=s,
                route="PINN->PINN",
                **evaluate(
                    copy.deepcopy(p2p).double(),
                    deterministic_grid(cfg.dense_n, torch.float64),
                    cfg,
                    "heat",
                ),
            )
        )
    df = pd.DataFrame(rows)
    df.to_csv(out / "fair_budget_raw.csv", index=False)
    return df


def run_heat_quadrature(out, cfg, seed=2026):
    data = data_sets(seed, cfg)
    base = init_model(seed, cfg)
    models = {}
    for name, loss in [("PINN", "mse"), ("MARINN", "marinn")]:
        m = clone_model(base, cfg)
        m, _, _ = train(m, data["train"], data["val"], residual_heat, cfg, loss)
        models[name] = m
    rows = []
    grids = [40, 80, 160, 320] if not cfg.dense_n < 320 else [20, 40, 80]
    for method, m in models.items():
        for n in grids:
            for prec in ["float32", "float64"]:
                mm = copy.deepcopy(m)
                dt = torch.float32 if prec == "float32" else torch.float64
                mm = mm.float() if prec == "float32" else mm.double()
                ev = evaluate(mm, deterministic_grid(n, dt), cfg, "heat")
                rows.append(dict(seed=seed, method=method, n=n, precision=prec, **ev))
    pd.DataFrame(rows).to_csv(out / "heat_quadrature_precision.csv", index=False)
    return pd.DataFrame(rows)


def run_smooth_burgers(out, cfg, seeds, wallclock=False):
    rows = []
    traces = []
    histories = []
    for s in seeds:
        data = data_sets(s, cfg)
        base, pinn, heat, _, _, _, _ = heat_checkpoint(s, cfg, data)

        # warm direct one-stage
        warm = clone_model(heat, cfg)
        warm, _, meta = train(
            warm, data["train"], data["val"], residual_burgers, cfg, "mse"
        )
        rows.append(
            dict(
                seed=s,
                method="Warm direct PINN 1200",
                **evaluate(
                    copy.deepcopy(warm).double(),
                    deterministic_grid(cfg.dense_n, torch.float64),
                    cfg,
                    "burgers",
                ),
                wall_seconds=meta["wall_seconds"],
            )
        )

        # MARINN Picard
        pic, tr, hh, pic_time = picard_solve(heat, data, cfg, "marinn")
        evp = evaluate(
            copy.deepcopy(pic).double(),
            deterministic_grid(cfg.dense_n, torch.float64),
            cfg,
            "burgers",
        )
        rows.append(
            dict(
                seed=s,
                method="MARINN-Picard",
                **evp,
                wall_seconds=pic_time,
                picard_steps=len(tr),
            )
        )
        tr["seed"] = s
        tr["method"] = "MARINN-Picard"
        traces.append(tr)
        if len(hh):
            hh["seed"] = s
            hh["method"] = "MARINN-Picard"
            histories.append(hh)

        # PINN Picard
        pp, trp, hhp, pp_time = picard_solve(heat, data, cfg, "mse")
        rows.append(
            dict(
                seed=s,
                method="PINN-Picard",
                **evaluate(
                    copy.deepcopy(pp).double(),
                    deterministic_grid(cfg.dense_n, torch.float64),
                    cfg,
                    "burgers",
                ),
                wall_seconds=pp_time,
                picard_steps=len(trp),
            )
        )
        trp["seed"] = s
        trp["method"] = "PINN-Picard"
        traces.append(trp)
        if len(hhp):
            hhp["seed"] = s
            hhp["method"] = "PINN-Picard"
            histories.append(hhp)

        # matched optimizer updates against MARINN-Picard stage count
        K = len(tr)
        cont = clone_model(heat, cfg)
        cont, _, metac = train(
            cont,
            data["train"],
            data["val"],
            residual_burgers,
            cfg,
            "mse",
            epochs=K * cfg.epochs,
        )
        rows.append(
            dict(
                seed=s,
                method="Warm direct matched steps continuous",
                **evaluate(
                    copy.deepcopy(cont).double(),
                    deterministic_grid(cfg.dense_n, torch.float64),
                    cfg,
                    "burgers",
                ),
                wall_seconds=metac["wall_seconds"],
                matched_stages=K,
            )
        )

        restarted = clone_model(heat, cfg)
        wall = 0.0
        for k in range(K):
            restarted, _, metar = train(
                restarted,
                data["train"],
                data["val"],
                residual_burgers,
                cfg,
                "mse",
                epochs=cfg.epochs,
            )
            wall += metar["wall_seconds"]
        rows.append(
            dict(
                seed=s,
                method="Warm direct matched steps restarted",
                **evaluate(
                    copy.deepcopy(restarted).double(),
                    deterministic_grid(cfg.dense_n, torch.float64),
                    cfg,
                    "burgers",
                ),
                wall_seconds=wall,
                matched_stages=K,
            )
        )

    df = pd.DataFrame(rows)
    df.to_csv(out / "smooth_burgers_all_controls.csv", index=False)
    pd.concat(traces, ignore_index=True).to_csv(
        out / "picard_stopping_trace.csv", index=False
    )
    if histories:
        pd.concat(histories, ignore_index=True).to_csv(
            out / "picard_inner_histories.csv", index=False
        )
    return df


def train_until_seconds(model, data, cfg, target_seconds):
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999), eps=1e-8)
    best = copy.deepcopy(model.state_dict())
    bestv = float("inf")
    ep = 0
    t0 = time.perf_counter()
    while True:
        ep += 1
        opt.zero_grad(set_to_none=True)
        r = residual_burgers(model, data["train"], cfg)
        loss = torch.mean(r * r)
        loss.backward()
        opt.step()
        if ep % cfg.val_every == 0:
            rv = residual_burgers(model, data["val"], cfg)
            vv = torch.mean(rv * rv).item()
            if vv < bestv:
                bestv = vv
                best = copy.deepcopy(model.state_dict())
            if time.perf_counter() - t0 >= target_seconds:
                break
    model.load_state_dict(best)
    return model, ep, time.perf_counter() - t0


def run_wallclock(out, cfg, seeds):
    rows = []
    for s in seeds:
        data = data_sets(s, cfg)
        _, _, heat, _, _, _, _ = heat_checkpoint(s, cfg, data)
        pic, tr, _, pic_time = picard_solve(
            heat, data, cfg, "marinn", record_history=False
        )
        direct = clone_model(heat, cfg)
        direct, steps, dtime = train_until_seconds(direct, data, cfg, pic_time)
        rows.append(
            dict(
                seed=s,
                method="MARINN-Picard",
                time_s=pic_time,
                steps=sum(tr.best_epoch.fillna(cfg.epochs)) if len(tr) else None,
                **evaluate(
                    copy.deepcopy(pic).double(),
                    deterministic_grid(cfg.dense_n, torch.float64),
                    cfg,
                    "burgers",
                ),
            )
        )
        rows.append(
            dict(
                seed=s,
                method="Direct wallclock matched",
                time_s=dtime,
                steps=steps,
                **evaluate(
                    copy.deepcopy(direct).double(),
                    deterministic_grid(cfg.dense_n, torch.float64),
                    cfg,
                    "burgers",
                ),
            )
        )
    df = pd.DataFrame(rows)
    df.to_csv(out / "strict_wallclock_reproduced.csv", index=False)
    return df


def run_viscosity(out, cfg, seeds):
    rows = []
    for nu in [0.10, 0.05, 0.02]:
        c = copy.deepcopy(cfg)
        c.nu = nu
        for s in seeds:
            data = data_sets(s, c)
            _, _, heat, _, _, _, _ = heat_checkpoint(s, c, data)
            pic, tr, _, wall = picard_solve(
                heat, data, c, "marinn", max_steps=c.picard_max, record_history=False
            )
            ev = evaluate(
                copy.deepcopy(pic).double(),
                deterministic_grid(c.dense_n, torch.float64),
                c,
                "burgers",
            )
            rows.append(
                dict(
                    seed=s,
                    nu=nu,
                    method="MARINN-Picard",
                    steps=len(tr),
                    wall_seconds=wall,
                    **ev,
                )
            )
    df = pd.DataFrame(rows)
    df.to_csv(out / "viscosity_reproduced.csv", index=False)
    return df


def run_moving_layer(out, cfg, seeds):
    c = copy.deepcopy(cfg)
    c.nu = 0.02
    rows = []
    for s in seeds:
        data = data_sets(s, c)
        base = init_model(s, c)
        for method, loss in [
            ("PINN-MSE", "mse"),
            ("MARINN", "marinn"),
            ("Flat-L4", "flat_l4"),
        ]:
            m = clone_model(base, c)
            # custom training loop because u0 differs
            opt = torch.optim.Adam(
                m.parameters(), lr=c.lr, betas=(0.9, 0.999), eps=1e-8
            )
            best = copy.deepcopy(m.state_dict())
            bestv = float("inf")
            t0 = time.perf_counter()
            for ep in range(1, c.epochs + 1):
                opt.zero_grad(set_to_none=True)
                r = residual_sharp(m, data["train"], c)
                obj = objective_from_residual(r, data["train"], loss)
                obj.backward()
                opt.step()
                if ep % c.val_every == 0:
                    rv = residual_sharp(m, data["val"], c)
                    vv = objective_from_residual(rv, data["val"], loss).item()
                    if vv < bestv:
                        bestv = vv
                        best = copy.deepcopy(m.state_dict())
            m.load_state_dict(best)
            rows.append(
                dict(
                    seed=s,
                    method=method,
                    wall_seconds=time.perf_counter() - t0,
                    **evaluate(
                        copy.deepcopy(m).double(),
                        deterministic_grid(c.dense_n, torch.float64),
                        c,
                        "sharp",
                    ),
                )
            )
    df = pd.DataFrame(rows)
    df.to_csv(out / "moving_layer_reproduced.csv", index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--experiment",
        default="all",
        choices=[
            "all",
            "heat_loss",
            "q_sensitivity",
            "fair_budget",
            "quadrature",
            "burgers",
            "wallclock",
            "viscosity",
            "moving_layer",
        ],
    )
    ap.add_argument("--output", default="outputs")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--save-checkpoints", action="store_true")
    args = ap.parse_args()
    cfg = Config()
    if args.quick:
        cfg = quick_cfg(cfg)
    torch.set_num_threads(cfg.threads)
    out = ensure(args.output)
    save_config(cfg, out)
    seeds = args.seeds or (SEEDS[:1] if args.quick else SEEDS)

    tasks = {
        "heat_loss": lambda: run_heat_loss(out, cfg, seeds),
        "q_sensitivity": lambda: run_q_sensitivity(out, cfg, seeds),
        "fair_budget": lambda: run_fair_budget(out, cfg, seeds),
        "quadrature": lambda: run_heat_quadrature(out, cfg, seeds[0]),
        "burgers": lambda: run_smooth_burgers(out, cfg, seeds),
        "wallclock": lambda: run_wallclock(out, cfg, seeds),
        "viscosity": lambda: run_viscosity(out, cfg, seeds),
        "moving_layer": lambda: run_moving_layer(out, cfg, seeds),
    }
    selected = list(tasks) if args.experiment == "all" else [args.experiment]
    for name in selected:
        print(f"\\n=== {name} ===", flush=True)
        t = time.perf_counter()
        tasks[name]()
        print(f"completed {name} in {time.perf_counter()-t:.1f} s", flush=True)


if __name__ == "__main__":
    main()
