from __future__ import annotations
import argparse, copy, json, math, time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

Tensor = torch.Tensor

SEEDS = [2024, 2025, 2026, 2027, 2028]


@dataclass
class Config:
    nu: float = 0.1
    alpha: float = 0.25
    hidden: int = 32
    depth: int = 3
    lr: float = 2e-3
    epochs: int = 1200
    val_every: int = 50
    nx_train: int = 40
    nt_train: int = 50
    nx_eval: int = 20
    nt_eval: int = 25
    dense_n: int = 320
    picard_tol: float = 1e-4
    picard_max: int = 20
    dtype: str = "float32"
    threads: int = 5

    @property
    def torch_dtype(self):
        return torch.float32 if self.dtype == "float32" else torch.float64


class MLP(torch.nn.Module):
    def __init__(self, hidden=32, depth=3):
        super().__init__()
        layers = [torch.nn.Linear(2, hidden), torch.nn.Tanh()]
        for _ in range(depth - 1):
            layers += [torch.nn.Linear(hidden, hidden), torch.nn.Tanh()]
        layers += [torch.nn.Linear(hidden, 1)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)


def clone_model(m: torch.nn.Module, cfg: Config) -> torch.nn.Module:
    n = MLP(cfg.hidden, cfg.depth)
    n.load_state_dict(copy.deepcopy(m.state_dict()))
    return n


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def jittered_tensor_grid(seed: int, nx: int, nt: int, dtype=torch.float32):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    xv = (torch.arange(nx, dtype=dtype) + torch.rand(nx, generator=g, dtype=dtype)) / nx
    tv = (torch.arange(nt, dtype=dtype) + torch.rand(nt, generator=g, dtype=dtype)) / nt
    tt, xx = torch.meshgrid(tv, xv, indexing="ij")
    return xx.reshape(-1, 1), tt.reshape(-1, 1), nx, nt


def deterministic_grid(n: int, dtype=torch.float64):
    xv = torch.linspace(0, 1, n, dtype=dtype)
    tv = torch.linspace(0, 1, n, dtype=dtype)
    tt, xx = torch.meshgrid(tv, xv, indexing="ij")
    return xx.reshape(-1, 1), tt.reshape(-1, 1), n, n


def data_sets(seed: int, cfg: Config):
    dt = cfg.torch_dtype
    return {
        "train": jittered_tensor_grid(10 * seed + 1, cfg.nx_train, cfg.nt_train, dt),
        "val": jittered_tensor_grid(10 * seed + 2, cfg.nx_eval, cfg.nt_eval, dt),
        "test": jittered_tensor_grid(10 * seed + 3, cfg.nx_eval, cfg.nt_eval, dt),
        "mon": jittered_tensor_grid(10 * seed + 4, cfg.nx_eval, cfg.nt_eval, dt),
    }


def heat_u0(x):
    return torch.sin(math.pi * x)


def sharp_u0(x, delta=0.05):
    return x * (1 - x) * torch.tanh((x - 0.35) / delta)


def hard_field(model, x, t, u0: Callable[[Tensor], Tensor]):
    return u0(x) + t * x * (1 - x) * model(torch.cat([x, t], dim=1))


def heat_exact(x, t):
    return torch.exp(-t) * torch.sin(math.pi * x)


def heat_forcing(x, t, nu):
    return (-1 + nu * math.pi**2) * torch.exp(-t) * torch.sin(math.pi * x)


def burgers_exact(x, t, alpha=0.25):
    return torch.exp(-t) * torch.sin(math.pi * x) + alpha * t * torch.exp(
        -t
    ) * torch.sin(2 * math.pi * x)


def burgers_forcing(x, t, nu, alpha=0.25):
    u = burgers_exact(x, t, alpha)
    ut = -torch.exp(-t) * torch.sin(math.pi * x) + alpha * (1 - t) * torch.exp(
        -t
    ) * torch.sin(2 * math.pi * x)
    ux = math.pi * torch.exp(-t) * torch.cos(
        math.pi * x
    ) + 2 * math.pi * alpha * t * torch.exp(-t) * torch.cos(2 * math.pi * x)
    uxx = -math.pi**2 * torch.exp(-t) * torch.sin(
        math.pi * x
    ) - 4 * math.pi**2 * alpha * t * torch.exp(-t) * torch.sin(2 * math.pi * x)
    return ut + u * ux - nu * uxx


def sharp_exact(x, t, delta=0.05):
    c = 0.35 + 0.30 * t
    z = (x - c) / delta
    return x * (1 - x) * torch.tanh(z)


def sharp_forcing(x, t, nu=0.02, delta=0.05):
    c = 0.35 + 0.30 * t
    z = (x - c) / delta
    th = torch.tanh(z)
    sech2 = 1 - th**2
    h = x * (1 - x)
    hp = 1 - 2 * x
    ut = h * sech2 * (-0.30 / delta)
    ux = hp * th + h * sech2 / delta
    uxx = -2 * th + 2 * hp * sech2 / delta - 2 * h * sech2 * th / (delta**2)
    u = h * th
    return ut + u * ux - nu * uxx


def derivatives(model, pack, u0):
    x, t, nx, nt = pack
    x = x.detach().clone().requires_grad_(True)
    t = t.detach().clone().requires_grad_(True)
    u = hard_field(model, x, t, u0)
    ut = torch.autograd.grad(
        u, t, torch.ones_like(u), create_graph=True, retain_graph=True
    )[0]
    ux = torch.autograd.grad(
        u, x, torch.ones_like(u), create_graph=True, retain_graph=True
    )[0]
    uxx = torch.autograd.grad(
        ux, x, torch.ones_like(ux), create_graph=True, retain_graph=True
    )[0]
    return u, ut, ux, uxx, x, t


def mixed_objective(r: Tensor, pack, q=4):
    _, _, nx, nt = pack
    R = r.reshape(nt, nx)
    return torch.mean(torch.mean(torch.abs(R) ** q, dim=1) ** (2.0 / q))


def mixed_norm(r: Tensor, pack, q=4):
    return torch.sqrt(mixed_objective(r, pack, q=q))


def flat_l4_objective(r: Tensor):
    """Degree-two, homogeneity-matched flat L4 residual objective."""
    return torch.mean(torch.abs(r) ** 4) ** 0.5


def residual_heat(model, pack, cfg):
    u, ut, ux, uxx, x, t = derivatives(model, pack, heat_u0)
    return ut - cfg.nu * uxx - heat_forcing(x, t, cfg.nu)


def residual_burgers(model, pack, cfg):
    u, ut, ux, uxx, x, t = derivatives(model, pack, heat_u0)
    return ut + u * ux - cfg.nu * uxx - burgers_forcing(x, t, cfg.nu, cfg.alpha)


def residual_sharp(model, pack, cfg):
    u0 = lambda x: sharp_u0(x, delta=0.05)
    u, ut, ux, uxx, x, t = derivatives(model, pack, u0)
    return ut + u * ux - cfg.nu * uxx - sharp_forcing(x, t, cfg.nu, 0.05)


def objective_from_residual(r, pack, loss_name="mse", q=4):
    if loss_name == "mse":
        return torch.mean(r * r)
    if loss_name == "marinn":
        return mixed_objective(r, pack, q=q)
    if loss_name == "flat_l4":
        return flat_l4_objective(r)
    if loss_name.startswith("q"):
        qq = int(loss_name[1:])
        return mixed_objective(r, pack, q=qq)
    raise ValueError(loss_name)


def train(
    model,
    train_pack,
    val_pack,
    residual_fn,
    cfg: Config,
    loss_name="mse",
    epochs=None,
    q=4,
    record=True,
    frozen_train=None,
    frozen_val=None,
):
    epochs = epochs or cfg.epochs
    opt = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0
    )
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = 0
    hist = []
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        opt.zero_grad(set_to_none=True)
        if frozen_train is None:
            r = residual_fn(model, train_pack, cfg)
        else:
            u, ut, ux, uxx, x, t = derivatives(model, train_pack, heat_u0)
            r = ut - cfg.nu * uxx - frozen_train
        loss = objective_from_residual(r, train_pack, loss_name, q=q)
        loss.backward()
        opt.step()

        if ep % cfg.val_every == 0 or ep == epochs:
            if frozen_val is None:
                rv = residual_fn(model, val_pack, cfg)
            else:
                u, ut, ux, uxx, x, t = derivatives(model, val_pack, heat_u0)
                rv = ut - cfg.nu * uxx - frozen_val
            vv = objective_from_residual(rv, val_pack, loss_name, q=q).detach().item()
            trv = loss.detach().item()
            if record:
                hist.append(dict(epoch=ep, train_objective=trv, val_objective=vv))
            if vv < best_val:
                best_val = vv
                best_epoch = ep
                best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return (
        model,
        pd.DataFrame(hist),
        dict(
            best_epoch=best_epoch,
            best_val=best_val,
            wall_seconds=time.perf_counter() - t0,
        ),
    )


def exact_derivatives_heat(x, t):
    ut = -torch.exp(-t) * torch.sin(math.pi * x)
    uxx = -math.pi**2 * torch.exp(-t) * torch.sin(math.pi * x)
    return ut, uxx


def exact_derivatives_burgers(x, t, alpha=0.25):
    ut = -torch.exp(-t) * torch.sin(math.pi * x) + alpha * (1 - t) * torch.exp(
        -t
    ) * torch.sin(2 * math.pi * x)
    uxx = -math.pi**2 * torch.exp(-t) * torch.sin(
        math.pi * x
    ) - 4 * math.pi**2 * alpha * t * torch.exp(-t) * torch.sin(2 * math.pi * x)
    return ut, uxx


def exact_derivatives_sharp(x, t, delta=0.05):
    c = 0.35 + 0.30 * t
    z = (x - c) / delta
    th = torch.tanh(z)
    sech2 = 1 - th**2
    h = x * (1 - x)
    hp = 1 - 2 * x
    ut = h * sech2 * (-0.30 / delta)
    uxx = -2 * th + 2 * hp * sech2 / delta - 2 * h * sech2 * th / (delta**2)
    return ut, uxx


def evaluate(model, pack, cfg, problem="heat"):
    if problem == "heat":
        u0 = heat_u0
        exact = heat_exact
        residual_fn = residual_heat
        dex = exact_derivatives_heat
    elif problem == "burgers":
        u0 = heat_u0
        exact = lambda x, t: burgers_exact(x, t, cfg.alpha)
        residual_fn = residual_burgers
        dex = lambda x, t: exact_derivatives_burgers(x, t, cfg.alpha)
    elif problem == "sharp":
        u0 = lambda x: sharp_u0(x, 0.05)
        exact = lambda x, t: sharp_exact(x, t, 0.05)
        residual_fn = residual_sharp
        dex = lambda x, t: exact_derivatives_sharp(x, t, 0.05)
    else:
        raise ValueError(problem)
    u, ut, ux, uxx, x, t = derivatives(model, pack, u0)
    ue = exact(x, t)
    rel = (
        torch.linalg.vector_norm((u - ue).detach())
        / torch.linalg.vector_norm(ue.detach())
    ).item()
    rr = residual_fn(model, pack, cfg)
    eta = mixed_norm(rr, pack, 4).detach().item()
    uet, uexx = dex(x, t)
    et = ut - uet
    exx = uxx - uexx
    n_et = mixed_norm(et, pack, 4).detach().item()
    n_exx = mixed_norm(exx, pack, 4).detach().item()
    emr = n_et + cfg.nu * n_exx
    return dict(
        rel_L2_exact=rel,
        residual_L2tL4=eta,
        e_t_L2tL4=n_et,
        e_xx_L2tL4=n_exx,
        E_MR=emr,
        max_residual=rr.detach().abs().max().item(),
    )


def frozen_forcing(prev, pack, cfg):
    u, ut, ux, uxx, x, t = derivatives(prev, pack, heat_u0)
    return (
        burgers_forcing(x, t, cfg.nu, cfg.alpha) - u.detach() * ux.detach()
    ).detach()


def picard_delta(a, b, pack):
    x, t, _, _ = pack
    with torch.no_grad():
        ua = hard_field(a, x, t, heat_u0)
        ub = hard_field(b, x, t, heat_u0)
    return (torch.linalg.vector_norm(ua - ub) / torch.linalg.vector_norm(ua)).item()


def picard_solve(
    init_model, data, cfg, inner_loss="marinn", max_steps=None, record_history=True
):
    max_steps = max_steps or cfg.picard_max
    current = clone_model(init_model, cfg)
    trace = []
    histories = []
    t0 = time.perf_counter()
    for k in range(max_steps):
        prev = clone_model(current, cfg)
        Ftr = frozen_forcing(prev, data["train"], cfg)
        Fva = frozen_forcing(prev, data["val"], cfg)
        cand = clone_model(prev, cfg)
        cand, hist, meta = train(
            cand,
            data["train"],
            data["val"],
            residual_burgers,
            cfg,
            loss_name=inner_loss,
            frozen_train=Ftr,
            frozen_val=Fva,
            record=record_history,
        )
        current = cand
        d = picard_delta(current, prev, data["mon"])
        rho = evaluate(current, data["mon"], cfg, "burgers")["residual_L2tL4"]
        trace.append(
            dict(
                step=k + 1,
                delta=d,
                nonlinear_residual=rho,
                best_epoch=meta["best_epoch"],
                stage_wall_seconds=meta["wall_seconds"],
            )
        )
        if record_history:
            hist["picard_step"] = k + 1
            histories.append(hist)
        if d < cfg.picard_tol:
            break
    return (
        current,
        pd.DataFrame(trace),
        pd.concat(histories, ignore_index=True) if histories else pd.DataFrame(),
        time.perf_counter() - t0,
    )


def save_config(cfg, outdir):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    (Path(outdir) / "config.json").write_text(json.dumps(asdict(cfg), indent=2))


def quick_cfg(cfg: Config):
    c = copy.deepcopy(cfg)
    c.epochs = 100
    c.val_every = 25
    c.nx_train = 20
    c.nt_train = 25
    c.nx_eval = 12
    c.nt_eval = 15
    c.dense_n = 80
    c.picard_max = 3
    return c
