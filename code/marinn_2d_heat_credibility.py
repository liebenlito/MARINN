import math, time, copy, os, json
from pathlib import Path
import numpy as np, pandas as pd, torch

torch.set_num_threads(int(os.environ.get("MARINN_THREADS", "2")))
DTYPE = torch.float32
WIDTH = 48
EPOCHS = 500
LR = 2e-3
OUT = Path("/mnt/data/MARINN_2D_HEAT_CREDIBILITY")
OUT.mkdir(exist_ok=True)


def set_seed(s):
    torch.manual_seed(s)
    np.random.seed(s)


class MLP(torch.nn.Module):
    def __init__(self, width=WIDTH):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, width),
            torch.nn.Tanh(),
            torch.nn.Linear(width, width),
            torch.nn.Tanh(),
            torch.nn.Linear(width, width),
            torch.nn.Tanh(),
            torch.nn.Linear(width, 1),
        )
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, z):
        return self.net(z)


def clone(m):
    n = MLP()
    n.load_state_dict(copy.deepcopy(m.state_dict()))
    return n


def tensor_grid(nx, ny, nt, T=1.0, seed=None, dtype=DTYPE):
    if seed is None:
        x = (torch.arange(nx, dtype=dtype) + 0.5) / nx
        y = (torch.arange(ny, dtype=dtype) + 0.5) / ny
        t = T * (torch.arange(nt, dtype=dtype) + 0.5) / nt
    else:
        g = torch.Generator()
        g.manual_seed(int(seed))
        x = (torch.arange(nx, dtype=dtype) + torch.rand(nx, generator=g, dtype=dtype)) / nx
        y = (torch.arange(ny, dtype=dtype) + torch.rand(ny, generator=g, dtype=dtype)) / ny
        t = (T* (torch.arange(nt, dtype=dtype) + torch.rand(nt, generator=g, dtype=dtype)) / nt)
    TT, YY, XX = torch.meshgrid(t, y, x, indexing="ij")
    return XX.reshape(-1, 1), YY.reshape(-1, 1), TT.reshape(-1, 1), (nx, ny, nt, T)


def u0(x, y):
    return torch.sin(math.pi * x) * torch.sin(math.pi * y)


def hard_field(m, x, y, t):
    # Normalized boundary factor max_{[0,1]^2} 16*x(1-x)y(1-y)=1.
    return u0(x, y) + 16.0 * t * x * (1 - x) * y * (1 - y) * m(torch.cat([x, y, t], 1))


def derivs(m, pack):
    x, y, t, shape = pack
    x = x.detach().clone().requires_grad_(True)
    y = y.detach().clone().requires_grad_(True)
    t = t.detach().clone().requires_grad_(True)
    u = hard_field(m, x, y, t)
    one = torch.ones_like(u)
    ut = torch.autograd.grad(u, t, one, create_graph=True, retain_graph=True)[0]
    ux = torch.autograd.grad(u, x, one, create_graph=True, retain_graph=True)[0]
    uy = torch.autograd.grad(u, y, one, create_graph=True, retain_graph=True)[0]
    uxx = torch.autograd.grad(ux, x, one, create_graph=True, retain_graph=True)[0]
    uyy = torch.autograd.grad(uy, y, one, create_graph=True, retain_graph=True)[0]
    return u, ut, ux, uy, uxx, uyy, x, y, t


def residual(m, pack, nu=0.1):
    u, ut, ux, uy, uxx, uyy, x, y, t = derivs(m, pack)
    f = (
        (-1 + 2 * nu * math.pi**2)
        * torch.exp(-t)
        * torch.sin(math.pi * x)
        * torch.sin(math.pi * y)
    )
    return ut - nu * (uxx + uyy) - f


def mixed_sq(r, pack):
    nx, ny, nt, T = pack[3]
    R = r.reshape(nt, ny, nx)
    return T * torch.mean(torch.mean(torch.abs(R) ** 4, dim=(1, 2)) ** 0.5)


def mixed_norm(r, pack):
    return torch.sqrt(mixed_sq(r, pack))


def objective(r, pack, kind):
    if kind == "mse":
        return torch.mean(r * r)
    if kind == "flat_l4":
        return torch.mean(torch.abs(r) ** 4) ** 0.5
    if kind == "marinn":
        # T=1 here, so normalized and continuum forms coincide.
        nx, ny, nt, T = pack[3]
        R = r.reshape(nt, ny, nx)
        return torch.mean(torch.mean(torch.abs(R) ** 4, dim=(1, 2)) ** 0.5)
    raise ValueError(kind)


def train(model, tr, va, kind, epochs=EPOCHS, lr=LR):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best = copy.deepcopy(model.state_dict())
    bestv = float("inf")
    hist = []
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        opt.zero_grad(set_to_none=True)
        r = residual(model, tr)
        loss = objective(r, tr, kind)
        loss.backward()
        opt.step()
        if ep == 1 or ep % 50 == 0:
            rv = residual(model, va)
            v = float(objective(rv, va, kind).detach())
            hist.append((ep, float(loss.detach()), v))
            if v < bestv:
                bestv = v
                best = copy.deepcopy(model.state_dict())
    model.load_state_dict(best)
    return (
        pd.DataFrame(hist, columns=["epoch", "train_loss", "val_loss"]),
        time.perf_counter() - t0,
        bestv,
    )


def eval_dense(model, n=36, nt=21, nu=0.1):
    xv = torch.linspace(1 / (n + 1), n / (n + 1), n, dtype=torch.float64)
    tv = torch.linspace(0, 1, nt, dtype=torch.float64)
    TT, YY, XX = torch.meshgrid(tv, xv, xv, indexing="ij")
    x = XX.reshape(-1, 1)
    y = YY.reshape(-1, 1)
    t = TT.reshape(-1, 1)
    md = copy.deepcopy(model).double()
    u, ut, ux, uy, uxx, uyy, x, y, t = derivs(md, (x, y, t, (n, n, nt, 1.0)))
    ue = torch.exp(-t) * torch.sin(math.pi * x) * torch.sin(math.pi * y)
    uet = -ue
    lape = -2 * math.pi**2 * ue
    e = u - ue
    et = ut - uet
    laperr = uxx + uyy - lape
    r = ut - nu * (uxx + uyy) - (-1 + 2 * nu * math.pi**2) * ue
    rel = (
        torch.linalg.vector_norm(e.detach()) / torch.linalg.vector_norm(ue.detach())
    ).item()
    eta = mixed_norm(r, (x, y, t, (n, n, nt, 1.0))).detach().item()
    E = (
        mixed_norm(et, (x, y, t, (n, n, nt, 1.0))).detach().item()
        + nu * mixed_norm(laperr, (x, y, t, (n, n, nt, 1.0))).detach().item()
    )
    return dict(
        rel_L2=rel, eta_MR=eta, E_MR=E, max_residual=r.detach().abs().max().item()
    )


def eval_dense_batched(model, n, nt, nu=0.1):
    # Per-time-slice AD keeps memory bounded, while quadrature is accumulated exactly.
    md = copy.deepcopy(model).double()
    xv = torch.linspace(1 / (n + 1), n / (n + 1), n, dtype=torch.float64)
    tv = torch.linspace(0, 1, nt, dtype=torch.float64)
    sum_e2 = sum_u2 = 0.0
    sum_r_l4sq = sum_et_l4sq = sum_lap_l4sq = 0.0
    maxr = 0.0
    for tt in tv:
        YY, XX = torch.meshgrid(xv, xv, indexing="ij")
        x = XX.reshape(-1, 1)
        y = YY.reshape(-1, 1)
        t = torch.full_like(x, float(tt))
        u, ut, ux, uy, uxx, uyy, x, y, t = derivs(md, (x, y, t, (n, n, 1, 1.0)))
        ue = torch.exp(-t) * torch.sin(math.pi * x) * torch.sin(math.pi * y)
        uet = -ue
        lape = -2 * math.pi**2 * ue
        e = (u - ue).detach()
        et = (ut - uet).detach()
        lap = (uxx + uyy - lape).detach()
        r = (ut - nu * (uxx + uyy) - (-1 + 2 * nu * math.pi**2) * ue).detach()
        sum_e2 += float(torch.sum(e * e))
        sum_u2 += float(torch.sum(ue.detach() ** 2))
        sum_r_l4sq += float(torch.mean(torch.abs(r) ** 4) ** 0.5)
        sum_et_l4sq += float(torch.mean(torch.abs(et) ** 4) ** 0.5)
        sum_lap_l4sq += float(torch.mean(torch.abs(lap) ** 4) ** 0.5)
        maxr = max(maxr, float(torch.max(torch.abs(r))))
    rel = math.sqrt(sum_e2 / sum_u2)
    eta = math.sqrt(sum_r_l4sq / nt)
    etn = math.sqrt(sum_et_l4sq / nt)
    lapn = math.sqrt(sum_lap_l4sq / nt)
    return dict(rel_L2=rel, eta_MR=eta, E_MR=etn + nu * lapn, max_residual=maxr)


def run_main5():
    rows = []
    for seed in [2024, 2025, 2026, 2027, 2028]:
        set_seed(seed)
        tr = tensor_grid(10, 10, 12, seed=100 * seed + 1)
        va = tensor_grid(8, 8, 10, seed=100 * seed + 2)
        base = MLP()
        sd = OUT / f"main5_seed{seed}"
        sd.mkdir(exist_ok=True)
        for name, kind in [
            ("PINN-MSE", "mse"),
            ("Flat-L4-squared", "flat_l4"),
            ("MARINN", "marinn"),
        ]:
            model = clone(base)
            hist, wall, bv = train(model, tr, va, kind)
            met = eval_dense(model, 36, 21)
            rows.append(
                dict(seed=seed, method=name, wall_seconds=wall, best_val=bv, **met)
            )
            hist.to_csv(sd / f"{name}_history.csv", index=False)
            torch.save(model.state_dict(), sd / f"{name}.pt")
            print("MAIN", seed, name, met, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "heat2d_conditioned_main5_raw.csv", index=False)
    summ = df.groupby("method").agg(
        {
            c: ["mean", "std"]
            for c in ["rel_L2", "eta_MR", "E_MR", "max_residual", "wall_seconds"]
        }
    )
    summ.to_csv(OUT / "heat2d_conditioned_main5_summary.csv")


def run_collocation():
    rows = []
    levels = [(10, 10, 12), (14, 14, 16), (18, 18, 20)]
    for seed in [2024, 2026, 2028]:
        for nx, ny, nt in levels:
            set_seed(seed)
            tr = tensor_grid(nx, ny, nt, seed=1000 * seed + nx)
            va = tensor_grid(10, 10, 12, seed=2000 * seed + nx)
            base = MLP()
            for name, kind in [("PINN-MSE", "mse"), ("MARINN", "marinn")]:
                model = clone(base)
                hist, wall, bv = train(model, tr, va, kind)
                met = eval_dense(model, 36, 21)
                rows.append(
                    dict(
                        seed=seed,
                        method=name,
                        nx=nx,
                        ny=ny,
                        nt=nt,
                        ntrain=nx * ny * nt,
                        wall_seconds=wall,
                        best_val=bv,
                        **met,
                    )
                )
                print("COLL", seed, nx, name, met, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "heat2d_collocation_sensitivity_raw.csv", index=False)
    df.groupby(["method", "ntrain"]).agg(
        {c: ["mean", "std"] for c in ["rel_L2", "eta_MR", "E_MR", "max_residual"]}
    ).to_csv(OUT / "heat2d_collocation_sensitivity_summary.csv")


def run_evalgrid():
    rows = []
    seed = 2026
    for name in ["PINN-MSE", "Flat-L4-squared", "MARINN"]:
        model = MLP()
        model.load_state_dict(
            torch.load(OUT / f"main5_seed{seed}/{name}.pt", map_location="cpu")
        )
        for n, nt in [(36, 21), (64, 41), (96, 61)]:
            met = eval_dense_batched(model, n, nt)
            rows.append(
                dict(seed=seed, method=name, n=n, nt=nt, npoints=n * n * nt, **met)
            )
            print("GRID", name, n, met, flush=True)
    pd.DataFrame(rows).to_csv(
        OUT / "heat2d_evaluation_grid_convergence.csv", index=False
    )


if __name__ == "__main__":
    run_main5()
    run_collocation()
    run_evalgrid()
    (OUT / "config.json").write_text(
        json.dumps(
            {
                "architecture": "3-48-48-48-1",
                "epochs": EPOCHS,
                "lr": LR,
                "boundary_factor": "16*x*(1-x)*y*(1-y), max normalized to 1",
                "main_seeds": [2024, 2025, 2026, 2027, 2028],
                "collocation_seeds": [2024, 2026, 2028],
                "collocation_levels": ["10x10x12", "14x14x16", "18x18x20"],
                "eval_grids": ["36x36x21", "64x64x41", "96x96x61"],
            },
            indent=2,
        )
    )
