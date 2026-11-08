from pathlib import Path
import numpy as np
import pandas as pd
from scipy.fft import dstn, idstn

OUT = Path("reference_validation")
OUT.mkdir(exist_ok=True)


def reaction_step(u, h):
    return u / np.sqrt(u * u + (1 - u * u) * np.exp(-2 * h))


def solve(N=40, nu=0.01, T=0.4, dt=0.001, A=0.9, nt_save=21):
    x = np.arange(1, N + 1) / (N + 1)
    X, Y = np.meshgrid(x, x, indexing="xy")
    u = A * np.sin(np.pi * X) * np.sin(np.pi * Y)
    k = np.arange(1, N + 1)
    lam = np.pi**2 * (k[:, None] ** 2 + k[None, :] ** 2)
    diff = np.exp(-nu * lam * dt)
    steps = int(round(T / dt))
    save_steps = np.linspace(0, steps, nt_save, dtype=int)
    states = [u.copy()]
    si = 1
    for n in range(1, steps + 1):
        u = reaction_step(u, dt / 2)
        u = idstn(dstn(u, type=1, norm="ortho") * diff, type=1, norm="ortho")
        u = reaction_step(u, dt / 2)
        if si < len(save_steps) and n == save_steps[si]:
            states.append(u.copy())
            si += 1
    return x, np.asarray(states)


def rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def mx(a, b):
    return np.max(np.abs(a - b))


refs = {dt: solve(N=40, dt=dt)[1] for dt in (0.002, 0.001, 0.0005)}
rows = []
for c, f in ((0.002, 0.001), (0.001, 0.0005)):
    rows.append(
        [
            "time refinement N=40",
            f"dt={c}",
            f"dt={f}",
            rel(refs[c], refs[f]),
            mx(refs[c], refs[f]),
        ]
    )

_, u40 = solve(N=40, dt=0.0005)
_, u81 = solve(N=81, dt=0.0005)
u81c = u81[:, 1::2, 1::2]

rows.append(
    [
        "space refinement dt=0.0005",
        "N=40",
        "N=81 restricted to N=40 nodes",
        rel(u40, u81c),
        mx(u40, u81c),
    ]
)

df = pd.DataFrame(
    rows,
    columns=["study", "coarse", "fine", "relative_L2_difference", "max_abs_difference"],
)

df.to_csv(OUT / "allen_cahn_reference_convergence.csv", index=False)
pd.DataFrame(
    [
        [
            "N=40, dt=0.001",
            "N=81, dt=0.0005 restricted to N=40 nodes",
            rel(refs[0.001], u81c),
            mx(refs[0.001], u81c),
        ]
    ],
    columns=[
        "reference_used",
        "refined_reference",
        "relative_L2_difference",
        "max_abs_difference",
    ],
).to_csv(OUT / "production_reference_vs_refined.csv", index=False)
print(df.to_string(index=False))
