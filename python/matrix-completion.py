# %%

import alpaqa
import alpaqa.problems.matrix_completion
import numpy as np
from pprint import pprint

# %%

rng = np.random.default_rng(seed=12345)

# Number of rows, columns, rank
m, n, r = 150, 150, 50
sparsity = 0.8

# Random data matrix A
U = rng.uniform(-1, 1, (m, r))
V = rng.uniform(-1, 1, (r, n))
X = U @ V
i = rng.uniform(0, 1, (m, n)) <= sparsity
M_rows, M_cols = np.nonzero(i)
M_values = X[M_rows, M_cols]

# %%

problem = alpaqa.problems.matrix_completion.load(
    m=m, n=n,M_values=M_values, M_rows=M_rows,M_cols=M_cols
)
cnt = alpaqa.problem_with_counters(problem)

# %%


class Callback:
    def __init__(self):
        self.resids = []

    def __call__(self, it: alpaqa.PANOCProgressInfo):
        if it.k == 0:
            self.resids.append([])
        self.resids[-1] += [it.fpr]


# %%

opt = alpaqa.PANOCParams()
opt.max_iter = 5000
opt.print_interval = 0
opt.stop_crit = alpaqa.FPRNorm
opt.print_interval = 100

alm_opt = alpaqa.ALMParams()
alm_opt.print_interval = 1
alm_opt.tolerance = 1e-8
alm_opt.dual_tolerance = 1e-8

direction = alpaqa.LBFGSDirection({"memory": 30})
solver = alpaqa.PANOCSolver(opt, direction)
solver.set_progress_callback(cb := Callback())
alm_solver = alpaqa.ALMSolver(alm_opt, solver)

X0 = np.zeros_like(X)
X0[M_rows, M_cols] = M_values
x, y, stats = alm_solver(cnt.problem, x=X0.ravel(order="F"))
X_sol = np.reshape(x, (m, n), order="F")
print()
print(cnt.evaluations)
print(problem)
print()
pprint(stats, sort_dicts=False)
cb.resids += [stats["ε"]]

# %%

import matplotlib.pyplot as plt

plt.figure()
plt.title("Residuals")
for r in cb.resids:
    plt.semilogy(r, ".-")
plt.xlabel("Inner iteration")
plt.ylabel("Residual")
plt.tight_layout()

plt.figure()
plt.title("Singular values")
plt.semilogy(np.linalg.svd(X, compute_uv=False), ".-", label="Original")
plt.semilogy(np.linalg.svd(X_sol, compute_uv=False), ".-", label="Reconstructed")
plt.legend()
plt.tight_layout()

plt.show()

# %%
