# %%

import alpaqa
import numpy as np
import platform
import os
from pprint import pprint

# %%

rng = np.random.default_rng(seed=12345)

# Number of observations, number of features, number of targets, number of batches
m, n, p, q = 16, 32, 2, 1
m, n, p, q = 1024, 2048, 1, 64
sparsity = 0.1

# Random data matrix A
A = rng.uniform(-1, 1, (m, n, q))
x_exact = rng.uniform(-1, 1, (n, p, q))
x_exact_zeros = rng.uniform(0, 1, (n, p, q)) > sparsity
# Sparse solution x_exact
x_exact[x_exact_zeros] = 0
# Noisy right-hand side b
Ax_exact = np.einsum("mnq,npq->mpq", A, x_exact)
b = Ax_exact + 0.05 * rng.standard_normal((m, p, q))
λ = 0.02
x0 = rng.uniform(-10, 10, (n, p, q)).ravel(order="F")

# %%
path = os.path.dirname(__file__)
ext = ".dll" if platform.system() == "Windows" else ".so"
problem_path = os.path.join(path, "../build/Release/lasso" + ext)
problem = alpaqa.DLProblem(
    problem_path,
    A=np.asfortranarray(A),
    b=np.asfortranarray(b),
    lambda_1=λ,
    lambda_2=0,
    cuda=False,
)
# You can change the regularization after initialization
if False:
    problem.call_extra_func("set_lambda_1", λ := 1)
    problem.call_extra_func("set_lambda_2", 0)

cnt = alpaqa.problem_with_counters(problem)

# %%


class Callback:
    def __init__(self):
        self.resids = []

    def __call__(self, it: alpaqa.PANOCProgressInfo):
        self.resids += [it.fpr]


# %%

opt = alpaqa.PANOCParams()
opt.max_iter = 5000
opt.print_interval = 0
opt.stop_crit = alpaqa.FPRNorm
opt.print_interval = 1

direction = alpaqa.LBFGSDirection()
solver = alpaqa.PANOCSolver(opt, direction)
solver.set_progress_callback(cb := Callback())

x, stats = solver(cnt.problem, dict(tolerance=1e-8), x=x0)
x = x.reshape((n, p, q), order="F")
Ax = np.einsum("mnq,npq->mpq", A, x)
print(cnt.evaluations)
pprint(stats)

# %%

import matplotlib.pyplot as plt

plt.figure()
plt.semilogy(cb.resids, ".-")

# %%

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
ax1.set_title("Error")
ax1.plot(b.ravel(), ".-", label="Measurements")
ax1.plot(Ax.ravel(), "o-", mfc="none", label="Reconstruction")

ax2.set_title("Solution")
ax2.plot(x_exact.ravel(), ".-", label="Noise-free")
ax2.plot(x.ravel(), "o-", mfc="none", label="Solution")
plt.legend()
plt.show()

# %%
