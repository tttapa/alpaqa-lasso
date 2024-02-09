# %%

import alpaqa
import alpaqa.problems.lasso
import numpy as np
from pprint import pprint

# %%

rng = np.random.default_rng(seed=12345)

# Number of observations, number of features, number of targets, number of batches
m, n, p, q = 16, 32, 2, 1
sparsity = 0.1

# Random data matrix A
A = rng.uniform(-1, 1, (m, n, q)) + 1j * np.zeros((m, n, q))
x_exact = rng.uniform(-1, 1, (n, p, q)) + 1j * np.zeros((n, p, q))
x_exact_zeros = rng.uniform(0, 1, (n, p, q)) > sparsity
# Sparse solution x_exact
x_exact[x_exact_zeros] = 0
# Noisy right-hand side b
Ax_exact = np.einsum("mnq,npq->mpq", A, x_exact)
b = Ax_exact + 0.05 * rng.standard_normal((m, p, q))
λ = 0.02
x0 = rng.uniform(-10, 10, (2, n, p, q)).ravel(order="F")

# %%
problem = alpaqa.problems.lasso.load(
    A=np.asfortranarray(A),
    b=np.asfortranarray(b),
    lambda_1=λ,
    lambda_2=0,
    cuda=False,
)
# You can change the regularization after initialization
if True:
    problem.call_extra_func("set_lambda_1", λ := 0.01)
    problem.call_extra_func("set_lambda_2", 0.01)

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
x = x.reshape((2, n, p, q), order="F")
x = x[0] + 1j * x[1]
Ax = np.einsum("mnq,npq->mpq", A, x)
print(cnt.evaluations)
print(problem)
pprint(stats)
cb.resids += [stats["ε"]]

# %%

import matplotlib.pyplot as plt

plt.figure()
plt.title("Residuals")
plt.semilogy(cb.resids, ".-")
plt.xlabel("Iteration")
plt.ylabel("Residual")

# %%

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12))
ax1.set_title("Error (Magnitude)")
ax1.plot(abs(b.ravel()), ".-", label="Measurements")
ax1.plot(abs(Ax.ravel()), "o-", mfc="none", label="Reconstruction")

ax2.set_title("Error (Argument)")
ax2.plot(np.angle(b.ravel()), ".-", label="Measurements")
ax2.plot(np.angle(Ax.ravel()), "o-", mfc="none", label="Reconstruction")
ax2.legend(loc="lower right")

ax3.set_title("Solution (Magnitude)")
ax3.plot(abs(x_exact.ravel()), ".-", label="Noise-free")
ax3.plot(abs(x.ravel()), "o-", mfc="none", label="Solution")

ax4.set_title("Solution (Argument)")
ax4.plot(np.angle(x_exact.ravel()), ".-", label="Noise-free")
ax4.plot(np.angle(x.ravel()), "o-", mfc="none", label="Solution")
ax4.legend(loc="lower right")

plt.tight_layout()
plt.show()

# %%
