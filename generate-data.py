import numpy as np
import sklearn.datasets

def write_data_file(name, X, y):
    if y.ndim == 1:
        y = np.vstack(y)
    print(name)
    print("  #observations:", X.shape[0])
    print("  #features:    ", X.shape[1])
    print("  #targets:     ", y.shape[1])
    assert y.shape[0] == X.shape[0]
    with open(f"{name}.csv", "w") as f:
        delim = dict(delimiter=",", newline="\n")
        np.savetxt(f, [X.shape + y.shape[1:]], **delim, fmt="%d")
        np.savetxt(f, y.T, **delim)
        np.savetxt(f, X.T, **delim)

for name in ("iris", "diabetes", "digits", "linnerud", "wine", "breast_cancer"):
    load = getattr(sklearn.datasets, f"load_{name}")
    X, y = load(return_X_y=True)
    write_data_file(name, X, y)

rng = np.random.default_rng(seed=12345)

# Number of observations, number of features, number of targets
m, n, p = 128, 256, 8
sparsity = 0.1

# Random data matrix A
A = rng.uniform(-1, 1, (m, n))
x_exact = rng.uniform(-1, 1, (n, p))
x_exact_zeros = rng.uniform(0, 1, (n, p)) > sparsity
# Sparse solution x_exact
x_exact[x_exact_zeros] = 0
# Noisy right-hand side b
Ax_exact = np.einsum("mn,np->mp", A, x_exact)
b = Ax_exact + 0.05 * rng.standard_normal((m, p))
λ = 0.02
x0 = rng.uniform(-10, 10, (n, p)).ravel(order="F")
write_data_file("random", A, b)
