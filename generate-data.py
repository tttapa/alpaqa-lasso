import numpy as np
import sklearn.datasets

for name in ("iris", "diabetes", "digits", "linnerud", "wine", "breast_cancer"):
    load = getattr(sklearn.datasets, f"load_{name}")
    X, y = load(return_X_y=True)
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
