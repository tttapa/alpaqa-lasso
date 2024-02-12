# alpaqa-lasso

Parallelized and GPU-accelerated lasso optimization problem definition for the
[**alpaqa**](https://github.com/kul-optec/alpaqa) solvers. Supports both real
and complex problems, with $`\ell_1`$ and $`\ell_2`$ regularization.

```math
\begin{equation}
    \underset{X}{\textbf{minimize}}
    \;\;\sum_{i=1}^q\;\; \frac{1}{2m} \big\| A_i X_i - B_i \big\|_F^2 + \frac{\lambda_2}{2} \big\|X_i\big\|_F^2 + \lambda_1 \big\|\mathrm{vec}(X_i)\big\|_1
\end{equation}
```

The matrices $`A_i`$ have dimensions $`m\times n`$, the matrices $`X_i`$ have
dimensions $`n\times p`$, and the matrices $`B_i`$ have dimensions $`m\times p`$.
Multiple batches of lasso problems can be solved at once, by providing order 3
tensors instead of matrices, where the dimension of largest stride $`q`$ is the
number of batches.

The problem definition is implemented in C++, and can be used with the
[`alpaqa-driver`](https://kul-optec.github.io/alpaqa/Doxygen/problems_2sparse-logistic-regression_8cpp-example.html)
utility, loading data from CSV files, or through the [alpaqa Python interface](https://kul-optec.github.io/alpaqa/Sphinx/index.html),
passing the data as NumPy arrays.

> [!IMPORTANT]  
> This project is still very much experimental, and it requires a pre-release
> version of alpaqa to work correctly.

## Installation

Download both the `alpaqa-python` and `alpaqa-lasso-python` artifacts from the
latest [GitHub Actions run](https://github.com/tttapa/alpaqa-lasso/actions),
unzip them, and install the appropriate Wheel files into your Python virtual
environment using `pip install filename.whl`.

**TODO:** Instructions for building from source, see
[.github/workflows/python.yml](https://github.com/tttapa/alpaqa-lasso/blob/main/.github/workflows/python.yml) for now.

## Examples

- [python/lasso.py](https://github.com/tttapa/alpaqa-lasso/blob/main/python/lasso.py) (real variables)
- [python/complex-lasso.py](https://github.com/tttapa/alpaqa-lasso/blob/main/python/complex-lasso.py) (complex variables)
- [generate-data.py](https://github.com/tttapa/alpaqa-lasso/blob/main/generate-data.py) (generates CSV data files for use with `alpaqa-driver`)
