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

## Build from source

```sh
# Download alpaqa
git clone https://github.com/kul-optec/alpaqa --branch=1.0.0a18
# Install the dependencies for alpaqa using Conan
conan install ./alpaqa --build=missing \
    -c tools.cmake.cmaketoolchain:generator="Ninja Multi-Config" \
    -s build_type=Release -o with_python=True
# Build and install the alpaqa Python package
python3 -m pip install ./alpaqa -v -C--local="$PWD/scripts/dev/alpaqa.toml"
# Add alpaqa itself to your Conan cache
conan export alpaqa
# Install the dependencies for alpaqa-lasso using Conan
conan install . --build=missing \
    -c tools.cmake.cmaketoolchain:generator="Ninja Multi-Config" \
    -s build_type=Release
# Build and install the alpaqa-lasso Python package
python3 -m pip install . -v
```

To enable CUDA support, edit the appropriate `scripts/dev/cudaXX.toml` file
and then use them during installation. For example:

```sh
# Build and install the alpaqa-lasso Python package
python3 -m pip install . -v -C--local="$PWD/scripts/dev/cuda11.toml"
```

## Examples

- [python/lasso.py](https://github.com/tttapa/alpaqa-lasso/blob/main/python/lasso.py) (real variables)
- [python/complex-lasso.py](https://github.com/tttapa/alpaqa-lasso/blob/main/python/complex-lasso.py) (complex variables)
- [generate-data.py](https://github.com/tttapa/alpaqa-lasso/blob/main/generate-data.py) (generates CSV data files for use with `alpaqa-driver`)
