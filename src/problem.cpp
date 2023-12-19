#include "problem.hpp"

#include <alpaqa/functions/l1-norm.hpp>
#include <alpaqa/functions/prox.hpp>
#include <alpaqa/util/io/csv.hpp>
#include <alpaqa/util/lifetime.hpp>

#include <array>
#include <fstream>
#include <stdexcept>
#include <string>

#if WITH_PYTHON
#include <pybind11/eigen/tensor.h>
#endif

namespace acl {

namespace {
real_t normSquared(const crtensor2 &t) {
    return cmvec{t.data(), t.size()}.squaredNorm();
}
} // namespace

real_t Problem::eval_f(const real_t *x_) const {
    cmtensor3 x{x_, n, p, q};
    const std::array mat_mat{Eigen::IndexPair<index_t>{1, 0}};
    real_t sq_norm   = 0;
    real_t sq_norm_x = 0;
#pragma omp parallel for reduction(+ : sq_norm) reduction(+ : sq_norm_x)
    for (index_t i = 0; i < q; ++i) {
        auto Ai  = A.chip(i, 2);
        auto Axi = Ax.chip(i, 2);
        auto xi  = x.chip(i, 2);
        auto bi  = b.chip(i, 2);
        Axi      = Ai.contract(xi, mat_mat) - bi;
        sq_norm += normSquared(Axi);
        sq_norm_x += normSquared(xi);
    }
    return 0.5 * loss_scale * sq_norm + 0.5 * λ_2 * sq_norm_x;
}

void Problem::eval_grad_f(const real_t *x_, real_t *g_) const {
    cmtensor3 x{x_, n, p, q};
    mtensor3 g{g_, n, p, q};
    const std::array mat_mat{Eigen::IndexPair<index_t>{1, 0}};
    const std::array mat_tp_mat{Eigen::IndexPair<index_t>{0, 0}};
#pragma omp parallel for
    for (index_t i = 0; i < q; ++i) {
        auto Ai  = A.chip(i, 2);
        auto Axi = Ax.chip(i, 2);
        auto xi  = x.chip(i, 2);
        auto bi  = b.chip(i, 2);
        auto gi  = g.chip(i, 2);
        Axi      = Ai.contract(xi, mat_mat) - bi;
        gi       = loss_scale * Ai.contract(Axi, mat_tp_mat) + λ_2 * xi;
    }
}

real_t Problem::eval_f_grad_f(const real_t *x_, real_t *g_) const {
    cmtensor3 x{x_, n, p, q};
    mtensor3 g{g_, n, p, q};
    const std::array mat_mat{Eigen::IndexPair<index_t>{1, 0}};
    const std::array mat_tp_mat{Eigen::IndexPair<index_t>{0, 0}};
    real_t sq_norm   = 0;
    real_t sq_norm_x = 0;
#pragma omp parallel for reduction(+ : sq_norm) reduction(+ : sq_norm_x)
    for (index_t i = 0; i < q; ++i) {
        auto Ai  = A.chip(i, 2);
        auto Axi = Ax.chip(i, 2);
        auto xi  = x.chip(i, 2);
        auto bi  = b.chip(i, 2);
        auto gi  = g.chip(i, 2);
        Axi      = Ai.contract(xi, mat_mat) - bi;
        gi       = loss_scale * Ai.contract(Axi, mat_tp_mat) + λ_2 * xi;
        sq_norm += normSquared(Axi);
        sq_norm_x += normSquared(xi);
    }
    return 0.5 * loss_scale * sq_norm + 0.5 * λ_2 * sq_norm_x;
}

void Problem::eval_hess_L_prod(const real_t *x_ [[maybe_unused]],
                               const real_t *y_ [[maybe_unused]], real_t scale,
                               const real_t *v_, real_t *Hv_) const {
    cmtensor3 v{v_, n, p, q};
    mtensor3 Hv{Hv_, n, p, q};
    const std::array mat_mat{Eigen::IndexPair<index_t>{1, 0}};
    const std::array mat_tp_mat{Eigen::IndexPair<index_t>{0, 0}};
#pragma omp parallel for
    for (index_t i = 0; i < q; ++i) {
        auto Ai  = A.chip(i, 2);
        auto Axi = Ax.chip(i, 2);
        auto vi  = v.chip(i, 2);
        auto Hvi = Hv.chip(i, 2);
        Axi      = Ai.contract(vi, mat_mat);
        Hvi      = scale * loss_scale * Ai.contract(Axi, mat_tp_mat) +
              (scale * λ_2) * vi;
    }
}

real_t Problem::eval_prox_grad_step(real_t γ, const real_t *x_,
                                    const real_t *grad_ψ_, real_t *x̂_,
                                    real_t *p_) const {
    alpaqa::functions::L1Norm<config_t> reg = λ_1;
    cmmat x{x_, n, p * q}, grad_ψ{grad_ψ_, n, p * q};
    mmat x̂{x̂_, n, p * q}, step{p_, n, p * q};
    real_t norm = 0;
#pragma omp parallel for reduction(+ : norm)
    for (index_t i = 0; i < p * q; i++)
        norm += alpaqa::prox_step(reg, x.col(i), grad_ψ.col(i), x̂.col(i),
                                  step.col(i), γ, -γ);
    return norm;
}

index_t Problem::eval_inactive_indices_res_lna(real_t γ, const real_t *x_,
                                               const real_t *grad_ψ_,
                                               index_t *J_) const {
    real_t thres = γ * λ_1;
    index_t nJ   = 0;
    for (index_t i = 0; i < n * p * q; i++)
        if (λ_1 == 0 || std::abs(x_[i] - γ * grad_ψ_[i]) > thres)
            J_[nJ++] = i;
    return nJ;
}

void Problem::load_data() {
    std::ifstream csv_file{data_file};
    if (!csv_file)
        throw std::runtime_error("Unable to open file '" + data_file.string() +
                                 "'");
    // Load dimensions (#observations, #features, #targets, #terms)
    auto dims = alpaqa::csv::read_row_std_vector<length_t>(csv_file);
    p = q = 1;
    if (dims.size() < 2 || dims.size() > 4)
        throw std::runtime_error("Invalid problem dimensions in data file \'" +
                                 data_file.string() + '\'');
    m = dims[0];
    n = dims[1];
    b.resize(m, p, q);
    A.resize(m, n, q);
    // Read the measurements
    for (length_t i = 0; i < q; ++i)
        for (length_t j = 0; j < p; ++j)
            alpaqa::csv::read_row(csv_file,
                                  mvec{b.data() + j * m + i * m * p, m});
    // Read the data
    for (length_t i = 0; i < q; ++i)
        for (length_t j = 0; j < n; ++j)
            alpaqa::csv::read_row(csv_file,
                                  mvec{A.data() + j * m + i * m * n, m});
}

void Problem::init(bool cuda) {
    loss_scale = 1 / static_cast<real_t>(m);
    Ax.resize(m, p, q);

    if (cuda)
#if ACL_WITH_CUDA
        init_cuda();
#else
        throw std::runtime_error("CUDA support disabled");
#endif
}

void Problem::config_funcs(bool cuda) {
    using P = Problem;
    using alpaqa::member_caller;
    funcs.n                = n * p * q; // number of unknowns
    funcs.m                = 0;
    funcs.eval_grad_f      = member_caller<&P::eval_grad_f>();
    funcs.eval_hess_L_prod = member_caller<&P::eval_hess_L_prod>();
    if (cuda) {
#if ACL_WITH_CUDA
        funcs.eval_f        = member_caller<&P::eval_f_cuda>();
        funcs.eval_f_grad_f = member_caller<&P::eval_f_grad_f_cuda>();
#else
        throw std::invalid_argument(
            "CUDA not supported. Recompile with WITH_CUDA=On.");
#endif
    } else {
        funcs.eval_f        = member_caller<&P::eval_f>();
        funcs.eval_f_grad_f = member_caller<&P::eval_f_grad_f>();
    }
    funcs.eval_prox_grad_step = member_caller<&P::eval_prox_grad_step>();
    funcs.eval_inactive_indices_res_lna =
        member_caller<&P::eval_inactive_indices_res_lna>();
    funcs.eval_g           = member_caller<&P::eval_g>();
    funcs.eval_grad_g_prod = member_caller<&P::eval_grad_g_prod>();
    funcs.eval_jac_g       = member_caller<&P::eval_jac_g>();
}

std::string Problem::get_name() const {
    return "alpaqa-lasso ('" + data_file.string() + "', " + std::to_string(m) +
           "×" + std::to_string(n) + "×" + std::to_string(p) + "×" +
           std::to_string(q) + ")";
}

#if WITH_PYTHON
py::object Problem::set_λ_1(py::args args, py::kwargs kwargs) {
    if (args.size() != 1)
        throw std::invalid_argument("Invalid number of positional arguments");
    if (!kwargs.empty())
        throw std::invalid_argument("Unexpected keyword arguments");
    return py::cast(std::exchange(λ_1, py::cast<real_t>(args[0])));
}
py::object Problem::set_λ_2(py::args args, py::kwargs kwargs) {
    if (args.size() != 1)
        throw std::invalid_argument("Invalid number of positional arguments");
    if (!kwargs.empty())
        throw std::invalid_argument("Unexpected keyword arguments");
    auto old = std::exchange(λ_2, py::cast<real_t>(args[0]));
#if ACL_WITH_CUDA
    update_λ_2_cuda();
#endif
    return py::cast(old);
}
#endif

Problem::Problem(fs::path csv_filename, real_t λ_1, real_t λ_2, bool blas,
                 bool cuda)
    : data_file(std::move(csv_filename)), λ_1(λ_1), λ_2(λ_2) {
    load_data();
    init(cuda);
    config_funcs(cuda);
}

#if WITH_PYTHON
Problem::Problem(const py::kwargs &kwargs) {
    λ_1 = py::cast<real_t>(kwargs["lambda_1"]);
    λ_2 = py::cast<real_t>(kwargs["lambda_2"]);
    A   = py::cast<cmtensor3>(kwargs["A"]);
    b   = py::cast<cmtensor3>(kwargs["b"]);
    n   = A.dimension(1);
    m   = A.dimension(0);
    p   = b.dimension(1);
    q   = b.dimension(2);
    if (m != b.dimension(0))
        throw std::invalid_argument("Number of rows of A and b should match");
    if (q != b.dimension(2))
        throw std::invalid_argument("Batch size of A and b should match");
    bool cuda = kwargs.contains("cuda") && py::cast<bool>(kwargs["cuda"]);
    init(cuda);
    config_funcs(cuda);
}
#endif

} // namespace acl
