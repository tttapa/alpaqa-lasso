#include "problem.hpp"

#include <alpaqa/functions/l1-norm.hpp>
#include <alpaqa/functions/prox.hpp>
#include <alpaqa/util/io/csv.hpp>
#include <alpaqa/util/lifetime.hpp>

#include <fstream>
#include <string>

#if WITH_PYTHON
#include <pybind11/eigen.h>
#endif

namespace acl {

real_t Problem::eval_f(const real_t *x_) const {
    cmmat x{x_, n, p * q};
    real_t scal      = 1 / static_cast<real_t>(m);
    real_t sq_norm   = 0;
    real_t sq_norm_x = 0;
#pragma omp parallel for reduction(+ : sq_norm) reduction(+ : sq_norm_x)
    for (index_t i = 0; i < q; i++) {
        auto Ai        = A.middleCols(i * n, n);
        auto xi        = x.middleCols(i * p, p);
        auto Aixi      = Ax.middleCols(i * p, p);
        auto bi        = b.middleCols(i * p, p);
        Aixi.noalias() = Ai * xi;
        sq_norm += (Aixi - bi).squaredNorm();
        sq_norm_x += xi.squaredNorm();
    }
    return 0.5 * scal * sq_norm + 0.5 * λ_2 * sq_norm_x;
}

void Problem::eval_grad_f(const real_t *x_, real_t *g_) const {
    cmmat x{x_, n, p * q};
    mmat g{g_, n, p * q};
    real_t scal = 1 / static_cast<real_t>(m);
#pragma omp parallel for
    for (index_t i = 0; i < q; i++) {
        auto Ai        = A.middleCols(i * n, n);
        auto xi        = x.middleCols(i * p, p);
        auto Aixi      = Ax.middleCols(i * p, p);
        auto bi        = b.middleCols(i * p, p);
        auto gi        = g.middleCols(i * p, p);
        Aixi.noalias() = Ai * xi - bi;
        gi.noalias()   = scal * (Ai.adjoint() * Aixi) + λ_2 * xi;
    }
}

real_t Problem::eval_f_grad_f(const real_t *x_, real_t *g_) const {
    cmmat x{x_, n, p * q};
    mmat g{g_, n, p * q};
    real_t scal      = 1 / static_cast<real_t>(m);
    real_t sq_norm   = 0;
    real_t sq_norm_x = 0;
#pragma omp parallel for reduction(+ : sq_norm) reduction(+ : sq_norm_x)
    for (index_t i = 0; i < q; i++) {
        auto Ai        = A.middleCols(i * n, n);
        auto xi        = x.middleCols(i * p, p);
        auto Aixi      = Ax.middleCols(i * p, p);
        auto bi        = b.middleCols(i * p, p);
        auto gi        = g.middleCols(i * p, p);
        Aixi.noalias() = Ai * xi - bi;
        sq_norm += Aixi.squaredNorm();
        gi.noalias() = scal * (Ai.adjoint() * Aixi) + λ_2 * xi;
        sq_norm_x += xi.squaredNorm();
    }
    return 0.5 * scal * sq_norm + 0.5 * λ_2 * sq_norm_x;
}

void Problem::eval_hess_L_prod(const real_t *x_ [[maybe_unused]],
                               const real_t *y_ [[maybe_unused]], real_t scale,
                               const real_t *v_, real_t *Hv_) const {
    cmmat v{v_, n, p * q};
    mmat Hv{Hv_, n, p * q};
    real_t scal = scale / static_cast<real_t>(m);
#pragma omp parallel for
    for (index_t i = 0; i < q; i++) {
        auto Ai        = A.middleCols(i * n, n);
        auto vi        = v.middleCols(i * p, p);
        auto Hvi       = Hv.middleCols(i * p, p);
        auto Aixi      = Ax.middleCols(i * p, p);
        Aixi.noalias() = Ai * vi;
        Hvi.noalias()  = scal * (Ai.adjoint() * Aixi) + (scale * λ_2) * vi;
    }
}

real_t Problem::eval_prox_grad_step(real_t γ, const real_t *x_,
                                    const real_t *grad_ψ_, real_t *x̂_,
                                    real_t *p_) const {
    alpaqa::functions::L1Norm<config_t, real_t> reg = λ_1;
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
    index_t nJ = 0;
    for (index_t i = 0; i < n * p * q; i++)
        if (λ_1 == 0 || std::abs(x_[i] - γ * grad_ψ_[i]) > γ * λ_1)
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
    b.resize(m, p * q);
    A.resize(m, n * q);
    // Read the measurements
    for (length_t i = 0; i < p * q; ++i)
        alpaqa::csv::read_row(csv_file, b.col(i));
    // Read the data
    for (length_t i = 0; i < n * q; ++i)
        alpaqa::csv::read_row(csv_file, A.col(i));
}

void Problem::init(bool cuda) {
    Ax.resize(m, p * q);

#if ACL_WITH_CUDA
    if (cuda) {
        handle     = cublasUniqueCreate();
        ones       = cudaAlloc<cuDoubleComplex>(F);
        minus_ones = cudaAlloc<cuDoubleComplex>(F);
        mus        = cudaAlloc<cuDoubleComplex>(F);
        A_gpu      = cudaAlloc<cuDoubleComplex>(m * n * F);
        b_gpu      = cudaAlloc<cuDoubleComplex>(m * F);
        x_gpu      = cudaAlloc<cuDoubleComplex>(n * F);
        Ax_gpu     = cudaAlloc<cuDoubleComplex>(m * F);
        norms_gpu  = cudaAlloc<real_t>(2);
        cvec work  = cvec::Ones(F);
        check(cublasSetVector(static_cast<int>(F), sizeof(cuDoubleComplex),
                              work.data(), 1, ones.get(), 1),
              "set ones");
        work = -cvec::Ones(F);
        check(cublasSetVector(static_cast<int>(F), sizeof(cuDoubleComplex),
                              work.data(), 1, minus_ones.get(), 1),
              "set minus_ones");
        work.setConstant(F, μ);
        check(cublasSetVector(static_cast<int>(F), sizeof(cuDoubleComplex),
                              work.data(), 1, mus.get(), 1),
              "set mus");
        auto stream1 = acl::cudaStreamAlloc(cudaStreamNonBlocking),
             stream2 = acl::cudaStreamAlloc(cudaStreamNonBlocking);
        check(cudaMemcpyAsync(A_gpu.get(), A.data(),
                              2 * sizeof(real_t) * m * n * F,
                              cudaMemcpyHostToDevice, stream1.get()),
              "memcpy A");
        check(cudaMemcpyAsync(b_gpu.get(), b.data(), 2 * sizeof(real_t) * m * F,
                              cudaMemcpyHostToDevice, stream2.get()),
              "memcpy b");
    }
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
    if (mus) {
        cvec work = cvec::Constant(F, μ);
        check(cublasSetVector(static_cast<int>(F), sizeof(cuDoubleComplex),
                              work.data(), 1, mus.get(), 1),
              "set mus");
    }
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
    A   = py::cast<crmat>(kwargs["A"]);
    b   = py::cast<crmat>(kwargs["b"]);
    q   = b.cols();
    m   = b.rows();
    if (A.rows() != m)
        throw std::invalid_argument("Number of rows of A and b should match");
    if (auto d = std::div(A.cols(), q); d.rem != 0)
        throw std::invalid_argument("Number of columns of A should be an "
                                    "integer multiple of that of b");
    else
        n = d.quot;
    bool cuda = kwargs.contains("cuda") && py::cast<bool>(kwargs["cuda"]);
    init(cuda);
    config_funcs(cuda);
}
#endif

} // namespace acl
