#pragma once

#include <alpaqa/config/config.hpp>
#include <alpaqa/dl/dl-problem.h>
#include <unsupported/Eigen/CXX11/Tensor>

#include <filesystem>
#include <string>

#if ACL_WITH_CUDA
#include "cuda-util.hpp"
#endif

#if WITH_PYTHON
#include <pybind11/pytypes.h>
#endif

namespace acl {

namespace py = pybind11;
namespace fs = std::filesystem;
USING_ALPAQA_CONFIG(alpaqa::DefaultConfig);
using tensor3   = Eigen::Tensor<real_t, 3>;
using mtensor3  = Eigen::TensorMap<tensor3>;
using cmtensor3 = Eigen::TensorMap<const tensor3>;
using rtensor3  = Eigen::TensorRef<tensor3>;
using crtensor3 = Eigen::TensorRef<const tensor3>;
using tensor2   = Eigen::Tensor<real_t, 2>;
using mtensor2  = Eigen::TensorMap<tensor2>;
using cmtensor2 = Eigen::TensorMap<const tensor2>;
using rtensor2  = Eigen::TensorRef<tensor2>;
using crtensor2 = Eigen::TensorRef<const tensor2>;

/**
 * ½‖vec(AᵢXᵢ - Bᵢ)‖² + λᵢ‖vec(Xᵢ)‖₁
 */
class Problem {
  public:
    alpaqa_problem_functions_t funcs{};

  private:
    fs::path data_file; ///< File we loaded the data from
    length_t m;         ///< Number of observations (samples)
    length_t n;         ///< Number of features
    length_t p;         ///< Number of components of each observation (targets)
    length_t q;         ///< Number of least squares terms
    real_t loss_scale;  ///< Scale factor of the quadratic loss
    real_t λ_1;         ///< ℓ₁-Regularization factor
    real_t λ_2;         ///< ℓ₂-Regularization factor
    tensor3 A;          ///< Feature matrix (m×n×q)
    tensor3 b;          ///< Observations (m×p×q)
    tensor3 AᵀA;        ///< Cached Hessian AᵀA (n×n×q)
    tensor3 Aᵀb;        ///< Cached product Aᵀb (n×p×q)
    mutable tensor3 Ax; ///< Work vector (m×p×q)
    mutable tensor3 w;  ///< Work vector (n×p×q)
    real_t bᵀb;
#if ACL_WITH_CUDA
    cublasUniqueHandle handle;
    cudaUniquePtr<real_t> zeros;
    cudaUniquePtr<real_t> ones;
    cudaUniquePtr<real_t> minus_ones;
    cudaUniquePtr<real_t> minus_twos;
    cudaUniquePtr<real_t> loss_scale_gpu;
    cudaUniquePtr<real_t> λ_2_gpu;
    cudaUniquePtr<real_t> A_gpu;
    cudaUniquePtr<real_t> b_gpu;
    cudaUniquePtr<real_t> x_gpu;
    cudaUniquePtr<real_t> g_gpu;
    cudaUniquePtr<real_t> Ax_gpu;
    cudaUniquePtr<real_t> norms_gpu;
    cudaUniquePtr<real_t> AᵀA_gpu;
    cudaUniquePtr<real_t> Aᵀb_gpu;
    cudaUniquePtr<real_t> w_gpu;
#endif

  public:
    /// Objective function.
    real_t eval_f(const real_t *x_) const;
    /// Gradient of objective.
    void eval_grad_f(const real_t *x_, real_t *g_) const;
    /// Objective and its gradient.
    real_t eval_f_grad_f(const real_t *x_, real_t *g_) const;
    /// Hessian-vector product of the Lagrangian.
    void eval_hess_L_prod(const real_t *x_, const real_t *y_, real_t scale,
                          const real_t *v_, real_t *Hv_) const;
    /// Proximal gradient step.
    real_t eval_prox_grad_step(real_t γ, const real_t *x_,
                               const real_t *grad_ψ_, real_t *x̂_,
                               real_t *p_) const;
    /// Active indices of the Jacobian of the proximal mapping.
    index_t eval_inactive_indices_res_lna(real_t γ, const real_t *x_,
                                          const real_t *grad_ψ_,
                                          index_t *J_) const;

#if ACL_WITH_CUDA
  private:
    real_t eval_f_cuda_stream(const real_t *x_, cudaStream_t stream) const;
    real_t eval_f_grad_f_cuda_stream(const real_t *x_, real_t *g_,
                                     cudaStream_t stream) const;
    real_t eval_f_grad_f_cuda_stream_cached(const real_t *x_, real_t *g_,
                                            cudaStream_t stream) const;

  public:
    void init_cuda(bool cache_hessian);
    void update_λ_2_cuda();
    /// Objective function.
    real_t eval_f_cuda(const real_t *x_) const;
    /// Objective and its gradient.
    real_t eval_f_grad_f_cuda(const real_t *x_, real_t *g_) const;
#endif

  public:
    /// Constraints function (unconstrained).
    void eval_g(const real_t *, real_t *) const {}
    /// Gradient-vector product of constraints.
    void eval_grad_g_prod(const real_t *, const real_t *, real_t *gr_) const {
        mvec{gr_, n * p * q}.setZero();
    }
    /// Jacobian of constraints.
    void eval_jac_g(const real_t *, real_t *) const {}

  private:
    /// Loads problem data from a CSV file.
    /// The first row contains the number of measurements, the number of
    /// features, and the number of frequencies.
    /// The following rows contain the complex measurements b, interleaving real
    /// and imaginary components, one row per frequency.
    /// Every following row contains a column of the complex data matrix A,
    /// again interleaving real and imaginary components, one row per
    /// measurement.
    void load_data();

    void init(bool cuda);
    void config_funcs(bool cuda);

  public:
    std::string get_name() const;
#if WITH_PYTHON
    py::object set_λ_1(py::args args, py::kwargs kwargs);
    py::object set_λ_2(py::args args, py::kwargs kwargs);
#endif

    /// Constructor loads CSV data file and exposes the problem functions by
    /// initializing the @c funcs member.
    Problem(fs::path csv_filename, real_t λ_1, real_t λ_2, bool blas,
            bool cuda);
#if WITH_PYTHON
    /// Constructor gets the data from a Python dict and exposes the problem
    /// functions by initializing the @c funcs member.
    explicit Problem(const py::kwargs &kwargs);
#endif
};

} // namespace acl
