#pragma once

#include <alpaqa/config/config.hpp>
#include <alpaqa/dl/dl-problem.h>

#include <filesystem>
#include <string>

#if WITH_PYTHON
#include <pybind11/pytypes.h>
namespace py = pybind11;
#endif

namespace acl {

namespace fs = std::filesystem;
USING_ALPAQA_CONFIG(alpaqa::DefaultConfig);

/**
 * ½‖vec(AᵢXᵢ - Bᵢ)‖² + λᵢ‖vec(Xᵢ)‖₁
 */
class Problem {
  public:
    alpaqa_problem_functions_t funcs{};

  protected:
    fs::path data_file; ///< File we loaded the data from (if any)
    length_t m;         ///< Number of observations (samples)
    length_t n;         ///< Number of features
    length_t p;         ///< Number of components of each observation (targets)
    length_t q;         ///< Number of least squares terms
    real_t loss_scale;  ///< Scale factor of the quadratic loss
    real_t λ_1;         ///< ℓ₁-Regularization factor
    real_t λ_2;         ///< ℓ₂-Regularization factor

  public:
    /// Objective function.
    virtual real_t eval_f(const real_t *x_) const = 0;
    /// Gradient of objective.
    virtual void eval_grad_f(const real_t *x_, real_t *g_) const = 0;
    /// Objective and its gradient.
    virtual real_t eval_f_grad_f(const real_t *x_, real_t *g_) const = 0;
    /// Hessian-vector product of the Lagrangian.
    virtual void eval_hess_L_prod(const real_t *x_, const real_t *y_,
                                  real_t scale, const real_t *v_,
                                  real_t *Hv_) const = 0;
    /// Proximal gradient step.
    virtual real_t eval_prox_grad_step(real_t γ, const real_t *x_,
                                       const real_t *grad_ψ_, real_t *x̂_,
                                       real_t *p_) const = 0;
    /// Active indices of the Jacobian of the proximal mapping.
    virtual index_t eval_inactive_indices_res_lna(real_t γ, const real_t *x_,
                                                  const real_t *grad_ψ_,
                                                  index_t *J_) const = 0;
    virtual bool provides_eval_inactive_indices_res_lna() const { return true; }
    /// Get the number of (real) variables.
    virtual length_t get_n() const { return n * p * q; }

  public:
    /// Constraints function (unconstrained).
    void eval_g(const real_t *, real_t *) const {}
    /// Gradient-vector product of constraints.
    void eval_grad_g_prod(const real_t *, const real_t *, real_t *gr_) const {
        mvec{gr_, get_n()}.setZero();
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
    virtual void load_data(fs::path csv_filename) = 0;
#if WITH_PYTHON
    virtual void load_data(py::kwargs kwargs) = 0;
#endif
    virtual void init() = 0;
    /// Initialize the @ref funcs struct.
    void config_funcs();

  public:
    [[nodiscard]] virtual std::string get_name() const;
#if WITH_PYTHON
    [[nodiscard]] virtual py::object set_λ_1(py::args args, py::kwargs kwargs);
    [[nodiscard]] virtual py::object set_λ_2(py::args args, py::kwargs kwargs);
#endif

  protected:
    Problem() = default;

  public:
    virtual ~Problem() = default;

  public:
    void initialize(fs::path csv_filename, real_t λ_1, real_t λ_2) {
        this->λ_1 = λ_1;
        this->λ_2 = λ_2;
        load_data(std::move(csv_filename));
        init();
        config_funcs();
    }
#if WITH_PYTHON
    void initialize(py::kwargs kwargs) {
        load_data(std::move(kwargs));
        init();
        config_funcs();
    }
#endif
};

} // namespace acl
