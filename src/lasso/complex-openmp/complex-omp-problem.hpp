#pragma once

#include <problem.hpp>

#include <optional>

namespace acl {

class ComplexOMPProblem : public Problem {
  public:
    /// Objective function.
    real_t eval_f(const real_t *x_) const override;
    /// Gradient of objective.
    void eval_grad_f(const real_t *x_, real_t *g_) const override;
    /// Objective and its gradient.
    real_t eval_f_grad_f(const real_t *x_, real_t *g_) const override;
    /// Hessian-vector product of the Lagrangian.
    void eval_hess_L_prod(const real_t *x_, const real_t *y_, real_t scale,
                          const real_t *v_, real_t *Hv_) const override;
    /// Proximal gradient step.
    real_t eval_prox_grad_step(real_t γ, const real_t *x_,
                               const real_t *grad_ψ_, real_t *x̂_,
                               real_t *p_) const override;
    /// Active indices of the Jacobian of the proximal mapping.
    index_t eval_inactive_indices_res_lna(real_t γ, const real_t *x_,
                                          const real_t *grad_ψ_,
                                          index_t *J_) const override;
    bool provides_eval_hess_L() const override { return false; }
    bool provides_eval_inactive_indices_res_lna() const override {
        return false;
    }
    /// Get the number of (real) variables.
    length_t get_n() const override { return 2 * n * p * q; }

  private:
    void load_data(fs::path csv_file) override;
#if WITH_PYTHON
    void load_data(py::kwargs kwargs) override;
#endif
    void init() override;

  private:
    struct {
        std::optional<crcmat> A;
        std::optional<crcmat> b;
    } data;
    struct {
        cmat A;
        cmat b;
    } storage;
    struct {
        cmat Ax;
    } mutable work;
#if WITH_PYTHON
    struct {
        py::object A;
        py::object b;
    } py_storage;
#endif

  public:
    [[nodiscard]] std::string get_name() const override {
        return "complex " + Problem::get_name();
    }

  public:
    ComplexOMPProblem() = default;
};

} // namespace acl