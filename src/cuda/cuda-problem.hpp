#include <problem.hpp>

#include <cuda/cuda-util.hpp>

namespace acl {

class CUDAProblem : public Problem {
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

  public:
#if WITH_PYTHON
    py::object set_λ_2(py::args args, py::kwargs kwargs) override;
#endif

  private:
    real_t eval_f_cuda_stream(const real_t *x_, cudaStream_t stream) const;
    real_t eval_f_grad_f_cuda_stream(const real_t *x_, real_t *g_,
                                     cudaStream_t stream) const;
    void load_data(fs::path csv_file) override;
#if WITH_PYTHON
    void load_data(py::kwargs kwargs) override;
#endif
    void init() override;

  private:
    cublasUniqueHandle handle;
    struct {
        struct {
            cudaUniquePtr<real_t> zeros;
            cudaUniquePtr<real_t> ones;
            cudaUniquePtr<real_t> minus_ones;
            cudaUniquePtr<real_t> minus_twos;
            cudaUniquePtr<real_t> loss_scale;
            cudaUniquePtr<real_t> λ_2;
        } constants;
        struct {
            cudaUniquePtr<real_t> A;
            cudaUniquePtr<real_t> b;
        } data;
        struct {
            cudaUniquePtr<real_t> x;
            cudaUniquePtr<real_t> Ax;
            cudaUniquePtr<real_t> norms;
        } work;
    } gpu;

  public:
    CUDAProblem() = default;
};

} // namespace acl