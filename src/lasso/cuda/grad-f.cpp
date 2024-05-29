#include <cuda/cuda-problem.hpp>

namespace acl {

void CUDAProblem::eval_grad_f(const real_t *x_, real_t *g_) const {
    static_cast<void>(eval_f_grad_f(x_, g_));
}

} // namespace acl