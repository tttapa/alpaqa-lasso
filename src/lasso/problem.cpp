#include <problem.hpp>

#include <stdexcept>
#include <string>

#if WITH_PYTHON
#include <pybind11/cast.h>
#endif

namespace acl {

void Problem::config_funcs() {
    name    = get_name();
    using P = Problem;
    using alpaqa::member_caller;
    funcs.n                       = get_n(); // number of unknowns
    funcs.m                       = 0;
    funcs.name                    = name.c_str();
    funcs.eval_objective_gradient = member_caller<&P::eval_grad_f>();
    funcs.eval_lagrangian_hessian_product =
        member_caller<&P::eval_hess_L_prod>();
    funcs.eval_objective              = member_caller<&P::eval_f>();
    funcs.eval_objective_and_gradient = member_caller<&P::eval_f_grad_f>();
    funcs.eval_proximal_gradient_step =
        member_caller<&P::eval_prox_grad_step>();
    if (provides_eval_inactive_indices_res_lna())
        funcs.eval_inactive_indices_res_lna =
            member_caller<&P::eval_inactive_indices_res_lna>();
    if (provides_eval_hess_L())
        funcs.eval_lagrangian_hessian = member_caller<&P::eval_hess_L>();
    funcs.eval_constraints = member_caller<&P::eval_g>();
    funcs.eval_constraints_gradient_product =
        member_caller<&P::eval_grad_g_prod>();
    funcs.eval_constraints_jacobian = member_caller<&P::eval_jac_g>();
}

std::string Problem::get_name() const {
    if (!data_file.empty())
        return "alpaqa-lasso ('" + data_file.string() + "', " +
               std::to_string(m) + "×" + std::to_string(n) + "×" +
               std::to_string(p) + "×" + std::to_string(q) + ")";
    else
        return "alpaqa-lasso (NumPy, " + std::to_string(m) + "×" +
               std::to_string(n) + "×" + std::to_string(p) + "×" +
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
    return py::cast(std::exchange(λ_2, py::cast<real_t>(args[0])));
}
#endif

void Problem::eval_hess_L(const real_t *, const real_t *, real_t,
                          real_t *) const {
    throw std::logic_error("eval_hess_L not supported");
}

index_t Problem::eval_inactive_indices_res_lna(real_t, const real_t *,
                                               const real_t *,
                                               index_t *) const {
    throw std::logic_error("eval_inactive_indices_res_lna not supported");
}

} // namespace acl
