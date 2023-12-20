#include <problem.hpp>

#include <stdexcept>
#include <string>

#if WITH_PYTHON
#include <pybind11/eigen/tensor.h>
#endif

namespace acl {

void Problem::config_funcs() {
    using P = Problem;
    using alpaqa::member_caller;
    funcs.n                   = n * p * q; // number of unknowns
    funcs.m                   = 0;
    funcs.eval_grad_f         = member_caller<&P::eval_grad_f>();
    funcs.eval_hess_L_prod    = member_caller<&P::eval_hess_L_prod>();
    funcs.eval_f              = member_caller<&P::eval_f>();
    funcs.eval_f_grad_f       = member_caller<&P::eval_f_grad_f>();
    funcs.eval_prox_grad_step = member_caller<&P::eval_prox_grad_step>();
    funcs.eval_inactive_indices_res_lna =
        member_caller<&P::eval_inactive_indices_res_lna>();
    funcs.eval_g           = member_caller<&P::eval_g>();
    funcs.eval_grad_g_prod = member_caller<&P::eval_grad_g_prod>();
    funcs.eval_jac_g       = member_caller<&P::eval_jac_g>();
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

} // namespace acl
