#include <lasso/export.h>

#include <cuda/cuda-problem.hpp>
#include <openmp/omp-problem.hpp>
#include <problem.hpp>

#include <alpaqa/params/params.hpp>

#include <algorithm>
#include <any>
#include <memory>
#include <span>
#include <stdexcept>
#include <string_view>

#if WITH_PYTHON
#include <pybind11/cast.h>
#endif

namespace acl {

using str_param_t = std::span<std::string_view>;
auto create_problem(const str_param_t &opts) {
    std::vector<unsigned> used(opts.size());
    // CSV file to load dataset from
    std::string_view datafilename;
    alpaqa::params::set_params(datafilename, "datafile", opts, used);
    if (datafilename.empty())
        throw std::invalid_argument("Missing option problem.datafile");
    // Regularization factors
    real_t λ = 0;
    alpaqa::params::set_params(λ, "lambda_1", opts, used);
    real_t μ = 0;
    alpaqa::params::set_params(μ, "lambda_2", opts, used);
    // CUDA
    bool cuda = false;
    alpaqa::params::set_params(cuda, "cuda", opts, used);
    // Check any unused options
    auto unused_opt = std::find(used.begin(), used.end(), 0);
    auto unused_idx = static_cast<size_t>(unused_opt - used.begin());
    if (unused_opt != used.end())
        throw std::invalid_argument("Unused problem option: " +
                                    std::string(opts[unused_idx]));
    std::unique_ptr<Problem> problem;
    if (cuda)
#if ACL_WITH_CUDA
        problem = std::make_unique<CUDAProblem>();
#else
        throw std::invalid_argument("CUDA support disabled");
#endif
    else
        problem = std::make_unique<OMPProblem>();
    problem->initialize(datafilename, λ, μ);
    return problem;
}

#if WITH_PYTHON
using py_param_t = std::tuple<py::args, py::kwargs>;
auto create_problem(const py_param_t &opts) {
    const auto &[args, kwargs] = opts;
    if (!args.empty())
        throw std::invalid_argument("Positional arguments not supported");
    bool cuda = kwargs.contains("cuda") && py::cast<bool>(kwargs["cuda"]);
    std::unique_ptr<Problem> problem;
    if (cuda)
#if ACL_WITH_CUDA
        problem = std::make_unique<CUDAProblem>();
#else
        throw std::invalid_argument("CUDA support disabled");
#endif
    else
        problem = std::make_unique<OMPProblem>();
    problem->initialize(kwargs);
    return problem;
}
#endif

auto create_problem(const std::any &user_data) {
    if (const auto *opts = std::any_cast<str_param_t>(&user_data))
        return create_problem(*opts);
#if WITH_PYTHON
    else if (const auto *opts = std::any_cast<py_param_t>(&user_data))
        return create_problem(*opts);
#endif
    else
        throw std::invalid_argument("Unsupported user data type");
}

} // namespace acl

/// Main entry point of this file, it is called by the
/// @ref alpaqa::dl::DLProblem class.
extern "C" LASSO_EXPORT alpaqa_problem_register_t
register_alpaqa_problem(void *user_data_v) noexcept try {
    using namespace acl;
    // Check and convert user arguments
    if (!user_data_v)
        throw std::invalid_argument("Missing user data");
    const auto &user_data = *reinterpret_cast<std::any *>(user_data_v);
    // Build and expose problem
    auto problem = create_problem(user_data);
    alpaqa_problem_register_t result;
    alpaqa::register_member_function(result, "get_name", &Problem::get_name);
#if WITH_PYTHON
    alpaqa::register_member_function(result, "set_lambda_1", &Problem::set_λ_1);
    alpaqa::register_member_function(result, "set_lambda_2", &Problem::set_λ_2);
#endif
    result.functions = &problem->funcs;
    result.instance  = problem.release();
    result.cleanup   = [](void *instance) {
        delete static_cast<Problem *>(instance);
    };
    return result;
} catch (...) {
    return {.exception = new alpaqa_exception_ptr_t{std::current_exception()}};
}
