#pragma once

#include <alpaqa/config/config.hpp>
#include <alpaqa/dl/dl-problem.h>
#include <alpaqa/functions/nuclear-norm.hpp>
#include <alpaqa/functions/prox.hpp>
#include <alpaqa/util/io/csv.hpp>

#include <cassert>
#include <filesystem>
#include <fstream>
#include <string>

#include <Eigen/Core>

#if WITH_PYTHON
#include <pybind11/cast.h>
#include <pybind11/eigen.h>
#include <pybind11/pytypes.h>
namespace py = pybind11;
#endif

namespace amc {

namespace fs = std::filesystem;
USING_ALPAQA_CONFIG(alpaqa::DefaultConfig);

/**
 * minimize ‖X‖*
 * subject to X[i,j] = M[i,j] for all (i,j) in Ω
 */
class Problem {
  public:
    alpaqa_problem_functions_t funcs{};

  protected:
    fs::path data_file; ///< File we loaded the data from (if any)
    std::string name;   ///< Name/brief description of the problem
    length_t m;         ///< Number of rows of M
    length_t n;         ///< Number of columns of M

    mutable alpaqa::functions::NuclearNorm<config_t> nuclear_norm;

    vec M_values;
    indexvec M_rows, M_cols;

  public:
    /// Objective function.
    real_t eval_f(const real_t *) { return 0; }
    /// Gradient of objective.
    void eval_grad_f(const real_t *, real_t *g_) const {
        mvec{g_, m * n}.setZero();
    }
    /// Objective and its gradient.
    real_t eval_f_grad_f(const real_t *, real_t *g_) const {
        mvec{g_, m * n}.setZero();
        return 0;
    }
    /// Proximal gradient step.
    real_t eval_prox_grad_step(real_t γ, const real_t *x_,
                               const real_t *grad_ψ_, real_t *x̂_,
                               real_t *p_) const {
        return alpaqa::prox_step(nuclear_norm, cmvec{x_, m * n},
                                 cmvec{grad_ψ_, m * n}, mvec{x̂_, m * n},
                                 mvec{p_, m * n}, γ, -γ);
    }

  public:
    /// Constraints function.
    void eval_g(const real_t *x_, real_t *g_) const {
        auto X = cmvec{x_, m * n}.reshaped(m, n);
        assert(M_values.size() == M_rows.size());
        assert(M_values.size() == M_cols.size());
        for (index_t i = 0; i < M_values.size(); ++i)
            *g_++ = X(M_rows(i), M_cols(i));
    }
    /// Gradient-vector product of constraints.
    void eval_grad_g_prod(const real_t *, const real_t *y_, real_t *gr_) const {
        auto G = mvec{gr_, m * n}.reshaped(m, n);
        G.setZero();
        assert(M_values.size() == M_rows.size());
        assert(M_values.size() == M_cols.size());
        for (index_t i = 0; i < M_values.size(); ++i)
            G(M_rows(i), M_cols(i)) = *y_++;
    }

    void initialize_box_D(real_t *lb_, real_t *ub_) const {
        mvec{lb_, M_values.size()} = M_values;
        mvec{ub_, M_values.size()} = M_values;
    }

  private:
    /// Loads problem data from a CSV file.
    /// The first row contains the number of rows of M, the number of columns
    /// of M, and the number of observed elements of M.
    /// The second row contains the row indices of the observed values,
    /// the third row contains the column indices of the observed values,
    /// and the fourth row contains the values themselves.
    void load_data(fs::path csv_filename);
#if WITH_PYTHON
    void load_data(py::kwargs kwargs);
#endif
    void init() { nuclear_norm = {1, m, n}; }
    /// Initialize the @ref funcs struct.
    void config_funcs();

  public:
    [[nodiscard]] std::string get_name() const {
        if (!data_file.empty())
            return "alpaqa-matrix-completion ('" + data_file.string() + "', " +
                   std::to_string(m) + "×" + std::to_string(n) + ":" +
                   std::to_string(M_values.size()) + ")";
        else
            return "alpaqa-matrix-completion (NumPy, " + std::to_string(m) +
                   "×" + std::to_string(n) + ":" +
                   std::to_string(M_values.size()) + ")";
    }

    void initialize(fs::path csv_filename) {
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

void Problem::config_funcs() {
    name    = get_name();
    using P = Problem;
    using alpaqa::member_caller;
    funcs.n                       = m * n;           // number of unknowns
    funcs.m                       = M_values.size(); // number of constraints
    funcs.name                    = name.c_str();
    funcs.eval_objective_gradient = member_caller<&P::eval_grad_f>();
    funcs.eval_objective          = member_caller<&P::eval_f>();
    funcs.eval_objective_and_gradient = member_caller<&P::eval_f_grad_f>();
    funcs.eval_proximal_gradient_step =
        member_caller<&P::eval_prox_grad_step>();
    funcs.eval_constraints = member_caller<&P::eval_g>();
    funcs.eval_constraints_gradient_product =
        member_caller<&P::eval_grad_g_prod>();
    funcs.initialize_box_D = member_caller<&P::initialize_box_D>();
}

void Problem::load_data(fs::path csv_file) {
    data_file = std::move(csv_file);
    std::ifstream ifile{data_file};
    if (!ifile)
        throw std::runtime_error("Unable to open file '" + data_file.string() +
                                 "'");
    // Load dimensions (#observations, #features, #targets, #terms)
    auto dims = alpaqa::csv::read_row_std_vector<length_t>(ifile);
    if (dims.size() != 3)
        throw std::runtime_error("Invalid problem dimensions in data file \'" +
                                 data_file.string() + '\'');
    m      = dims[0];
    n      = dims[1];
    auto p = dims[2];
    M_rows.resize(p);
    alpaqa::csv::read_row(ifile, M_rows);
    M_cols.resize(p);
    alpaqa::csv::read_row(ifile, M_cols);
    M_values.resize(p);
    alpaqa::csv::read_row(ifile, M_values);
}

#if WITH_PYTHON
void Problem::load_data(py::kwargs kwargs) {
    m        = py::cast<length_t>(kwargs.attr("pop")("m"));
    n        = py::cast<length_t>(kwargs.attr("pop")("n"));
    M_values = py::cast<decltype(M_values)>(kwargs.attr("pop")("M_values"));
    M_rows   = py::cast<decltype(M_rows)>(kwargs.attr("pop")("M_rows"));
    M_cols   = py::cast<decltype(M_cols)>(kwargs.attr("pop")("M_cols"));
    if (!py::cast<py::dict>(kwargs).empty())
        throw std::invalid_argument(
            "Unknown arguments: " +
            py::cast<std::string>(py::str(", ").attr("join")(kwargs)));
}
#endif

} // namespace amc
