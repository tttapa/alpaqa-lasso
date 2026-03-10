#pragma once

#include <problem.hpp>

#include <guanaqo/lifetime.hpp>

#include <cassert>

namespace acl {

/// Convert complex matrix to real matrix.
inline auto c2r(crcmat in) {
    return cmmat{
        guanaqo::start_lifetime_as_array<real_t>(
            in.data(), static_cast<size_t>(in.size() * 2)),
        in.rows() * 2,
        in.cols(),
    };
}
/// Convert complex matrix to real matrix.
inline auto c2r(rcmat in) {
    return mmat{
        guanaqo::start_lifetime_as_array<real_t>(
            in.data(), static_cast<size_t>(in.size() * 2)),
        in.rows() * 2,
        in.cols(),
    };
}
/// Convert real matrix to complex matrix.
inline auto r2c(crmat in) {
    assert(in.size() % 2 == 0);
    return cmcmat{
        guanaqo::start_lifetime_as_array<cplx_t>(
            in.data(), static_cast<size_t>(in.size() / 2)),
        in.rows() / 2,
        in.cols(),
    };
}
/// Convert real matrix to complex matrix.
inline auto r2c(rmat in) {
    assert(in.size() % 2 == 0);
    return mcmat{
        guanaqo::start_lifetime_as_array<cplx_t>(
            in.data(), static_cast<size_t>(in.size() / 2)),
        in.rows() / 2,
        in.cols(),
    };
}

} // namespace acl
