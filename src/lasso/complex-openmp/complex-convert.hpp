#pragma once

#include <problem.hpp>

#include <alpaqa/util/lifetime.hpp>

namespace acl {

/// Convert complex matrix to real matrix.
inline auto c2r(crcmat in) {
    return cmmat{
        alpaqa::util::start_lifetime_as_array<real_t>(
            in.data(), static_cast<size_t>(in.size() * 2)),
        in.rows() * 2,
        in.cols(),
    };
}
/// Convert complex matrix to real matrix.
inline auto c2r(rcmat in) {
    return mmat{
        alpaqa::util::start_lifetime_as_array<real_t>(
            in.data(), static_cast<size_t>(in.size() * 2)),
        in.rows() * 2,
        in.cols(),
    };
}
/// Convert real matrix to complex matrix.
inline auto r2c(crmat in) {
    assert(in.size() % 2 == 0);
    return cmcmat{
        alpaqa::util::start_lifetime_as_array<cplx_t>(
            in.data(), static_cast<size_t>(in.size() / 2)),
        in.rows() / 2,
        in.cols(),
    };
}
/// Convert real matrix to complex matrix.
inline auto r2c(rmat in) {
    assert(in.size() % 2 == 0);
    return mcmat{
        alpaqa::util::start_lifetime_as_array<cplx_t>(
            in.data(), static_cast<size_t>(in.size() / 2)),
        in.rows() / 2,
        in.cols(),
    };
}

} // namespace acl
