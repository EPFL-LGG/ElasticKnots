#ifndef HELPERS_HH
#define HELPERS_HH

#include <ElasticRods/PeriodicRod.hh>

template<typename TIn, typename AllocIn, typename TOut, typename AllocOut>
static void castStdADVector(const std::vector<TIn, AllocIn> &in, std::vector<TOut, AllocOut> &out) {
    out.clear();
    out.reserve(in.size());
    for (const auto &val : in) out.push_back(autodiffCast<TOut>(val));
}

#endif