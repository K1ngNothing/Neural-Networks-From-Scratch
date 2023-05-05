#include <random>

namespace model {

static std::mt19937_64& GetRNG() {
    static std::mt19937_64 rng(42);
    return rng;
}

}  // namespace model