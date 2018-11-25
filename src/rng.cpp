
#include <random>

#include "cnn/rng.hpp"

namespace cnn
{

std::default_random_engine g_generator;

void set_seed(int val)
{
    g_generator.seed(val);
}

// refer to
// http://www.cplusplus.com/reference/random/uniform_int_distribution/
int uniform(int low, int high)
{
    std::uniform_int_distribution<int> distribution(low, high);
    return distribution(g_generator);
}

// refer to
// http://www.cplusplus.com/reference/random/normal_distribution/normal_distirbution/
double gaussian(double mean, double stddev)
{
    std::normal_distribution<double> distribution(mean, stddev);
    return distribution(g_generator);
}


// refer to
// http://www.cplusplus.com/reference/random/bernoulli_distribution/bernoulli_distribution/
bool bernoulli(double p)
{
    std::bernoulli_distribution distribution(p);
    return distribution(g_generator);
}

}  // namespace cnn
