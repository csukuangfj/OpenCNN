

#include <random>

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


}  // namespace cnn
