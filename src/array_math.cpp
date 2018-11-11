
#include <random>

#include "cnn/array.hpp"
#include "cnn/array_math.hpp"

namespace cnn
{

static std::default_random_engine g_generator;

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
template<typename Dtype>
void gaussian(Array<Dtype>* arr, Dtype mean, Dtype stddev)
{
    std::normal_distribution<Dtype> distribution(mean, stddev);
    for (int i = 0; i < arr->total_; i++)
    {
        arr->d_[i] = distribution(g_generator);
    }
}

template
void gaussian(Array<double>* arr, double mean, double stddev);

template
void gaussian(Array<float>* arr, float mean, float stddev);
}  // namespace cnn
