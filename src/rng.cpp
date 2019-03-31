/*  ---------------------------------------------------------------------
  Copyright 2018-2019 Fangjun Kuang
  email: csukuangfj at gmail dot com
  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a COPYING file of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>
  -----------------------------------------------------------------  */

#include <random>

#include "cnn/rng.hpp"

namespace cnn {

std::default_random_engine g_generator;

void set_seed(int val) { g_generator.seed(val); }

// refer to
// http://www.cplusplus.com/reference/random/uniform_int_distribution/
int uniform(int low, int high) {
  std::uniform_int_distribution<int> distribution(low, high);
  return distribution(g_generator);
}

// refer to
// http://www.cplusplus.com/reference/random/normal_distribution/normal_distirbution/
double gaussian(double mean, double stddev) {
  std::normal_distribution<double> distribution(mean, stddev);
  return distribution(g_generator);
}

// refer to
// http://www.cplusplus.com/reference/random/bernoulli_distribution/bernoulli_distribution/
bool bernoulli(double p) {
  std::bernoulli_distribution distribution(p);
  return distribution(g_generator);
}

}  // namespace cnn
