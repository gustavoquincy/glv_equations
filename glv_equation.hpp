#include <iostream>
#include <vector>
#include <cmath>

#include <omp.h>
#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/external/openmp/openmp.hpp>

typedef std::vector<double_t> state_type;
typedef runge_kutta_dopri5<state_type, double_t, state_type, double_t, openmp_range_algebra> stepper_type;

state_type equations(state_type, size_t, std::vector<double_t> /* growth rate */, std::vector<double_t> /* Sigma */, std::vector<std::vector<double_t>> /* interaction matrix */, double_t /* dilution */);

