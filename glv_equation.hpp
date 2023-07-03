#include <iostream>
#include <vector>
#include <cmath>

#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/external/thrust/thrust.hpp>

#include <boost/random/mersenne_twister.hpp>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>

using namespace std;
using namespace boost::numeric::odeint;

typedef double_t value_type;
typedef thrust::device_vector< value_type > state_type;
typedef thrust::device_vector< device_vector < value_type >> matrix_type;
typedef runge_kutta_dopri5< state_type, value_type, state_type, double_t, openmp_range_algebra> stepper_type;


state_type equations(state_type, size_t, state_type  /* growth rate */, state_type  /* Sigma */,  matrix_type /* interaction matrix */, value_type /* dilution */);

