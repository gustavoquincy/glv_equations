#ifndef glv_equation.h
#define glv_equation.h

#include <iostream>
#include <vector>
#include <cmath>

#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/external/thrust/thrust.hpp>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>

#include "pcg_random.hpp"
//using pcg c++ implementation, pcg64

using namespace std;
using namespace boost::numeric::odeint;

typedef double_t value_type;
typedef thrust::device_vector< value_type > state_type;
typedef thrust::device_vector< device_vector < value_type >> matrix_type;
typedef runge_kutta_dopri5< state_type, value_type, state_type, value_type > stepper_type;


state_type equations(state_type, size_t, state_type  /* growth rate */, state_type  /* Sigma */,  matrix_type /* interaction matrix */, value_type /* dilution */);

struct generalized_lotka_volterra_system
{
    struct generalized_lotka_volterra_functor
    {
        template< class Tuple >
        __host__ __device__
        void operator()( Tuple t ) const
    };

    generalized_lotka_volterra_system( size_t num_species ): m_num_species( num_species );

    template< class State, class Deriv >
    void operator()( const State &y, Deriv &dydt, value_type t) const

    size_t m_num_species;
};

    
#endif //glv_equation.h