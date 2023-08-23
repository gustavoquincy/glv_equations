#ifndef glv_equation.h
#define glv_equation.h

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>

#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/external/thrust/thrust.hpp>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>

#include "pcg_random.hpp"
//using pcg c++ implementation, pcg64, compilation requires -std=c++11 flag
#include "curand.h"

using namespace std;
using namespace boost::numeric::odeint;

typedef double_t value_type;
typedef thrust::device_vector< value_type > state_type;
typedef thrust::device_vector< device_vector < value_type >> matrix_type;
typedef runge_kutta_dopri5< state_type, value_type, state_type, value_type > stepper_type;
// TODO: use this

struct larger_than_zero
{
    __host__ __device__
    bool operator(const value_type x); 
}

struct generalized_lotka_volterra_system
{
    struct generalized_lotka_volterra_functor
    {
        template< class Tuple >
        __host__ __device__
        void operator()( Tuple t ) const
    };

    generalized_lotka_volterra_system( size_t num_species ): m_num_species( num_species ); // num_species being passed to m_num_species

    void operator(state_type , state_type, state_type, state_type, matrix_type );

    state_type get_growth_rate();

    void set_growth_rate( state_type );

    value_type get_dilution();

    void set_dilution( value_type );

    value_type get_Sigma();

    void set_Sigma( state_type );

    value_type get_interaction();

    void set_interaction( matrix_type );

    size_t m_num_species;
    //m_num_species registered in struct, but not involved in any arithmetic
    
    state_type m_Sigma, m_growth_rate;
    
    value_type m_dilution;

    matrix_type m_interaction;
};

struct uniform_gen
{
    __host__
    uniform_gen();

    operator();z    
}

    
#endif //glv_equation.h