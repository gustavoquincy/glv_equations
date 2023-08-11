#include "glv_equation.h"

state_type equations(state_type y, size_t NumSpecies, state_type growth_rate, state_type Sigma, matrix_type interaction, value_type dilution)
{

}

const state_type growth_rate;
const state_type Sigma;
const matrix_type interaction;
const value_type dilution;

struct larger_than_zero
{
    bool operator()(const value_type x) { return x > 0; }
}

struct generalized_lotka_volterra_system
{
    struct generalized_lotka_volterra_functor
    {
        void operator( Tuple t ) /* tuple t = { y, dydt, growth_rate, Sigma, interaction column } */
        {   
            interaction[]
            state_type result = thrust::copy_if( )
            
            thrust::get<1>(t) = thrust::get<0>(t) * thrust::get<2>(t) * ( 1 + m_neg_sum + thrust::get<3>(t) * m_pos_sum / ( 1 + m_pos_sum )) - m_dilution * thrust::get<0>(t);
        }
    }

    void operator ( state_type &y , state_type &dydt, state_type growth_rate, state_type Sigma, matrix_type interaction, value_type dilution )
    {

        thrust::for_each(
                thrust::make_zip_iterator( thrust::make_tuple( y.begin(), dydt.begin() ) ),
                thrust::make_zip_iterator( thrust::make_tuple( y.end(), dydt.end() ) ),
                generalized_lotka_volterra_functor()
        )
    }
    
    state_type get_growth_rate() { return m_growth_rate; }

    void set_growth_rate( state_type growth_rate ) { thrust::copy( growth_rate.begin(), growth_rate.end(), m_growth_rate.begin() ); }

    value_type get_dilution() { return m_dilution; }

    void set_dilution( value_type dilution ) { m_dilution = dilution; }

    state_type 

}