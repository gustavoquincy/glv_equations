#include "glv_equation.h"

struct larger_than_zero
{
    bool operator()(const value_type x) { return x > 0; }
}

struct generalized_lotka_volterra_system
{
    generalized_lotka_volterra_system( size_t num_species ): m_num_species( num_species ) { }

    struct generalized_lotka_volterra_functor
    {
        void operator( Tuple t ) /* tuple t = { y, dydt, growth_rate, Sigma, interaction column } */
        {   
            thrust::device_vector<value_type> result( m_num_species );
            thrust::transform( y.begin(), y.end(), thrust::get<4>(t).begin(), result.begin(), thrust::multiplies<value_type>());
            thrust::device_vector<value_type> copy_result( m_num_species );
            thrust::fill( copy_result.begin(), copy_result.end(), 0);
            thrust::copy_if( result.begin(), result.end(), copy_result.begin(), larger_than_zero());
            value_type m_pos_sum = thrust::reduce( copy_result.begin(), copy_result.end(), 0 );
            thrust::fill( copy_result.begin(), copy_result.end(), 0);
            thrust::copy_if( result.begin(), result.end(), copy_result.begin(), !larger_than_zero());
            value_type m_neg_sum = thrust::reduce( copy_result.begin(), copy_result.end(), 0 );
            thrust::get<1>(t) = thrust::get<0>(t) * thrust::get<2>(t) * ( 1 + m_neg_sum + thrust::get<3>(t) * m_pos_sum / ( 1 + m_pos_sum )) - m_dilution * thrust::get<0>(t);
        }
    };


    void operator ( state_type y , state_type dydt, state_type &growth_rate, state_type &Sigma, matrix_type &interaction )
    {
        thrust::for_each(
                thrust::make_zip_iterator( thrust::make_tuple( y.begin(), dydt.begin(), growth_rate.begin(), Sigma.begin(), interaction.begin() ) ),
                thrust::make_zip_iterator( thrust::make_tuple( y.end(), dydt.end(), growth_rate.end(), Sigma.end(), interaction.end() ) ),
                generalized_lotka_volterra_functor()
        );

    };
    
    state_type get_growth_rate() { return m_growth_rate; }

    void set_growth_rate( state_type growth_rate ) { thrust::copy( growth_rate.begin(), growth_rate.end(), m_growth_rate.begin() ); }

    value_type get_dilution() { return m_dilution; }

    void set_dilution( value_type dilution ) { m_dilution = dilution; }

    value_type get_Sigma() { return m_Sigma; }

    void set_Sigma( state_type Sigma ) { thrust::copy( Sigma.begin(), Sigma.end(), m_Sigma.begin() ); }

    value_type get_interaction() { return m_interaction; }

    void set_interaction( matrix_type interaction ) { thrust::copy( interaction.begin(), interaction.end(), m_interaction.begin() ); }



}


const state_type growth_rate;
const state_type Sigma;
const matrix_type interaction;
const value_type dilution;
