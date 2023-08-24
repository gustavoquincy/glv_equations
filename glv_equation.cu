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

typedef double_t value_type;
typedef thrust::host_vector< value_type > host_type;
typedef thrust::device_vector< value_type > state_type;
typedef runge_kutta_dopri5< state_type, value_type, state_type, value_type > stepper_type;

struct larger_than_zero
{   
    __host__ __device__
    bool operator()(const value_type x) { return x > 0; }
};

struct generalized_lotka_volterra_system
{
    generalized_lotka_volterra_system( size_t num_species ): m_num_species( num_species ) { }

    struct generalized_lotka_volterra_functor
    {
        template< class Tuple >
        __host__ __device__
        void operator()( Tuple t ) /* tuple t = { y, dydt, growth_rate, Sigma, interaction column } */
        {   
            thrust::device_vector<value_type> result( m_num_species );
            thrust::transform( y.begin(), y.end(), thrust::get<4>(t).begin(), result.begin(), thrust::multiplies<value_type>());
            thrust::device_vector<value_type> copy_result( m_num_species );
            thrust::fill( copy_result.begin(), copy_result.end(), 0);
            thrust::copy_if( result.begin(), result.end(), copy_result.begin(), larger_than_zero());
            value_type m_pos_sum = thrust::reduce( copy_result.begin(), copy_result.end(), 0.0 );
            thrust::fill( copy_result.begin(), copy_result.end(), 0);
            thrust::copy_if( result.begin(), result.end(), copy_result.begin(), !larger_than_zero());
            value_type m_neg_sum = thrust::reduce( copy_result.begin(), copy_result.end(), 0.0 );
            // steps above for derivation of m_pos_sum and m_neg_sum
            thrust::get<1>(t) = thrust::get<0>(t) * thrust::get<2>(t) * ( 1 + m_neg_sum + thrust::get<3>(t) * m_pos_sum / ( 1 + m_pos_sum )) - m_dilution * thrust::get<0>(t);
        }
    };


    void operator()( state_type& y , state_type& dydt, state_type growth_rate, state_type Sigma, state_type interaction )
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

    void set_dilution( value_type dilution ) { thrust::copy( dilution.begin(), dilution.end(), m_dilution.begin() ); }

    value_type get_Sigma() { return m_Sigma; }

    void set_Sigma( state_type Sigma ) { thrust::copy( Sigma.begin(), Sigma.end(), m_Sigma.begin() ); }

    value_type get_interaction() { return m_interaction; }

    void set_interaction( matrix_type interaction ) { thrust::copy( interaction.begin(), interaction.end(), m_interaction.begin() ); }

};

// generator for random variable of uniform distribution U(a, b)
struct uniform_gen {
    uniform_gen(value_type a, value_type b): m_a(a), m_b(b) {}
    
    __host__
    value_type operator()() {
        pcg64 rng(pcg_extras::seed_seq_from<std::random_device{});
        // make a random number engine, use the 64-bit generator, 2^128 period, 2^127 streams
        std::uniform_real_distribution<double_t> uniform_dist(m_a, m_b);
        return uniform_dist(rng);
    }

    value_type m_a, m_b;
};

struct set_growthrate
{ 
    template<class Tuple >
    __host__
    void opeartor()( Tuple& t ) {
        thrust::get<3>(t) = thrust::get<0>(t) - thrust::get<1>(t) + 2 * thrust::get<1>(t) * thrust::get<2>(t); // t = { growth_rate_mean, growth_rate_width, unit_random_vec, growth_rate}
    }
};

struct is_below_promote_density
{   
    template<class Tuple >
    __host__
    bool operator()( Tuple t ) /* t = { 0 threshold_vector, 1 promote_dense, 2 compete_dense, 3 promote_mean, 4 promote_width, 5 compete_mean, 6 compete_width, 7 unit_random_vec, 8 interaction } (arity = 9)*/
    {
        return thrust::get<0>(t) <= thrust::get<1>(t);
    }
};

struct is_above_compete_density
{
    template<class Tuple >
    __host__
    bool opeartor()( Tuple t )
    {
        return thrust::get<0>(t) >= thrust::get<2>(t);
    }
};

struct set_promote_value
{
    template<class Tuple >
    __host__
    void operator()( Tuple& t )
    {
        thrust::get<8>(t) = thrust::get<3>(t) - thrust::get<4>(t) + 2 * thrust::get<4>(t) * thrust::get<7>(t);
    }
};

struct set_compete_value
{
    template<class Tuple >
    __host__
    void operator()( Tuple& t )
    {
        thrust::get<8>(t) = -1 * (thrust::get<5>(t) - thrust::get<6>(t) + 2 * thrust::get<6>(t) * thrust::get<7>(t));
    }
};

struct is_diagonal
{
    is_diagonal(size_t num_species): m_num_species(num_species) {}

    template<class Tuple >
    __host__
    bool operator()( Tuple t ) /* t = { index, interaction }*/
    {
        return thrust::get<0>(t) % (m_num_species + 1) == 1;
    }

    size_t m_num_species;
};

struct set_minus_one
{
    template<class Tuple >
    __host__
    void operator()( Tuple& t ) {
        thrust::get<1>(t) = -1.0;
    }
};

struct set_dilution
{
    set_dilution(value_type growth_rate_mean): m_growth_rate_mean(growth_rate_mean) {}

    __host__
    void operator()(value_type& di) {
        host_type random_vec_a(1), random_vec_b(1);
        thrust::generate(random_vec_a.begin(), random_vec_a.end(), uniform_gen(0, m_growth_rate_mean));
        thrust::generate(random_vec_b.begin(), random_vec_b.end(), uniform_gen(0, 0.3));
        di = random_vec_a[0] < random_vec_b[0] ? random_vec_a[0] :random_vec_b[0];
    }

    value_type m_growth_rate_mean;
};

struct normalize
{
    normalize(value_type normalized_by): m_normalized_by(normalized_by) {}
    
    __host__ __device__
    void operator()(value_type& x) {
        x /= m_normalized_by;
    }

    value_type m_normalized_by;
}

const size_t num_species = 10;
// initalize parameters, set the number of species to 10 in the generalized lv equation

const size_t outerloop = 200;  
// randomization for growth_rate, Sigma, interaction and dilution

const size_t innerloop = 500;
// randomization for initial condition of glv ODE

int main() {

    host_type growth_rate_host(num_species * outerloop)/* copy innerloop times */, Sigma_host(num_species * outerloop)/* copy innerloop times */, dilution_host(1 * outerloop) /* copy num_species*innerloop times */, interaction_host(num_species * num_species * outerloop) /* copy innerloop times */, initial_host(num_species * outerloop * innerloop);
    // outerloop - samplesize / innerloop - precision
    // randomize growth rate
    size_t dim = growth_rate_host.size();
    host_type growth_rate_mean(dim), growth_rate_width(dim), unit_random_vec(dim);
    host_type growth_rate_mean_host(1);
    thrust::generate(growth_rate_mean_host.begin(), growth_rate_mean_host.end(), uniform_gen(0.1, 1.5));
    thrust::fill(growth_rate_mean.begin(), groth_rate_mean.end(), growth_rate_mean_host[0]);
    host_type growth_rate_width_host(1);
    thrust::generate(growth_rate_width_host.begin(), growth_rate_width_host.end(), uniform_gen(0, growth_rate_mean_host[0]));
    thrust::fill(growth_rate_width.begin(), growth_rate_width.end(), growth_rate_width_host[0]);
    thrust::generate(unit_random_vec.begin(), unit_random_vec.end(), uniform_gen(0, 1.0));
    thrust::for_each( 
        thrust::make_zip_iterator( thrust::make_tuple( growth_rate_mean.begin(), growth_rate_width.begin(), unit_random_vec.begin(), growth_rate_host.begin() )),
        thrust::make_zip_iterator( thrust::make_tuple( growth_rate_mean.end(), growth_rate_width.end(), unit_random_vec.end(), growth_rate_host.end() )),
        set_growthrate() 
    );
    // randomize interaction
    size_t dim = interaction_host.size()
    host_type compete_dense(1), promote_dense(1);
    thrust::generate(compete_dense.begin(), compete_dense.end(), uniform_gen(0.5, 1.0));
    thrust::generate(promote_dense.begin(), promote_dense.end(), uniform_gen(0, 1 - compete_dense[0]));
    host_type promote_mean(dim), promote_width(dim), compete_mean(dim), compete_width(dim);
    host_type promote_mean_host(1), promote_width_host(1), compete_mean_host(1), compete_width_host(1);
    thrust::generate(compete_mean_host.begin(), compete_mean_host.end(), uniform_gen(0.5, 2.0)); 
    thrust::fill(compete_mean.begin(), compete_mean.end(), compete_mean_host[0]); 
    thrust::generate(promote_mean_host.begin(), promote_mean_host.end(), uniform_gen(0.01, 1.0));
    thrust::fill(promote_mean.begin(), promote_mean.end(), promote_mean_host[0]);
    thrust::generate(compete_width_host.begin(), compete_width_host.end(), uniform_gen(0, compete_mean_host[0]));
    thrust::fill(compete_width.begin(), compete_width.end(), compete_width_host[0]);
    thrust::generate(promote_width_host.begin(), promote_width_host.end(), uniform_gen(0, promote_mean_host[0]));
    thrust::fill(promote_width.begin(), promote_width.end(), promote_width_host[0]); 
    // generate once, then fill the device vector
    host_type threshold_vector(dim), unit_random_vec(dim);
    thrust::generate(threshold_vector.begin(), threshold_vector.end(), uniform_gen(0, 1.0));
    thrust::generate(unit_random_vec.begin(), unit_random_vec.end(), uniform_gen(0, 1.0));
    thrust::transform_if( 
        thrust::make_zip_iterator( thrust::make_tuple( threshold_vector.begin(), promote_dense.begin(), compete_dense.begin(), promote_mean.begin(), promote_width.begin(), compete_mean.begin(), compete_width.begin(), unit_random_vec.begin(), interaction_host.begin() )),
        thrust::make_zip_iterator( thrust::make_tuple( threshold_vector.end(), promote_dense.end(), compete_dense.end(), promote_mean.end(), promote_width.end(), compete_mean.end(), compete_width.end(), unit_random_vec.end(), interaction_host.end() )),
        thrust::make_zip_iterator( thrust::make_tuple( threshold_vector.begin(), promote_dense.begin(), compete_dense.begin(), promote_mean.begin(), promote_width.begin(), compete_mean.begin(), compete_width.begin(), unit_random_vec.begin(), interaction_host.begin() )),
        is_below_promote_density(),
        set_promote_value() 
    );
    thrust::transform_if( 
        thrust::make_zip_iterator( thrust::make_tuple( threshold_vector.begin(), promote_dense.begin(), compete_dense.begin(), promote_mean.begin(), promote_width.begin(), compete_mean.begin(), compete_width.begin(), unit_random_vec.begin(), interaction_host.begin() )),
        thrust::make_zip_iterator( thrust::make_tuple( threshold_vector.end(), promote_dense.end(), compete_dense.end(), promote_mean.end(), promote_width.end(), compete_mean.end(), compete_width.end(), unit_random_vec.end(), interaction_host.end() )),
        thrust::make_zip_iterator( thrust::make_tuple( threshold_vector.begin(), promote_dense.begin(), compete_dense.begin(), promote_mean.begin(), promote_width.begin(), compete_mean.begin(), compete_width.begin(), unit_random_vec.begin(), interaction_host.begin() )),
        is_above_compete_density(),
        set_compete_value() 
    );
    thrust::host_vector<size_t> index(dim);
    thrust::sequence(index.begin(), index.end(), 1);
    thrust::transform_if( 
        thrust::make_zip_iterator( thrust::make_tuple( index.begin(), interaction_host.begin() )), 
        thrust::make_zip_iterator( thrust::make_tuple( index.end(), interation_host.end() )), 
        thrust::make_zip_iterator( thrust::make_tuple( index.begin(), interaction_host.begin() )), 
        set_minus_one(),
        is_diagonal(num_species) 
    );
    // randomize Sigma
    thrust::generate(Sigma_host.begin(), Sigma_host.end(), uniform_gen(0, 0.5));
    // randomize dilution
    thrust::for_each(dilution_host.begin(), dilution_host.end(), set_dilution(growth_rate_mean_host[0]));
    // randomize initial
    thrust::generate(initial_host.begin(), initial_host.end(), uniform_gen(0, 1.0));
    value_type initial_sum = thrust::reduce(initial_host.begin(), initial_host.end(), 0.0);
    thrust::for_each(initial_host.begin(), initial_host.end(), normalize(initial_sum));

    state_type growth_rate = growth_rate_host;
    state_type interaction = interaction_host;
    state_type Sigma = Sigma_host;
    state_type dilution = dilution_host;
    state_type initial = initial_host;
    // TODO: use curand to generate random numbers on device w/ curand kernel

    // TODO: solve ODE
    generalized_lotka_volterra_system glv( num_species );
    glv.set_growth_rate(growth_rate);
    glv.set_Sigma(Sigma);
    glv.set_interaction(interaction);
    glv.set_dilution(dilution);
    integrate_adaptive( make_controlled(1.0e-6, 1.0e-6, stepper_type()), glv, )

    // TODO: parse results with Euclidean distance aka 2-norm



    return 0;
}