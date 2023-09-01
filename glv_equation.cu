#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>

#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/external/thrust/thrust.hpp>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>


#include "pcg-cpp-0.98/include/pcg_random.hpp"
//using pcg c++ implementation, pcg64, compilation requires -std=c++11 flag

typedef double_t value_type;
typedef thrust::host_vector< value_type > host_type;
typedef thrust::device_vector< value_type > state_type;

struct generalized_lotka_volterra_system
{
    const size_t m_num_species, m_innerloop, m_outerloop;
    state_type m_growth_rate, m_Sigma, m_interaction, m_dilution;
    state_type interaction_column, growth_rate_i, Sigma_i, interaction_i, dilution_ni;  
    
    // m_growth_rate(num_species * outerloop)/* copy innerloop times */, m_Sigma(num_species * outerloop)/* copy innerloop times */, m_dilution(1 * outerloop) /* copy num_species*innerloop times */, m_interaction(num_species * num_species * outerloop) /* copy innerloop times */ 

    generalized_lotka_volterra_system( size_t num_species, size_t innerloop, size_t outerloop, state_type growth_rate, state_type Sigma, state_type interaction, state_type dilution )
    : m_num_species(num_species), m_innerloop(innerloop), m_outerloop(outerloop), m_growth_rate(growth_rate), m_Sigma(Sigma), m_interaction(interaction), m_dilution(dilution) {
        state_type growth_rate_i_scoped( m_growth_rate.size() * m_innerloop );
        state_type Sigma_i_scoped( m_Sigma.size() * m_innerloop );
        state_type dilution_i_scoped( m_dilution.size() * m_innerloop );
        state_type interaction_i_scoped( m_interaction.size() * m_innerloop );
        for (int i = 0; i < m_innerloop; ++i) {
            thrust::copy(m_growth_rate.begin(), m_growth_rate.end(), growth_rate_i_scoped.begin() + i * m_growth_rate.size());
            thrust::copy(m_Sigma.begin(), m_Sigma.end(), Sigma_i_scoped.begin() + i * m_Sigma.size());
            thrust::copy(m_dilution.begin(), m_dilution.end(), dilution_i_scoped.begin() + i * m_dilution.size());
            thrust::copy(m_interaction.begin(), m_interaction.end(), interaction_i_scoped.begin() + i * m_interaction.size());
        }
        growth_rate_i = growth_rate_i_scoped;
        Sigma_i = Sigma_i_scoped;
        interaction_i = interaction_i_scoped;
        state_type dilution_ni_scoped( dilution_i_scoped.size() * m_num_species );
        state_type interaction_column_scoped( m_num_species * m_innerloop * m_outerloop );
        for (int i = 0; i < m_num_species; ++i) {
            thrust::copy(dilution_i_scoped.begin(), dilution_i_scoped.end(), dilution_ni_scoped.begin() + i * dilution_i_scoped.size());
            for (int j = 0; j < m_num_species * m_innerloop * m_outerloop; ++j) {
                interaction_column[j] = m_interaction[ i + m_num_species * j ];
            }     
        }
        dilution_ni = dilution_ni_scoped;
        interaction_column = interaction_column_scoped;
    }

    struct generalized_lotka_volterra_functor
    {
        generalized_lotka_volterra_functor( value_type pos_sum, value_type neg_sum ): m_pos_sum(pos_sum), m_neg_sum(neg_sum) { }

        template< class Tuple >
        __host__ __device__
        void operator()( Tuple t ) /* tuple t = { y, dydt, growth_rate, Sigma, dilution } */
        {   
            thrust::get<1>(t) = thrust::get<0>(t) * thrust::get<2>(t) * ( 1 + m_neg_sum + thrust::get<3>(t) * m_pos_sum / ( 1 + m_pos_sum )) - thrust::get<4>(t) * thrust::get<0>(t);
        }

        value_type m_pos_sum, m_neg_sum;
    };

    struct larger_than_zero
    {   
        __host__ __device__
        bool operator()(value_type x) { return x > 0; }
    };

    struct smaller_than_zero
    {
        __host__ __device__
        bool operator()(value_type x) { return x < 0; }
    };

    void operator()( state_type& y , state_type& dydt, value_type t)
    {
        state_type result(y.size());
        thrust::transform(y.begin(), y.end(), interaction_column.begin(), result.begin(), thrust::multiplies<value_type>());
        state_type copy_result(y.size());
        thrust::fill(copy_result.begin(), copy_result.end(), 0.0);
        thrust::copy_if(result.begin(), result.end(), copy_result.begin(), larger_than_zero());
        value_type possum = thrust::reduce(copy_result.begin(), copy_result.end(), 0.0);
        thrust::fill(copy_result.begin(), copy_result.end(), 0.0);
        thrust::copy_if(result.begin(), result.end(), copy_result.begin(), smaller_than_zero());
        value_type negsum = thrust::reduce(copy_result.begin(), copy_result.end(), 0.0);
        thrust::for_each(
                thrust::make_zip_iterator( thrust::make_tuple( y.begin(), dydt.begin(), growth_rate_i.begin(), Sigma_i.begin(), dilution_ni.begin() ) ),
                thrust::make_zip_iterator( thrust::make_tuple( y.end(), dydt.end(), growth_rate_i.end(), Sigma_i.end(), dilution_ni.begin() ) ),
                generalized_lotka_volterra_functor(possum, negsum)
        );
        std::clog << t << "\n";
    }

};

// generator for random variable of uniform distribution U(a, b)
struct uniform_gen
{
    uniform_gen(value_type a, value_type b): m_a(a), m_b(b) {}
    
    __host__
    value_type operator()() {
        pcg64 rng(pcg_extras::seed_seq_from<std::random_device>{});
        // make a random number engine, use the 64-bit generator, 2^128 period, 2^127 streams
        std::uniform_real_distribution<double_t> uniform_dist(m_a, m_b);
        return uniform_dist(rng);
    }

    value_type m_a, m_b;
};

struct set_growthrate
{ 
    template<class T >
    __host__
    void operator()( T& t ) {
        thrust::get<3>(t) = thrust::get<0>(t) - thrust::get<1>(t) + 2 * thrust::get<1>(t) * thrust::get<2>(t); // t = { growth_rate_mean, growth_rate_width, unit_random_vec, growth_rate}
    }
};

struct is_below_promote_density
{   
    template<class T >
    __host__
    bool operator()( T t ) /* t = { 0 threshold_vector, 1 promote_dense, 2 compete_dense, 3 promote_mean, 4 promote_width, 5 compete_mean, 6 compete_width, 7 unit_random_vec, 8 interaction } (arity = 9)*/
    {
        return thrust::get<0>(t) <= thrust::get<1>(t);
    }
};

struct is_above_compete_density
{
    template<class T >
    __host__
    bool operator()( T t )
    {
        return thrust::get<0>(t) >= thrust::get<2>(t);
    }
};

struct set_promote_value
{
    template<class T >
    __host__
    T operator()( T t )
    {
        thrust::get<8>(t) = thrust::get<3>(t) - thrust::get<4>(t) + 2 * thrust::get<4>(t) * thrust::get<7>(t);
        return t;
    }
};

struct set_compete_value
{
    template<class T >
    __host__
    T operator()( T t )
    {
        thrust::get<8>(t) = -1 * (thrust::get<5>(t) - thrust::get<6>(t) + 2 * thrust::get<6>(t) * thrust::get<7>(t));
        return t;
    }
};

struct index_transform
{
    index_transform(size_t num_species): m_num_species(num_species) {
        m_counter = 0;
        m_i = 1;
    }

    __host__
    void operator()(size_t& idx)
    {
        bool is_diag = idx % (m_num_species + 1) == m_i;
        if ( is_diag ) m_counter += 1;
        if ( m_counter == m_num_species ) {
            m_i = (m_i + 1) % (m_num_species + 1);
            m_counter = 0;
        }
        idx = is_diag;
    }

    const size_t m_num_species;
    size_t m_i, m_counter;
};

struct set_minus_one
{
    template<class T >
    __host__
    T operator()( T t ) {
        thrust::get<1>(t) = -1.0;
        return t;
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
};

struct is_diagonal
{
    template<class T >
    __host__
    bool operator()(T t) /* t = { index, interaction }*/ {
        return thrust::get<0>(t);
    }
};

const size_t num_species = 10;
// initalize parameters, set the number of species to 10 in the generalized lv equation

const size_t outerloop = 200;  
// samplesize

const size_t innerloop = 500;
// precision

const value_type dt = 0.01;

int main() {

    host_type growth_rate_host(num_species * outerloop)/* copy innerloop times */, Sigma_host(num_species * outerloop)/* copy innerloop times */, dilution_host(1 * outerloop) /* copy num_species*innerloop times */, interaction_host(num_species * num_species * outerloop) /* copy innerloop times */, initial_host(num_species * outerloop * innerloop);

    // randomize growth rate start
    size_t dim = growth_rate_host.size();
    host_type growth_rate_mean_seed(1);
    thrust::generate(growth_rate_mean_seed.begin(), growth_rate_mean_seed.end(), uniform_gen(0.1, 1.5));
    state_type growth_rate_mean(dim);
    thrust::fill(growth_rate_mean.begin(), growth_rate_mean.end(), growth_rate_mean_seed[0]);
    host_type growth_rate_width_seed(1);
    thrust::generate(growth_rate_width_seed.begin(), growth_rate_width_seed.end(), uniform_gen(0, growth_rate_mean_seed[0]));
    state_type growth_rate_width(dim);
    thrust::fill(growth_rate_width.begin(), growth_rate_width.end(), growth_rate_width_seed[0]);
    host_type unit_random_vec_host(dim);
    thrust::generate(unit_random_vec_host.begin(), unit_random_vec_host.end(), uniform_gen(0, 1.0));
    state_type unit_random_vec = unit_random_vec_host;
    state_type growth_rate = growth_rate_host;
    thrust::for_each( 
        thrust::make_zip_iterator( thrust::make_tuple( growth_rate_mean.begin(), growth_rate_width.begin(), unit_random_vec.begin(), growth_rate.begin() )),
        thrust::make_zip_iterator( thrust::make_tuple( growth_rate_mean.end(), growth_rate_width.end(), unit_random_vec.end(), growth_rate.end() )),
        set_growthrate() 
    );
    // randomize growth rate end

    // randomize interaction start
    dim = interaction_host.size();
    host_type compete_dense_host(1), promote_dense_host(1);
    thrust::generate(compete_dense_host.begin(), compete_dense_host.end(), uniform_gen(0.5, 1.0));
    state_type compete_dense = compete_dense_host;
    thrust::generate(promote_dense_host.begin(), promote_dense_host.end(), uniform_gen(0, 1 - compete_dense[0]));
    state_type promote_dense = promote_dense_host;

    host_type promote_mean_seed(1), promote_mean_host(dim), promote_width_seed(1), promote_width_host(dim), compete_mean_seed(1), compete_mean_host(dim), compete_width_seed(1), compete_width_host(dim);
    thrust::generate(compete_mean_seed.begin(), compete_mean_seed.end(), uniform_gen(0.5, 2.0));
    thrust::fill(compete_mean_host.begin(), compete_mean_host.end(), compete_mean_seed[0]);
    state_type compete_mean = compete_mean_host;
    thrust::generate(promote_mean_seed.begin(), promote_mean_seed.end(), uniform_gen(0.01, 1.0));
    thrust::fill(promote_mean_host.begin(), promote_mean_host.end(), promote_mean_seed[0]);
    state_type promote_mean = promote_mean_host;
    thrust::generate(compete_width_seed.begin(), compete_width_seed.end(), uniform_gen(0, compete_mean_seed[0]));
    thrust::fill(compete_width_host.begin(), compete_width_host.end(), compete_width_seed[0]);
    state_type compete_width = compete_width_host;
    thrust::generate(promote_width_seed.begin(), promote_width_seed.end(), uniform_gen(0, promote_mean_seed[0]));
    thrust::fill(promote_width_host.begin(), promote_width_host.end(), promote_width_seed[0]);
    state_type promote_width = promote_width_host;

    host_type threshold_vector_host(dim);
    thrust::generate(threshold_vector_host.begin(), threshold_vector_host.end(), uniform_gen(0, 1.0));
    state_type threshold_vector = threshold_vector_host;
    thrust::generate(unit_random_vec_host.begin(), unit_random_vec_host.end(), uniform_gen(0, 1.0));
    unit_random_vec = unit_random_vec_host;

    state_type interaction = interaction_host;
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
    thrust::host_vector<size_t> index_host(dim);
    thrust::sequence(index_host.begin(), index_host.end(), 1);
    thrust::for_each(index_host.begin(), index_host.end(), index_transform(num_species));
    state_type index = index_host;
    thrust::transform_if( 
        thrust::make_zip_iterator( thrust::make_tuple( index.begin(), interaction.begin() )), 
        thrust::make_zip_iterator( thrust::make_tuple( index.end(), interaction.end() )), 
        thrust::make_zip_iterator( thrust::make_tuple( index.begin(), interaction.begin() )), 
        set_minus_one(),
        is_diagonal() 
    );
    // randomize interaction end

    // randomize Sigma
    thrust::generate(Sigma_host.begin(), Sigma_host.end(), uniform_gen(0, 0.5));
    state_type Sigma = Sigma_host;
    // randomize dilution
    state_type dilution = dilution_host;
    thrust::for_each(dilution.begin(), dilution.end(), set_dilution(growth_rate_mean_seed[0]));
    // randomize initial
    thrust::generate(initial_host.begin(), initial_host.end(), uniform_gen(0, 1.0));
    state_type initial = initial_host;
    value_type initial_sum = thrust::reduce(initial.begin(), initial.end(), 0.0);
    thrust::for_each(initial.begin(), initial.end(), normalize(initial_sum));
    // TODO: use curand to generate random numbers on device w/ curand kernel

    generalized_lotka_volterra_system glv_system( num_species, innerloop, outerloop, growth_rate, Sigma, interaction, dilution );
    integrate_adaptive( make_dense_output(1.0e-6, 1.0e-5, runge_kutta_dopri5< state_type, value_type, state_type, value_type >()), glv_system, initial, 0.0, 10.0, 0.01);

    // TODO: parse results with Euclidean distance aka 2-norm


    return 0;
}