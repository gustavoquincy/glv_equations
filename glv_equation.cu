#include "glv_equation.h"

struct larger_than_zero
{   
    bool operator(const value_type x) { return x > 0; }
};

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
            // steps above for derivation of m_pos_sum and m_neg_sum
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



};

// generator for random variable of uniform distribution U(0, 1)
struct uniform_gen {
    double_t operator()() {
        pcg64 rng(pcg_extras::seed_seq_from<std::random_device{});
        // make a random number engine, use the 64-bit generator, 2^128 period, 2^127 streams
        std::uniform_real_distribution<double_t> uniform_dist(0, 1.0);
        return uniform_dist(rng);
    }
};

int main() {
    const size_t num_species = 10;
    // initalize parameters, set the number of species to 10 in the generalized lv equation

    const size_t outerloop = 200;  
    // randomization for growth_rate, Sigma, interaction and dilution

    const size_t innerloop = 500;
    // randomization for initial condition of glv ODE

    state_type growth_rate(num_species * outerloop), Sigma(num_species * outerloop), dilution(1 * outerloop), interaction(num_species * num_species * outerloop), initial(num_species * outerloop * innerloop);
    // thrust.fill(v.begin(), v.end(), value)
    thrust::generate(growth_rate.begin(), growth_rate.end(), uniform_gen());
    thrust::generate(Sigma.begin(), Sigma.end(), uniform_gen());
    thrust::generate(dilution.begin(), dilution.end(), uniform_gen());
    thrust::generate(interaction.begin(), interaction.end(), uniform_gen());
    thrust::generate(initial.begin(), initial.end(), uniform_gen());
    //suppose here we have randomized the parameters

    /*
    TODO: might be of use
    ```
    include "curand.h"
    curandStatus_t
    curandGenerateUniformDouble(
        curandGenerator_t generator, 
        double *outputPtr, size_t num)
    ```
    */

    std::uniform_real_distribution<value_type> uniform_real_dist(0, 1.0);
    std::uniform_int_distribution<size_t> uniform_int_dist(2, 5);


    // TODO: solve ODE


    // TODO: parse results with Euclidean distance aka 2-norm



    return 0;
}