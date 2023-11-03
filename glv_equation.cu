#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
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
#include <arrow/api.h>
#include <arrow/csv/api.h>
#include <arrow/io/api.h>
#include <arrow/compute/api.h>
#include <parquet/arrow/writer.h>
#include <arrow/util/type_fwd.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>
#include <omp.h>

#pragma GCC diagnostic ignored "-Wunused-result"

using namespace boost::numeric::odeint;

typedef double_t value_type;
typedef thrust::host_vector< value_type > host_type;
typedef thrust::device_vector< value_type > state_type;

__global__ __launch_bound__(1024) void setup_kernel(curandState *state, int seed)
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  /* Each thread gets device index seed, a different sequence number, no offset */
  curand_init(seed, id, 0, &state[id]);
}

__global__ __launch_bound__(1024) void initialize_parameters_growth_sigma(curandState *state, double_t *growth_rate, double_t *sigma, int sampleSize)
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  curandState localState = state[id];
// no dim 10**6
  if (id < sampleSize) {
    double_t growth_mean = 0.1 + 1.4 * curand_uniform_double(&localState);
    double_t growth_width = growth_mean * curand_uniform_double(&localstate);
    growth_rate[id] = growth_mean - growth_width + 2 * growth_width * curand_uniform_double(&localState);
    sigma[id] = 0.5 * curand_uniform_double(&localState);
  }
}

__global__ __launch_bound__(1024) void initialize_parameters_interaction(curandState *state, double_t *interaction, int sampleSize)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curandState localState = state[id];
    if (id < sampleSize) {
        // n2o dim 10**8
        double_t compete_mean = 0.5 + 1.5 * curand_uniform_double(&localState);
        double_t compete_dense = 0.5 + 0.5 * curand_uniform_double(&localState);
        double_t promote_mean = 0.01 + 0.99 * curand_uniform_double(&localState);
        double_t promote_dense = ( 1 -  compete_dense ) * curand_uniform_double(&localState);
        double_t compete_width = compete_mean * curand_uniform_double(&localState);
        double_t promote_width = promote_mean * curand_uniform_double(&localState);
        (curand_uniform(&localState) <= promote_dense) ? 
            interaction[id] = promote_mean - promote_width + 2 * promote_width * curand_uniform_double(&localState) : (curand_uniform(&localState) >= compete_dense) ? 
            interaction[id] = -1 * (compete_mean - compete_width + 2 * compete_width * curand_uniform_double(&localState)) : 0 ;
    }
}

__global__ __launch_bound__(1024) void initialize_parameters_dilution(curandState *state, double_t *dilution, int sampleSize)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curandState localState = state[id];
    if (id < sampleSize) {
        // i dim 10**3
        dilution[id] = std::min(growth_mean * curand_uniform_double(&localState), 0.3 * curand_uniform_double(&localState));
    }
}

__global__ __launch_bound__(1024) void initialize_initial(curandState *state, double_t *initial, int sampleSize)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curandState localState = state[id];
    if (id < sampleSize) {
        // noi dim 10**9
        initial[id] = curand_uniform_double(&localState);
    }
}

struct generalized_lotka_volterra_system
{
    const size_t m_num_species, m_innerloop, m_outerloop;
    state_type m_growth_rate, m_Sigma, m_interaction, m_dilution; //pass-in value
    state_type growth_rate_i, Sigma_i, interaction_i, dilution_ni;  //operator-use value
    
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
        for (int i = 0; i < m_num_species; ++i) {
            thrust::copy(dilution_i_scoped.begin(), dilution_i_scoped.end(), dilution_ni_scoped.begin() + i * dilution_i_scoped.size());
        }
        dilution_ni = dilution_ni_scoped;
    }

    struct generalized_lotka_volterra_functor
    {
        template< class Tuple >
        __host__ __device__
        void operator()( Tuple t )/* tuple t = { y, dydt, growth_rate, Sigma, dilution, pos_sum, neg_sum } (arity = 7)*/
        {   
            thrust::get<1>(t) = thrust::get<0>(t) * thrust::get<2>(t) * ( 1 + thrust::get<6>(t) + thrust::get<3>(t) * thrust::get<5>(t) / ( 1 + thrust::get<5>(t) )) - thrust::get<4>(t) * thrust::get<0>(t);
        }
    };

    void operator()( state_type& y , state_type& dydt, value_type t)
    {
        // copy y n times to make it n^2*io
        state_type y_n( y.size() * m_num_species );
        for (int i=0; i < m_num_species; ++i) {
            thrust::copy( y.begin(), y.end(), y_n.begin() + i * y.size() );
        }
        // multiply interaction with y piecewisely
        state_type result( interaction_i.size() );
        thrust::transform( y_n.begin(), y_n.end(), interaction_i.begin(), result.begin(), thrust::multiplies<value_type>() );
        // find pos_sum and neg_sum for every n in the result vector
        host_type result_host( result.size() );
        result_host = result;
        host_type pos_sum_host( m_num_species * m_innerloop * m_outerloop ), neg_sum_host( m_num_species * m_innerloop * m_outerloop );
        for (int i=0; i< m_num_species * m_innerloop * m_outerloop; ++i) {
            value_type pos_sum = 0.0;
            value_type neg_sum = 0.0;
            for (int j=0; j < m_num_species; ++j) {
                value_type vec_val = result_host[ i * m_num_species + j ];
                vec_val > 0 ? pos_sum += vec_val : neg_sum += vec_val;
            }
            pos_sum_host[i] = pos_sum;
            neg_sum_host[i] = neg_sum;
        }
        // then we have noi-dim pos_sum and noi-dim neg_sum
        state_type pos_sum( pos_sum_host.size() ), neg_sum( neg_sum_host.size() );
        pos_sum = pos_sum_host;
        neg_sum = neg_sum_host;

        thrust::for_each(
                thrust::make_zip_iterator( thrust::make_tuple( y.begin(), dydt.begin(), growth_rate_i.begin(), Sigma_i.begin(), dilution_ni.begin(), pos_sum.begin(), neg_sum.begin() ) ),
                thrust::make_zip_iterator( thrust::make_tuple( y.end(), dydt.end(), growth_rate_i.end(), Sigma_i.end(), dilution_ni.end(), pos_sum.end(), neg_sum.end() ) ),
                generalized_lotka_volterra_functor()
        );

        std::clog << "10 species abundance" << "\n";
        for (int i=0; i<y.size(); ++i) {
            std::clog << y[i] << std::endl;
            // store y
        }
        // write to arrow object once only
        std::clog << t << "\n";
    }

};

#pragma region
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
    __host__ __device__
    T operator()( T t ) {
        thrust::get<1>(t) = -1.0;
        return t;
    }
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
    __host__ __device__
    bool operator()(T t) /* t = { index, interaction }*/ {
        return thrust::get<0>(t);
    }
};
#pragma endregion

arrow::Status initial_condition_csv(double_t *growth_rate, double_t *Sigma, double_t *interaction, double_t *dilution , int64_t size) {
  arrow::DoubleBuilder doublebuilder;
  ARROW_RETURN_NOT_OK(doublebuilder.AppendValues(in, size));
  std::shared_ptr<arrow::Array> random_number;
  ARROW_ASSIGN_OR_RAISE(random_number, doublebuilder.Finish());
  std::shared_ptr<arrow::ChunkedArray> random_number_chunks = std::make_shared<arrow::ChunkedArray>(random_number);
  std::shared_ptr<arrow::Field> field_random_number;
  std::shared_ptr<arrow::Schema> schema;
  field_random_number = arrow::field("random_number", arrow::float64());
  schema = arrow::schema({field_random_number});
  std::shared_ptr<arrow::Table> table = arrow::Table::Make(schema, {random_number_chunks});
  std::shared_ptr<arrow::io::FileOutputStream> outfile;
  ARROW_ASSIGN_OR_RAISE(outfile, arrow::io::FileOutputStream::Open("test_out.csv"));
  ARROW_ASSIGN_OR_RAISE(auto csv_writer, arrow::csv::MakeCSVWriter(outfile, table->schema()));
  ARROW_RETURN_NOT_OK(csv_writer->WriteTable(*table));
  ARROW_RETURN_NOT_OK(csv_writer->Close());

  return arrow::Status::OK();
}

const size_t num_species = 3; //10
// initalize parameters, set the number of species to 10 in the generalized lv equation

const size_t outerloop = 1000; //200  
// samplesize

const size_t innerloop = 200; //500
// precision

const unsigned int threadPerBlock = 1024;
const unsigned int blockCount = 207520; //the multiply is just larger than 8.5 * 10**8
const unsigned int totalThreads = threadPerblock * blockCount;

int main( int arc, char* argv[] ) 
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    int sampleSize = num_species * outerloop;
    curandState *devStates;
    double_t *devResults, *hostResults, *devResults2, *hostResults2;
    hostResults = (double_t *)calloc(sampleSize * deviceCount, sizeof(double_t));
    #pragma omp parallel for num_threads(4) private(devResults, devResults2, devStates) shared(sampleSize, totalThreads, blockCount, threadPerBlock)
    for (int dev=0; dev < deviceCount; ++dev) {
        cudaSetDevice(dev);
        cudaMalloc((void **)&devResults, sampleSize * sizeof(double_t));
        cudaMalloc((void **)&devResults2, sampleSize * sizeof(double_t));
        cudaMemset(devResults, 0, sampleSize * sizeof(double_t));
        cudaMemset(devResults2, 0, sampleSize * sizeof(double_t));
        cudaMalloc((void **)&devStates, totalThreads * sizeof(curandState));
        setup_kernel<<<blockCount, threadPerBlock>>>(devStates, dev);
        initialize_parameters_growth_sigma<<<blockCount, threadPerBlock>>>(devStates, devResults, devResults2, sampleSize);
        cudaMemcpy(hostResults + dev * sampleSize, devResults, sampleSize * sizeof(double_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(hostResults2 + dev * sampleSize, devResults2, sampleSize * sizeof(double_t), cudaMemcpyDeviceToHost);
        cudaFree(devResults);
        cudaFree(devResults2);
    }
    state_type growth_rate(hostResults, hostResults +  sampleSize * deviceCount);
    state_type Sigma(hostResults2, hostResults2 + sampleSize * deviceCount);
    free(hostResults);
    free(hostResults2);

    sampleSize = num_species * num_species * outerloop;
    hostResults = (double_t *)calloc(sampleSize * deviceCount, sizeof(double_t));
    #pragma omp parallel for num_threads(4) private(devResults, devStates) shared(sampleSize, totalThreads, blockCount, threadPerBlock)
    for (int dev=0; dev < deviceCount; ++dev) {
        cudaSetDevice(dev);
        cudaMalloc((void **)&devResults, sampleSize * sizeof(double_t));
        cudaMemset(devResults, 0, sampleSize * sizeof(double_t));
        // cudaMalloc((void **)&devStates, totalThreads * sizeof(curandState));
        // setup_kernel<<<blockCount, threadPerBlock>>>(devStates, dev);
        initialize_parameters_interaction<<<blockCount, threadPerBlock>>>(devStates, devResults, sampleSize);
        cudaMemcpy(hostResults + dev * sampleSize, devResults, sampleSize * sizeof(double_t), cudaMemcpyDeviceToHost);
        cudaFree(devResults);
    }
    state_type interaction(hostResults, hostResults +  sampleSize * deviceCount);
    free(hostResults);
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
    
    sampleSize = innerloop;
    hostResults = (double_t *)calloc(sampleSize * deviceCount, sizeof(double_t));
    #pragma omp parallel for num_threads(4) private(devResults, devStates) shared(sampleSize, totalThreads, blockCount, threadPerBlock)
    for (int dev=0; dev < deviceCount; ++dev) {
        cudaSetDevice(dev);
        cudaMalloc((void **)&devResults, sampleSize * sizeof(double_t));
        cudaMemset(devResults, 0, sampleSize * sizeof(double_t));
        intialize_parameters_dilution<<<blockCount, threadPerBlock>>>(devStates, devResults, sampleSize);
        cudaMemcpy(hostResults + dev * sampleSize, devResults, sampleSize * sizeof(double_t), cudaMemcpyDeviceToHost);
        cudaFree(devResults);
    }
    state_type dilution(hostResults, hostResults +  sampleSize * deviceCount);
    free(hostResults);
    
    sampleSize = num_species * innerloop * outerloop;
    hostResults = (double_t *)calloc(sampleSize * deviceCount, sizeof(double_t));
    #pragma omp parallel for num_threads(4) private(devResults, devStates) shared(sampleSize, totalThreads, blockCount, threadPerBlock)
    for (int dev=0; dev < deviceCount; ++dev) {
        cudaSetDevice(dev);
        cudaMalloc((void **)&devResults, sampleSize * sizeof(double_t));
        cudaMemset(devResults, 0, sampleSize * sizeof(double_t));
        initialize_initial<<<blockCount, threadPerBlock>>>(devStates, devResults, sampleSize);
        cudaMemcpy(hostResults + dev * sampleSize, devResults, sampleSize * sizeof(double_t), cudaMemcpyDeviceToHost);
        cudaFree(devResults);
    }
    state_type initial(hostResults, hostResults +  sampleSize * deviceCount);
    for (int i = 0; i < innerloop * outerloop; ++i ) {
        double_t sum = thrust::reduce(initial.begin() + i * num_species, initial.begin() + (i + 1) * num_species, 0);
        thrust::for_each(initial.begin() + i * num_species, initial.begin() + (i + 1) * num_species, normalize(sum));
    }

    typedef runge_kutta_dopri5< state_type , value_type , state_type , value_type > stepper_type;
    generalized_lotka_volterra_system glv_system( num_species, innerloop, outerloop, growth_rate, Sigma, interaction, dilution );

    integrate_adaptive( make_dense_output(1.0e-6, 1.0e-6, stepper_type() ), glv_system, initial , 0.0, 1.0, 0.01);

    // TODO: parse results with Euclidean distance aka 2-norm
    

    return 0;
}