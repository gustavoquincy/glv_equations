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
    #include <thrust/sequence.h>
    #include <arrow/api.h>
    #include <arrow/csv/api.h>
    #include <arrow/io/api.h>
    #include <arrow/ipc/api.h>
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

    #pragma region //curand kernels

    __global__ __launch_bounds__(1024) void initialize_parameters_growth_sigma(curandState *state, double_t *growth_rate, double_t *sigma, int sampleSize, int dev, int offset, double_t growth_mean)
    {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(dev, id, offset, &state[id]);
    // no dim 10**6
    if (id < sampleSize) {
        double_t growth_width = growth_mean * curand_uniform_double(&state[id]);
        growth_rate[id] = growth_mean - growth_width + 2 * growth_width * curand_uniform_double(&state[id]);
        sigma[id] = 0.5 * curand_uniform_double(&state[id]);
    }
    }

    __global__ __launch_bounds__(1024) void initialize_parameters_interaction(curandState *state, double_t *interaction, int sampleSize, int dev, int offset)
    {
        int id = threadIdx.x + blockIdx.x * blockDim.x;
        curand_init(dev, id, offset, &state[id]);
        if (id < sampleSize) {
            // n2o dim 10**8
            double_t compete_mean = 0.5 + 1.5 * curand_uniform_double(&state[id]);
            double_t compete_dense = 0.5 + 0.5 * curand_uniform_double(&state[id]);
            double_t promote_mean = 0.01 + 0.99 * curand_uniform_double(&state[id]);
            double_t promote_dense = ( 1 -  compete_dense ) * curand_uniform_double(&state[id]);
            double_t compete_width = compete_mean * curand_uniform_double(&state[id]);
            double_t promote_width = promote_mean * curand_uniform_double(&state[id]);
            (curand_uniform(&state[id]) <= promote_dense) ? 
                interaction[id] = promote_mean - promote_width + 2 * promote_width * curand_uniform_double(&state[id]) : (curand_uniform(&state[id]) >= compete_dense) ? 
                interaction[id] = -1 * (compete_mean - compete_width + 2 * compete_width * curand_uniform_double(&state[id])) : 0 ;
        }
    }

    __global__ __launch_bounds__(1024) void initialize_parameters_dilution(curandState *state, double_t *dilution, int sampleSize, int dev, int offset, double_t growth_mean)
    {
        int id = threadIdx.x + blockIdx.x * blockDim.x;
        curand_init(dev, id, offset, &state[id]);
        if (id < sampleSize) {
            // i dim 10**3
            dilution[id] = std::min(growth_mean * curand_uniform_double(&state[id]), 0.3 * curand_uniform_double(&state[id]));
        }
    }

    __global__ __launch_bounds__(1024) void initialize_initial(curandState *state, double_t *initial, int sampleSize, int dev, int offset)
    {
        int id = threadIdx.x + blockIdx.x * blockDim.x;
        curand_init(dev, id, offset, &state[id]);
        if (id < sampleSize) {
            // noi dim 10**9
            initial[id] = curand_uniform_double(&state[id]);
        }
    }
    #pragma endregion

    arrow::Status state_write_table(double_t *state, int n, int o, int i, double_t time) {
        arrow::DoubleBuilder doublebuilder;
        ARROW_RETURN_NOT_OK(doublebuilder.AppendNulls(n * i)); // ni, the column length of the table
        ARROW_ASSIGN_OR_RAISE(std::shared_ptr<arrow::Array> arr_null, doublebuilder.Finish());
        std::shared_ptr<arrow::Table> state_table = arrow::Table::Make(arrow::schema({arrow::field("", arrow::float64())}), {arr_null});
        for (int j = 0; j < o; ++j){
            for (int k = 0; k < i; ++k) {
                double_t *p_start = state + n * j + n * o * k;
                double_t *p_end = state + n * j + n * o * k + n;
                for (double_t *ptr = p_start; ptr < p_end; ++ptr) {
                    ARROW_RETURN_NOT_OK(doublebuilder.AppendValues(ptr, 1));
                }
            }
            ARROW_ASSIGN_OR_RAISE(std::shared_ptr<arrow::Array> sys_timesnap, doublebuilder.Finish());
            std::shared_ptr<arrow::ChunkedArray> sys_timesnap_chunks = std::make_shared<arrow::ChunkedArray>(sys_timesnap);
            std::string str = "system_" + std::to_string(j);
            std::shared_ptr<arrow::Field> field = arrow::field(str, arrow::float64());
            ARROW_ASSIGN_OR_RAISE(state_table, state_table->AddColumn(1, field, sys_timesnap_chunks));
        } 
        ARROW_ASSIGN_OR_RAISE(state_table, state_table->RemoveColumn(0)); // remove the null column, aka the first column
        std::shared_ptr<arrow::io::FileOutputStream> outfile;
        std::string str = "system_state_at_time_" + std::to_string(time) + ".csv";
        ARROW_ASSIGN_OR_RAISE(outfile, arrow::io::FileOutputStream::Open(str));
        ARROW_ASSIGN_OR_RAISE(auto csv_writer, arrow::csv::MakeCSVWriter(outfile, state_table->schema()));
        ARROW_RETURN_NOT_OK(csv_writer->WriteTable(*state_table));
        ARROW_RETURN_NOT_OK(csv_writer->Close());
        return arrow::Status::OK();
    }

    struct generalized_lotka_volterra_system
    {
        const size_t m_num_species, m_innerloop, m_outerloop;
        state_type m_growth_rate, m_Sigma, m_interaction, m_dilution; //pass-in value
        state_type growth_rate_i, Sigma_i, interaction_i, dilution_ni;  //operator-use value
        
        // m_growth_rate(no), m_Sigma(no), m_dilution(o), m_interaction(nno)

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
                thrust::get<1>(t) = thrust::get<0>(t) * thrust::get<2>(t) * ( 1 + thrust::get<6>(t) + 0.0 * thrust::get<3>(t) * thrust::get<5>(t) / ( 1 + thrust::get<5>(t) )) - thrust::get<4>(t) * thrust::get<0>(t);
            }
        };

        void operator()( state_type& y , state_type& dydt, value_type t)
        {
            // copy y n times to make it n^2*io
            state_type y_n( y.size() * m_num_species );
            for (int i=0; i <  m_innerloop * m_outerloop; ++i) {
                for (int j=0; j < m_num_species; ++j) {
                        thrust::copy( y.begin() + i * m_num_species, y.begin() + (i + 1) * m_num_species, y_n.begin() + i * m_num_species * m_num_species + j * m_num_species );
                }
            }
            // multiply interaction with y piecewisely
            state_type result( interaction_i.size() );
            thrust::transform( y_n.begin(), y_n.end(), interaction_i.begin(), result.begin(), thrust::multiplies<value_type>() );
            // find pos_sum and neg_sum for every n in the result vector
            host_type result_host( result.size() );
            result_host = result;
            host_type pos_sum_host( m_num_species * m_innerloop * m_outerloop ), neg_sum_host( m_num_species * m_innerloop * m_outerloop );
            for (int i=0; i < m_num_species * m_innerloop * m_outerloop; ++i) {
                value_type pos_sum = 0.0;
                value_type neg_sum = 0.0;
                for (int j=0; j < m_num_species; ++j) {
                    value_type vec_val = result_host[ i * m_num_species + j ];
                    vec_val > 0 ? pos_sum += vec_val : neg_sum += vec_val;
                }
                // std::clog << "print pos_sum: " << pos_sum << std::endl;
                // std::clog << "print neg_sum: " << neg_sum << std::endl;
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

            double_t *raw_y = thrust::raw_pointer_cast(y.data());
            arrow::Status status = state_write_table(raw_y, m_num_species, m_outerloop, m_innerloop, t);
            if (!status.ok()) {
                std::clog << status.ToString() << std::endl;
            }
        }

    };

    #pragma region //write table arrow status
    arrow::Status growth_rate_sigma_write_table(double_t *growth_rate, double_t *Sigma, int64_t size) {
    arrow::DoubleBuilder doublebuilder;
    ARROW_RETURN_NOT_OK(doublebuilder.AppendValues(growth_rate, size));
    std::shared_ptr<arrow::Array> growth_rate_arr;
    ARROW_ASSIGN_OR_RAISE(growth_rate_arr, doublebuilder.Finish());
    std::shared_ptr<arrow::ChunkedArray> growth_rate_chunks = std::make_shared<arrow::ChunkedArray>(growth_rate_arr);
    ARROW_RETURN_NOT_OK(doublebuilder.AppendValues(Sigma, size));
    std::shared_ptr<arrow::Array> sigma_arr;
    ARROW_ASSIGN_OR_RAISE(sigma_arr, doublebuilder.Finish());
    std::shared_ptr<arrow::ChunkedArray> sigma_chunks = std::make_shared<arrow::ChunkedArray>(sigma_arr);
    std::shared_ptr<arrow::Field> field_growth_rate, field_sigma;
    std::shared_ptr<arrow::Schema> schema;
    field_growth_rate = arrow::field("growth rate", arrow::float64());
    field_sigma = arrow::field("sigma", arrow::float64());
    schema = arrow::schema({field_growth_rate, field_sigma});
    std::shared_ptr<arrow::Table> table = arrow::Table::Make(schema, {growth_rate_chunks, sigma_chunks});
    std::shared_ptr<arrow::io::FileOutputStream> outfile;
    ARROW_ASSIGN_OR_RAISE(outfile, arrow::io::FileOutputStream::Open("growth_rate_and_sigma.csv"));
    ARROW_ASSIGN_OR_RAISE(auto csv_writer, arrow::csv::MakeCSVWriter(outfile, table->schema()));
    ARROW_RETURN_NOT_OK(csv_writer->WriteTable(*table));
    ARROW_RETURN_NOT_OK(csv_writer->Close());
    return arrow::Status::OK();
    }

    arrow::Status interaction_write_table(double_t *interaction, int64_t size) {
    arrow::DoubleBuilder doublebuilder;
    ARROW_RETURN_NOT_OK(doublebuilder.AppendValues(interaction, size));
    std::shared_ptr<arrow::Array> interaction_arr;
    ARROW_ASSIGN_OR_RAISE(interaction_arr, doublebuilder.Finish());
    std::shared_ptr<arrow::ChunkedArray> interaction_chunks = std::make_shared<arrow::ChunkedArray>(interaction_arr);
    std::shared_ptr<arrow::Field> field_interaction;
    std::shared_ptr<arrow::Schema> schema;
    field_interaction = arrow::field("interaction matrix", arrow::float64());
    schema = arrow::schema({field_interaction});
    std::shared_ptr<arrow::Table> table = arrow::Table::Make(schema, {interaction_chunks});
    std::shared_ptr<arrow::io::FileOutputStream> outfile;
    ARROW_ASSIGN_OR_RAISE(outfile, arrow::io::FileOutputStream::Open("interaction.csv"));
    ARROW_ASSIGN_OR_RAISE(auto csv_writer, arrow::csv::MakeCSVWriter(outfile, table->schema()));
    ARROW_RETURN_NOT_OK(csv_writer->WriteTable(*table));
    ARROW_RETURN_NOT_OK(csv_writer->Close());
    return arrow::Status::OK();
    }

    arrow::Status dilution_write_table(double_t *dilution, int64_t size) {
    arrow::DoubleBuilder doublebuilder;
    ARROW_RETURN_NOT_OK(doublebuilder.AppendValues(dilution, size));
    std::shared_ptr<arrow::Array> dilution_arr;
    ARROW_ASSIGN_OR_RAISE(dilution_arr, doublebuilder.Finish());
    std::shared_ptr<arrow::ChunkedArray> dilution_chunks = std::make_shared<arrow::ChunkedArray>(dilution_arr);
    std::shared_ptr<arrow::Field> field_dilution;
    std::shared_ptr<arrow::Schema> schema;
    field_dilution = arrow::field("dilution", arrow::float64());
    schema = arrow::schema({field_dilution});
    std::shared_ptr<arrow::Table> table = arrow::Table::Make(schema, {dilution_chunks});
    std::shared_ptr<arrow::io::FileOutputStream> outfile;
    ARROW_ASSIGN_OR_RAISE(outfile, arrow::io::FileOutputStream::Open("dilution.csv"));
    ARROW_ASSIGN_OR_RAISE(auto csv_writer, arrow::csv::MakeCSVWriter(outfile, table->schema()));
    ARROW_RETURN_NOT_OK(csv_writer->WriteTable(*table));
    ARROW_RETURN_NOT_OK(csv_writer->Close());
    return arrow::Status::OK();
    }
    arrow::Status initial_write_table(double_t *initial, int64_t size) {
    arrow::DoubleBuilder doublebuilder;
    ARROW_RETURN_NOT_OK(doublebuilder.AppendValues(initial, size));
    std::shared_ptr<arrow::Array> initial_arr;
    ARROW_ASSIGN_OR_RAISE(initial_arr, doublebuilder.Finish());
    std::shared_ptr<arrow::ChunkedArray> initial_chunks = std::make_shared<arrow::ChunkedArray>(initial_arr);
    std::shared_ptr<arrow::Field> field_initial;
    std::shared_ptr<arrow::Schema> schema;
    field_initial = arrow::field("initial", arrow::float64());
    schema = arrow::schema({field_initial});
    std::shared_ptr<arrow::Table> table = arrow::Table::Make(schema, {initial_chunks});
    std::shared_ptr<arrow::io::FileOutputStream> outfile;
    ARROW_ASSIGN_OR_RAISE(outfile, arrow::io::FileOutputStream::Open("initial.csv"));
    ARROW_ASSIGN_OR_RAISE(auto csv_writer, arrow::csv::MakeCSVWriter(outfile, table->schema()));
    ARROW_RETURN_NOT_OK(csv_writer->WriteTable(*table));
    ARROW_RETURN_NOT_OK(csv_writer->Close());
    return arrow::Status::OK();
    }
    #pragma endregion

    #pragma region //functor for thrust vector interaction and initial
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

    const size_t num_species = 3; //10

    const size_t outerloop = 200; //1000  

    const size_t innerloop = 200; //500

    const unsigned int threadPerBlock = 1024;
    const unsigned int blockCount = 207520; //the multiply is just larger than 8.5 * 10**8
    const unsigned int totalThreads = threadPerBlock * blockCount;

    int main( int arc, char* argv[] ) 
    {
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);

        int noSize = num_species * outerloop / deviceCount;
        int nnoSize = num_species * num_species * outerloop / deviceCount;
        int oSize = outerloop / deviceCount;
        int noiSize = num_species * outerloop * innerloop / deviceCount;
        curandState *devStates;
        double_t *growth_rate_host, *growth_rate_dev, *sigma_host, *sigma_dev, *interaction_host, *interaction_dev, *dilution_host, *dilution_dev, *initial_host, *initial_dev;
        growth_rate_host = (double_t *)calloc(noSize * deviceCount, sizeof(double_t));
        sigma_host = (double_t *)calloc(noSize * deviceCount, sizeof(double_t));
        //interaction_host = (double_t *)calloc(nnoSize * deviceCount, sizeof(double_t));
        dilution_host = (double_t *)calloc(oSize * deviceCount, sizeof(double_t));
        initial_host = (double_t *)calloc(noiSize * deviceCount, sizeof(double_t));
        double_t random = (double_t)rand() / RAND_MAX;
        double_t growth_mean = 0.1 + 1.4 * random;
        #pragma omp parallel for num_threads(4) private(devStates, growth_rate_dev, sigma_dev, interaction_dev, dilution_dev, initial_dev) shared(totalThreads, blockCount, threadPerBlock, noSize, nnoSize, oSize, noiSize)
        for (int dev=0; dev < deviceCount; ++dev) {
            cudaSetDevice(dev);
            cudaMalloc((void **)&growth_rate_dev, noSize * sizeof(double_t));
            cudaMalloc((void **)&sigma_dev, noSize * sizeof(double_t));
            cudaMemset(growth_rate_dev, 0, noSize * sizeof(double_t));
            cudaMemset(sigma_dev, 0, noSize * sizeof(double_t));
            cudaMalloc((void **)&devStates, totalThreads * sizeof(curandState));
            initialize_parameters_growth_sigma<<<blockCount, threadPerBlock>>>(devStates, growth_rate_dev, sigma_dev, noSize, dev, 0, growth_mean);
            cudaMemcpy(growth_rate_host + dev * noSize, growth_rate_dev, noSize * sizeof(double_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(sigma_host + dev * noSize, sigma_dev, noSize * sizeof(double_t), cudaMemcpyDeviceToHost);
            cudaFree(growth_rate_host);
            cudaFree(sigma_host);
            /*
            cudaMalloc((void **)&interaction_dev, nnoSize * sizeof(double_t));
            cudaMemset(interaction_dev, 0, nnoSize * sizeof(double_t));
            initialize_parameters_interaction<<<blockCount, threadPerBlock>>>(devStates, interaction_dev, nnoSize, dev, 1);
            cudaMemcpy(interaction_host + dev * nnoSize, interaction_dev, nnoSize * sizeof(double_t), cudaMemcpyDeviceToHost);
            cudaFree(interaction_dev);
            */
            cudaMalloc((void **)&dilution_dev, oSize * sizeof(double_t));
            cudaMemset(dilution_dev, 0, oSize * sizeof(double_t));
            initialize_parameters_dilution<<<blockCount, threadPerBlock>>>(devStates, dilution_dev, oSize, dev, 2, growth_mean);
            cudaMemcpy(dilution_host + dev * oSize, dilution_dev, oSize * sizeof(double_t), cudaMemcpyDeviceToHost);
            cudaFree(dilution_dev);
            cudaMalloc((void **)&initial_dev, noiSize * sizeof(double_t));
            cudaMemset(initial_dev, 0, noiSize * sizeof(double_t));
            initialize_initial<<<blockCount, threadPerBlock>>>(devStates, initial_dev, noiSize, dev, 3);
            cudaMemcpy(initial_host + dev * noiSize, initial_dev, noiSize * sizeof(double_t), cudaMemcpyDeviceToHost);
            cudaFree(initial_dev);
        }
        state_type growth_rate(growth_rate_host, growth_rate_host +  noSize * deviceCount);
        state_type Sigma(sigma_host, sigma_host + noSize * deviceCount);
        free(growth_rate_host);
        free(sigma_host);
        /*
        state_type interaction(interaction_host, interaction_host +  nnoSize * deviceCount);
        free(interaction_host);
        */
        state_type interaction(nnoSize * deviceCount);
        size_t dim = interaction.size();
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
        state_type dilution(dilution_host, dilution_host +  oSize * deviceCount);
        free(dilution_host);
        state_type initial(initial_host, initial_host +  noiSize * deviceCount);
        free(initial_host);
        for (int i = 0; i < innerloop * outerloop - 1; ++i ) {
            double_t sum = thrust::reduce(initial.begin() + i * num_species, initial.begin() + (i + 1) * num_species, 0.0);
            thrust::for_each(initial.begin() + i * num_species, initial.begin() + (i + 1) * num_species, normalize(sum));
        }

        int64_t size = growth_rate.size();
        double_t *raw_growth_rate = thrust::raw_pointer_cast(growth_rate.data());
        double_t *raw_sigma = thrust::raw_pointer_cast(Sigma.data());
        growth_rate_sigma_write_table(raw_growth_rate, raw_sigma, size);
        double_t *raw_interaction = thrust::raw_pointer_cast(interaction.data());
        size = interaction.size();
        interaction_write_table(raw_interaction, size);
        double_t *raw_dilution = thrust::raw_pointer_cast(dilution.data());
        size = dilution.size();
        dilution_write_table(raw_dilution, size);
        size = initial.size();
        double_t *raw_initial = thrust::raw_pointer_cast(initial.data());
        initial_write_table(raw_initial, size);

        typedef runge_kutta_dopri5< state_type , value_type , state_type , value_type > stepper_type;
        generalized_lotka_volterra_system glv_system( num_species, innerloop, outerloop, growth_rate/*no*/, Sigma/*no*/, interaction/*nno*/, dilution/*o*/);

        integrate_adaptive( make_dense_output(1.0e-6, 1.0e-6, stepper_type() ), glv_system, initial/*noi*/ , 0.0, 100.0, 0.1);

        // TODO: parse results with Euclidean distance aka 2-norm
        

        return 0;
    }