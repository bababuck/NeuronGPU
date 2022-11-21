/*
Copyright (C) 2021 Bruno Golosio
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <config.h>
#include <stdio.h>
#include <stdint.h>
#include <cmath>
#include <iostream>
#include <string>
#include <algorithm>
#include <curand.h>
#include "spike_buffer.h"
#include "cuda_error.h"
#include "send_spike.h"
#include "get_spike.h"
#include "connect_mpi.h"

#include "spike_generator.h"
#include "multimeter.h"
#include "poisson.h"
#include "getRealTime.h"
#include "random.h"
#include "neurongpu.h"
#include "nested_loop.h"
#include "dir_connect.h"
#include "rev_spike.h"
#include "spike_mpi.h"
#include "iaf_psc_exp_g.h"
#include "poiss_gen.h"
#include "poiss_gen_variables.h"

using namespace iaf_psc_exp_g_ns;

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#define THREAD_MAXNUM omp_get_max_threads()
#define THREAD_IDX omp_get_thread_num()
#else
#define THREAD_MAXNUM 1
#define THREAD_IDX 0
#endif
#define NUM_BLOCKS 30
#define GroupResolution 3
				    //#define VERBOSE_TIME

#define I_syn var[i_I_syn]
#define V_m_rel var[i_V_m_rel]
#define refractory_step var[i_refractory_step]
#define I_e param[i_I_e]
#define tau_m_ group_param_[i_tau_m]
#define C_m_ group_param_[i_C_m]
#define E_L_ group_param_[i_E_L]
#define Theta_rel_ group_param_[i_Theta_rel]
#define V_reset_rel_ group_param_[i_V_reset_rel]
#define tau_syn_ group_param_[i_tau_syn]
#define t_ref_ group_param_[i_t_ref]

__constant__ double NeuronGPUTime;
__constant__ long long NeuronGPUTimeIdx;
__constant__ float NeuronGPUTimeResolution;


extern __device__ int *nodes_per_block;
extern __device__ int *InternRevSpikeNum;
extern __device__ int *InternRevSpikeNConn;
extern __device__ int *InternSpikeTargetNum;
extern __device__ int *InternSpikeNum;
extern __device__ int InternMaxSpikeNum;

extern __constant__ NodeGroupStruct NodeGroupArray[];
extern __device__ signed char *NodeGroupMap;

extern __device__ double atomicAddDouble(double* address, double val);

__device__ iaf_psc_exp_g **d_node_vect;
__device__ poiss_gen *poisson_inputs;

enum KernelFloatParamIndexes {
  i_time_resolution = 0,
  i_max_spike_num_fact,
  i_max_spike_per_host_fact,
  N_KERNEL_FLOAT_PARAM
};

enum KernelIntParamIndexes {
  i_rnd_seed = 0,
  i_verbosity_level,
  i_max_spike_buffer_size,
  i_remote_spike_height_flag,
  N_KERNEL_INT_PARAM
};

const std::string kernel_float_param_name[N_KERNEL_FLOAT_PARAM] = {
  "time_resolution",
  "max_spike_num_fact",
  "max_spike_per_host_fact"
};

const std::string kernel_int_param_name[N_KERNEL_INT_PARAM] = {
  "rnd_seed",
  "verbosity_level",
  "max_spike_buffer_size",
  "remote_spike_height_flag"
};

__global__ void initNodesPerBlock(int *d_nodes_per_block, int value)
{
  nodes_per_block = d_nodes_per_block;
  *nodes_per_block = value;
}

NeuronGPU::NeuronGPU()
{
  random_generator_ = new curandGenerator_t;
  CURAND_CALL(curandCreateGenerator(random_generator_,
				    CURAND_RNG_PSEUDO_DEFAULT));
  poiss_generator_ = new PoissonGenerator;
  multimeter_ = new Multimeter;
  net_connection_ = new NetConnection;
  
  SetRandomSeed(54321ULL);
  
  calibrate_flag_ = false;

  start_real_time_ = getRealTime();
  max_spike_buffer_size_ = 20;
  t_min_ = 0.0;
  sim_time_ = 1000.0;        //Simulation time in ms
  n_poiss_node_ = 0;
  n_remote_node_ = 0;
  SetTimeResolution(0.1);  // time resolution in ms
  max_spike_num_fact_ = 1.0;
  max_spike_per_host_fact_ = 1.0;
  
  error_flag_ = false;
  error_message_ = "";
  error_code_ = 0;
  
  on_exception_ = ON_EXCEPTION_EXIT;

  verbosity_level_ = 4;
  
  mpi_flag_ = false;
#ifdef HAVE_MPI
  connect_mpi_ = new ConnectMpi;
  connect_mpi_->net_connection_ = net_connection_;
  connect_mpi_->remote_spike_height_ = false;
#endif
  
  NestedLoop::Init();
  SpikeBufferUpdate_time_ = 0;
  poisson_generator_time_ = 0;
  neuron_Update_time_ = 0;
  copy_ext_spike_time_ = 0;
  SendExternalSpike_time_ = 0;
  SendSpikeToRemote_time_ = 0;
  RecvSpikeFromRemote_time_ = 0;
  NestedLoop_time_ = 0;
  GetSpike_time_ = 0;
  SpikeReset_time_ = 0;
  ExternalSpikeReset_time_ = 0;
  first_simulation_flag_ = true;
}

NeuronGPU::~NeuronGPU()
{
  multimeter_->CloseFiles();
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  if (calibrate_flag_) {
    FreeNodeGroupMap();
    FreeGetSpikeArrays();
  }

  for (unsigned int i=0; i<node_vect_.size(); i++) {
    delete node_vect_[i];
  }

#ifdef HAVE_MPI
  delete connect_mpi_;
#endif

  delete net_connection_;
  delete multimeter_;
  delete poiss_generator_;
  curandDestroyGenerator(*random_generator_);
  delete random_generator_;
}

int NeuronGPU::SetRandomSeed(unsigned long long seed)
{
  kernel_seed_ = seed + 12345;
  CURAND_CALL(curandDestroyGenerator(*random_generator_));
  random_generator_ = new curandGenerator_t;
  CURAND_CALL(curandCreateGenerator(random_generator_,
				    CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(*random_generator_, seed));
  poiss_generator_->random_generator_ = random_generator_;

  return 0;
}

int NeuronGPU::SetTimeResolution(float time_res)
{
  time_resolution_ = time_res;
  net_connection_->time_resolution_ = time_res;
  
  return 0;
}

int NeuronGPU::SetMaxSpikeBufferSize(int max_size)
{
  max_spike_buffer_size_ = max_size;
  
  return 0;
}

int NeuronGPU::GetMaxSpikeBufferSize()
{
  return max_spike_buffer_size_;
}

int NeuronGPU::CreateNodeGroup(int n_node, int n_port)
{
  int i_node_0 = node_group_map_.size();

#ifdef HAVE_MPI
  if ((int)connect_mpi_->extern_connection_.size() != i_node_0) {
    throw ngpu_exception("Error: connect_mpi_.extern_connection_ and "
			 "node_group_map_ must have the same size!");
  }
#endif

  if ((int)net_connection_->connection_.size() != i_node_0) {
    throw ngpu_exception("Error: net_connection_.connection_ and "
			 "node_group_map_ must have the same size!");
  }
  if ((net_connection_->connection_.size() + n_node) > MAX_N_NEURON) {
    throw ngpu_exception(std::string("Maximum number of neurons ")
			 + std::to_string(MAX_N_NEURON) + " exceeded");
  }
  if (n_port > MAX_N_PORT) {
    throw ngpu_exception(std::string("Maximum number of ports ")
			 + std::to_string(MAX_N_PORT) + " exceeded");
  }
  int i_group = node_vect_.size() - 1;
  node_group_map_.insert(node_group_map_.end(), n_node, i_group);
  
  std::vector<ConnGroup> conn;
  std::vector<std::vector<ConnGroup> >::iterator it
    = net_connection_->connection_.end();
  net_connection_->connection_.insert(it, n_node, conn);

#ifdef HAVE_MPI
  std::vector<ExternalConnectionNode > conn_node;
  std::vector<std::vector< ExternalConnectionNode> >::iterator it1
    = connect_mpi_->extern_connection_.end();
  connect_mpi_->extern_connection_.insert(it1, n_node, conn_node);
#endif
  
  node_vect_[i_group]->Init(i_node_0, n_node, n_port, i_group, &kernel_seed_);
  node_vect_[i_group]->get_spike_array_ = InitGetSpikeArray(n_node, n_port);
  
  return i_node_0;
}

NodeSeq NeuronGPU::CreatePoissonGenerator(int n_node, float rrate)
{
  CheckUncalibrated("Poisson generator cannot be created after calibration");
  if (n_poiss_node_ != 0) {
    throw ngpu_exception("Number of poisson generators cannot be modified.");
  }
  else if (n_node <= 0) {
    throw ngpu_exception("Number of nodes must be greater than zero.");
  }
  
  n_poiss_node_ = n_node;               
 
  BaseNeuron *bn = new BaseNeuron;
  node_vect_.push_back(bn);
  int i_node_0 = CreateNodeGroup( n_node, 0);
  
  float lambda = rrate*time_resolution_ / 1000.0; // rate is in Hz, time in ms
  poiss_generator_->Create(random_generator_, i_node_0, n_node, lambda);
    
  return NodeSeq(i_node_0, n_node);
}


int NeuronGPU::CheckUncalibrated(std::string message)
{
  if (calibrate_flag_ == true) {
    throw ngpu_exception(message);
  }
  
  return 0;
}

int NeuronGPU::Calibrate()
{
  CheckUncalibrated("Calibration can be made only once");
  ConnectRemoteNodes();
  calibrate_flag_ = true;
  BuildDirectConnections();

#ifdef HAVE_MPI
  gpuErrchk(cudaMemcpyToSymbol(NeuronGPUMpiFlag, &mpi_flag_, sizeof(bool)));
#endif

  if (verbosity_level_>=1) {
    std::cout << MpiRankStr() << "Calibrating ...\n";
  }
  
  neural_time_ = t_min_;
  	    
  NodeGroupArrayInit();
  
  max_spike_num_ = (int)round(max_spike_num_fact_
                 * net_connection_->connection_.size()
  		 * net_connection_->MaxDelayNum());
  max_spike_num_ = (max_spike_num_>1) ? max_spike_num_ : 1;

  max_spike_per_host_ = (int)round(max_spike_per_host_fact_
                 * net_connection_->connection_.size()
  		 * net_connection_->MaxDelayNum());
  max_spike_per_host_ = (max_spike_per_host_>1) ? max_spike_per_host_ : 1;
  
  SpikeInit(max_spike_num_, NUM_BLOCKS);
  SpikeBufferInit(net_connection_, max_spike_buffer_size_);

#ifdef HAVE_MPI
  if (mpi_flag_) {
    // remove superfluous argument mpi_np
    connect_mpi_->ExternalSpikeInit(connect_mpi_->extern_connection_.size(),
				    connect_mpi_->mpi_np_,
				    max_spike_per_host_);
  }
#endif
  
  if (net_connection_->NRevConnections()>0) {
    RevSpikeInit(net_connection_, NUM_BLOCKS); 
  }
  
  multimeter_->OpenFiles();
  
  for (unsigned int i=0; i<node_vect_.size(); i++) {
    node_vect_[i]->Calibrate(t_min_, time_resolution_);
  }
  
  SynGroupCalibrate();
  
  gpuErrchk(cudaMemcpyToSymbol(NeuronGPUTimeResolution, &time_resolution_,
			       sizeof(float)));
///////////////////////////////////

  return 0;
}

__device__ void InternPoissGenSendSpikeKernel(curandState *curand_state, double t,
                                        float time_step, float *param_arr,
                                        int n_param,
                                        DirectConnection *dir_conn_array,
                                        uint64_t n_dir_conn)
{
  for (int i_conn=threadIdx.x;i_conn<n_dir_conn;i_conn+=blockDim.x) {
    DirectConnection dir_conn = dir_conn_array[i_conn];
    int irel = dir_conn.irel_source_;
    int i_target = dir_conn.i_target_;
    if (i_target / *nodes_per_block != blockIdx.x) continue;
    int port = dir_conn.port_;
    float weight = dir_conn.weight_;
    float delay = dir_conn.delay_;
    float *param = param_arr + irel*n_param;
    double t_rel = t - origin - delay;

    if ((t_rel>=start) && (t_rel<=stop)){
      int n = curand_poisson(curand_state+i_conn, time_step*rate);
      if (n>0) { // //Send direct spike (i_target, port, weight*n);
        /////////////////////////////////////////////////////////////////
        int i_group=NodeGroupMap[i_target];
        int i = port*NodeGroupArray[i_group].n_node_ + i_target
          - NodeGroupArray[i_group].i_node_0_;
        double d_val = (double)(weight*n);
        atomicAddDouble(&NodeGroupArray[i_group].intern_get_spike_array_[i], d_val); 
        ////////////////////////////////////////////////////////////////
      }
    }
  }
}

__device__
int InternSendDirectSpikes(double t, float time_step)
{
  InternPoissGenSendSpikeKernel(poisson_inputs->d_curand_state_, t, time_step,
                          poisson_inputs->param_arr_, poisson_inputs->n_param_,
                          poisson_inputs->d_dir_conn_array_, poisson_inputs->n_dir_conn_);

  return 0;
}

__device__
int InternClearGetSpikeArrays(int n_nodes)
{
  for (int i=0; i<n_nodes; i++) {
    BaseNeuron *bn = d_node_vect[i];
    if (bn->intern_get_spike_array_ != NULL) {
      int nodes = bn->n_node_;
      int nodes_p_block = ((nodes + NUM_BLOCKS-1)/NUM_BLOCKS);
      int start_node = blockIdx.x * nodes_p_block;
      int finish_node = start_node + nodes_p_block; 
      int ports = bn->n_port_;
      for (int j = threadIdx.x + start_node; j<finish_node;j+=blockDim.x) {
        for (int k=0;k<ports;++k){
          if (j < nodes) {
            bn->intern_get_spike_array_[j + k * nodes] = 0;
	  }
	}
      }
    }
  }
  return 0;
}

__device__
int OldInternClearGetSpikeArrays(int n_nodes)
{
  int start_node = blockIdx.x * *nodes_per_block;
  int last_node = start_node + *nodes_per_block;
  int stride = blockDim.x/32;
  for (unsigned int i=start_node+(threadIdx.x/32); i<last_node && i<n_nodes; i+=stride) {
    BaseNeuron *bn = d_node_vect[i];
    if (bn->intern_get_spike_array_ != NULL) {
      for (int j=threadIdx.x % 32;j<bn->n_node_*bn->n_port_;j+=32) {
        bn->intern_get_spike_array_[j] = 0; 
      }
    }
  }
  return 0;
} 


int NeuronGPU::InternClearGetSpikeArraysTest(int n_nodes)
{
  for (int i=0; i<n_nodes; i++) {
    BaseNeuron *bn = node_vect_[i];
    if (bn->get_spike_array_ != NULL) {
      gpuErrchk(cudaMemset(bn->intern_get_spike_array_, 0, bn->n_node_*bn->n_port_*sizeof(double)));
    }
  }
  return 0;
} 

int NeuronGPU::Simulate(float sim_time) {
  sim_time_ = sim_time;
  return Simulate();
}

__global__ void initNodeVect(iaf_psc_exp_g **vec)
{
  d_node_vect = vec;
}

__global__ void initNode(iaf_psc_exp_g *node, int i)
{
  d_node_vect[i] = node;
}

__global__ void initPoiss(poiss_gen *node)
{
  poisson_inputs = node;
}

__global__ void initNodeParam(int i, float* d_group_param)
{
  d_node_vect[i]->group_param_ = d_group_param;
}

int NeuronGPU::Simulate()
{
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  StartSimulation();
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  for (long long it=0; it<Nt_; it+=(GroupResolution)) {
    SimulationStep();
    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk( cudaPeekAtLastError() );  
  }
  EndSimulation();

  return 0;
}

int NeuronGPU::StartSimulation()
{
  if (!calibrate_flag_) {
    Calibrate();
  }
#ifdef HAVE_MPI                                                                                                            
  if (mpi_flag_) {
    MPI_Barrier(MPI_COMM_WORLD);
  }
#endif
  if (first_simulation_flag_) {
    gpuErrchk(cudaMemcpyToSymbol(NeuronGPUTime, &neural_time_, sizeof(double)));
    multimeter_->WriteRecords(neural_time_);
    build_real_time_ = getRealTime();
    first_simulation_flag_ = false;
  }
  if (verbosity_level_>=1) {
    std::cout << MpiRankStr() << "Simulating ...\n";
    printf("Neural activity simulation time: %.3lf\n", sim_time_);
  }
  
  neur_t0_ = neural_time_;
  it_ = 0;
  Nt_ = (long long)round(sim_time_/time_resolution_);

  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  int* d_nodes_per_block;
  cudaMalloc(&d_nodes_per_block, sizeof(int));
  initNodesPerBlock<<<1,1>>>(d_nodes_per_block, (node_vect_[0]->n_node_+NUM_BLOCKS-1) / NUM_BLOCKS);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  iaf_psc_exp_g **temp_node_vect;
  cudaMalloc(&temp_node_vect, node_vect_.size()*sizeof(iaf_psc_exp_g*));
  initNodeVect<<<1,1>>>(temp_node_vect);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  
  for (unsigned int i=0;i<node_vect_.size();++i){
    float *d_group_param;
    cudaMalloc(&d_group_param, node_vect_[i]->n_group_param_*sizeof(float));
    iaf_psc_exp_g *temp;
    cudaMalloc(&temp, sizeof(iaf_psc_exp_g));
    initNode<<<1,1>>>(temp, i);
    cudaMemcpy(temp, (iaf_psc_exp_g*)node_vect_[i], sizeof(iaf_psc_exp_g), cudaMemcpyHostToDevice);
    initNodeParam<<<1,1>>>(i, d_group_param);  
    cudaMemcpy(d_group_param, node_vect_[i]->group_param_, node_vect_[i]->n_group_param_*sizeof(float), cudaMemcpyHostToDevice);
  }
  poiss_gen *d_poiss_gen;
  cudaMalloc(&d_poiss_gen, sizeof(poiss_gen));
  initPoiss<<<1,1>>>(d_poiss_gen);
  cudaMemcpy(d_poiss_gen, node_vect_[node_vect_.size()-1], sizeof(poiss_gen), cudaMemcpyHostToDevice); 
  gpuErrchk( cudaPeekAtLastError() );  
  gpuErrchk( cudaDeviceSynchronize() ); 
  return 0;
}

int NeuronGPU::EndSimulation()
{

  end_real_time_ = getRealTime();

  //multimeter_->CloseFiles();
  //neuron.rk5.Free();

  if (verbosity_level_>=1) {
    std::cout << MpiRankStr() << "Simulation time: " <<
      (end_real_time_ - build_real_time_) << "\n";
  }
  
  return 0;
}

__device__ double h_intern_propagator_32( double tau_syn, double tau, double C, double h )
{
  const double P32_linear = 1.0 / ( 2.0 * C * tau * tau ) * h * h
    * ( tau_syn - tau ) * exp( -h / tau );
  const double P32_singular = h / C * exp( -h / tau );
  const double P32 =
    -tau / ( C * ( 1.0 - tau / tau_syn ) ) * exp( -h / tau_syn )
    * expm1( h * ( 1.0 / tau_syn - 1.0 / tau ) );

  const double dev_P32 = fabs( P32 - P32_singular );

  if ( tau == tau_syn || ( fabs( tau - tau_syn ) < 0.1 && dev_P32 > 2.0
			   * fabs( P32_linear ) ) )
  {
    return P32_singular;
  }
  else
  {
    return P32;
  }
}

__device__ void iaf_psc_exp_g_InternUpdate
( int n_node, int i_node_0, float *var_arr, float *param_arr, int n_var,
  int n_param, float Theta_rel, float V_reset_rel, int n_refractory_steps,
  float P11, float P22, float P21, float P20, int _it)
{
  for (int i_neuron = threadIdx.x+blockIdx.x*blockDim.x; i_neuron<n_node;i_neuron+=gridDim.x*blockDim.x){//((n_node+NUM_BLOCKS-1)/NUM_BLOCKS);i_neuron+=blockDim.x) {
    float *var = var_arr + n_var*i_neuron;
    float *param = param_arr + n_param*i_neuron; 
    if ( refractory_step > 0.0 ) {
      // neuron is absolute refractory
      refractory_step -= 1.0;
    }
    else { // neuron is not refractory, so evolve V
      V_m_rel = V_m_rel * P22 + I_syn * P21 + I_e * P20;
    }
    // exponential decaying PSC
    I_syn *= P11;
    if (V_m_rel >= Theta_rel ) { // threshold crossing
      PushSpike(i_node_0 + i_neuron, 1.0, _it);
      V_m_rel = V_reset_rel;
      refractory_step = n_refractory_steps;
    }
  }
}

__device__ int InternUpdate(long long it, double t1, iaf_psc_exp_g* neuron, int _it)
{
  // std::cout << "iaf_psc_exp_g neuron update\n";
  float h = neuron->time_resolution_;
  float P11 = exp( -h / neuron->tau_syn_ );
  float P22 = exp( -h / neuron->tau_m_ );
  float P21 = h_intern_propagator_32( neuron->tau_syn_, neuron->tau_m_, neuron->C_m_, h );
  float P20 = neuron->tau_m_ / neuron->C_m_ * ( 1.0 - P22 );
  int n_refractory_steps = int(round(neuron->t_ref_ / h));

  iaf_psc_exp_g_InternUpdate
    (neuron->n_node_, neuron->i_node_0_, neuron->var_arr_, neuron->param_arr_, neuron->n_var_, neuron->n_param_,
      neuron->Theta_rel_, neuron->V_reset_rel_, n_refractory_steps, P11, P22, P21, P20,_it);
  //gpuErrchk( cudaDeviceSynchronize() );
  
  return 0;
}

int NeuronGPU::CreateRecord(std::string file_name, std::string *var_name_arr,
			    int *i_node_arr, int *port_arr,
			    int n_node)
{
  std::vector<BaseNeuron*> neur_vect;
  std::vector<int> i_neur_vect;
  std::vector<int> port_vect;
  std::vector<std::string> var_name_vect;
  for (int i=0; i<n_node; i++) {
    var_name_vect.push_back(var_name_arr[i]);
    int i_group = node_group_map_[i_node_arr[i]];
    i_neur_vect.push_back(i_node_arr[i] - node_vect_[i_group]->i_node_0_);
    port_vect.push_back(port_arr[i]);
    neur_vect.push_back(node_vect_[i_group]);
  }

  return multimeter_->CreateRecord(neur_vect, file_name, var_name_vect,
  				   i_neur_vect, port_vect);

}

int NeuronGPU::CreateRecord(std::string file_name, std::string *var_name_arr,
			    int *i_node_arr, int n_node)
{
  std::vector<int> port_vect(n_node, 0);
  return CreateRecord(file_name, var_name_arr, i_node_arr,
		      port_vect.data(), n_node);
}

std::vector<std::vector<float> > *NeuronGPU::GetRecordData(int i_record)
{
  return multimeter_->GetRecordData(i_record);
}

int NeuronGPU::GetNodeSequenceOffset(int i_node, int n_node, int &i_group)
{
  if (i_node<0 || (i_node+n_node > (int)node_group_map_.size())) {
    throw ngpu_exception("Unrecognized node in getting node sequence offset");
  }
  i_group = node_group_map_[i_node];  
  if (node_group_map_[i_node+n_node-1] != i_group) {
    throw ngpu_exception("Nodes belong to different node groups "
			 "in setting parameter");
  }
  return node_vect_[i_group]->i_node_0_;
}
  
std::vector<int> NeuronGPU::GetNodeArrayWithOffset(int *i_node, int n_node,
						   int &i_group)
{
  int in0 = i_node[0];
  if (in0<0 || in0>(int)node_group_map_.size()) {
    throw ngpu_exception("Unrecognized node in setting parameter");
  }
  i_group = node_group_map_[in0];
  int i0 = node_vect_[i_group]->i_node_0_;
  std::vector<int> nodes;
  nodes.assign(i_node, i_node+n_node);
  for(int i=0; i<n_node; i++) {
    int in = nodes[i];
    if (in<0 || in>=(int)node_group_map_.size()) {
      throw ngpu_exception("Unrecognized node in setting parameter");
    }
    if (node_group_map_[in] != i_group) {
      throw ngpu_exception("Nodes belong to different node groups "
			   "in setting parameter");
    }
    nodes[i] -= i0;
  }
  return nodes;
}

int NeuronGPU::SetNeuronParam(int i_node, int n_node,
			      std::string param_name, float val)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, n_node, i_group);
  
  return node_vect_[i_group]->SetScalParam(i_neuron, n_node, param_name, val);
}

int NeuronGPU::SetNeuronParam(int *i_node, int n_node,
			      std::string param_name, float val)
{
  int i_group;
  std::vector<int> nodes = GetNodeArrayWithOffset(i_node, n_node,
						  i_group);
  return node_vect_[i_group]->SetScalParam(nodes.data(), n_node,
					   param_name, val);
}

int NeuronGPU::SetNeuronParam(int i_node, int n_node, std::string param_name,
			      float *param, int array_size)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, n_node, i_group);
  if (node_vect_[i_group]->IsPortParam(param_name)) {
      return node_vect_[i_group]->SetPortParam(i_neuron, n_node, param_name,
					       param, array_size);
  }
  else {
    return node_vect_[i_group]->SetArrayParam(i_neuron, n_node, param_name,
					      param, array_size);
  }
}

int NeuronGPU::SetNeuronParam( int *i_node, int n_node,
			       std::string param_name, float *param,
			       int array_size)
{
  int i_group;
  std::vector<int> nodes = GetNodeArrayWithOffset(i_node, n_node,
						  i_group);
  if (node_vect_[i_group]->IsPortParam(param_name)) {  
    return node_vect_[i_group]->SetPortParam(nodes.data(), n_node,
					     param_name, param, array_size);
  }
  else {
    return node_vect_[i_group]->SetArrayParam(nodes.data(), n_node,
					      param_name, param, array_size);
  }    
}

int NeuronGPU::IsNeuronScalParam(int i_node, std::string param_name)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, 1, i_group);
  
  return node_vect_[i_group]->IsScalParam(param_name);
}

int NeuronGPU::IsNeuronPortParam(int i_node, std::string param_name)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, 1, i_group);
  
  return node_vect_[i_group]->IsPortParam(param_name);
}

int NeuronGPU::IsNeuronArrayParam(int i_node, std::string param_name)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, 1, i_group);
  
  return node_vect_[i_group]->IsArrayParam(param_name);
}

int NeuronGPU::SetNeuronIntVar(int i_node, int n_node,
			      std::string var_name, int val)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, n_node, i_group);
  
  return node_vect_[i_group]->SetIntVar(i_neuron, n_node, var_name, val);
}

int NeuronGPU::SetNeuronIntVar(int *i_node, int n_node,
			      std::string var_name, int val)
{
  int i_group;
  std::vector<int> nodes = GetNodeArrayWithOffset(i_node, n_node,
						  i_group);
  return node_vect_[i_group]->SetIntVar(nodes.data(), n_node,
					var_name, val);
}

int NeuronGPU::SetNeuronVar(int i_node, int n_node,
			      std::string var_name, float val)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, n_node, i_group);
  
  return node_vect_[i_group]->SetScalVar(i_neuron, n_node, var_name, val);
}

int NeuronGPU::SetNeuronVar(int *i_node, int n_node,
			      std::string var_name, float val)
{
  int i_group;
  std::vector<int> nodes = GetNodeArrayWithOffset(i_node, n_node,
						  i_group);
  return node_vect_[i_group]->SetScalVar(nodes.data(), n_node,
					   var_name, val);
}

int NeuronGPU::SetNeuronVar(int i_node, int n_node, std::string var_name,
			      float *var, int array_size)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, n_node, i_group);
  if (node_vect_[i_group]->IsPortVar(var_name)) {
      return node_vect_[i_group]->SetPortVar(i_neuron, n_node, var_name,
					       var, array_size);
  }
  else {
    return node_vect_[i_group]->SetArrayVar(i_neuron, n_node, var_name,
					      var, array_size);
  }
}

int NeuronGPU::SetNeuronVar( int *i_node, int n_node,
			       std::string var_name, float *var,
			       int array_size)
{
  int i_group;
  std::vector<int> nodes = GetNodeArrayWithOffset(i_node, n_node,
						  i_group);
  if (node_vect_[i_group]->IsPortVar(var_name)) {  
    return node_vect_[i_group]->SetPortVar(nodes.data(), n_node,
					   var_name, var, array_size);
  }
  else {
    return node_vect_[i_group]->SetArrayVar(nodes.data(), n_node,
					    var_name, var, array_size);
  }    
}

int NeuronGPU::IsNeuronIntVar(int i_node, std::string var_name)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, 1, i_group);
  
  return node_vect_[i_group]->IsIntVar(var_name);
}

int NeuronGPU::IsNeuronScalVar(int i_node, std::string var_name)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, 1, i_group);
  
  return node_vect_[i_group]->IsScalVar(var_name);
}

int NeuronGPU::IsNeuronPortVar(int i_node, std::string var_name)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, 1, i_group);
  
  return node_vect_[i_group]->IsPortVar(var_name);
}

int NeuronGPU::IsNeuronArrayVar(int i_node, std::string var_name)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, 1, i_group);
  
  return node_vect_[i_group]->IsArrayVar(var_name);
}


int NeuronGPU::GetNeuronParamSize(int i_node, std::string param_name)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, 1, i_group);
  if (node_vect_[i_group]->IsArrayParam(param_name)!=0) {
    return node_vect_[i_group]->GetArrayParamSize(i_neuron, param_name);
  }
  else {
    return node_vect_[i_group]->GetParamSize(param_name);
  }
}

int NeuronGPU::GetNeuronVarSize(int i_node, std::string var_name)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, 1, i_group);
  if (node_vect_[i_group]->IsArrayVar(var_name)!=0) {
    return node_vect_[i_group]->GetArrayVarSize(i_neuron, var_name);
  }
  else {
    return node_vect_[i_group]->GetVarSize(var_name);
  }
}


float *NeuronGPU::GetNeuronParam(int i_node, int n_node,
				 std::string param_name)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, n_node, i_group);
  if (node_vect_[i_group]->IsScalParam(param_name)) {
    return node_vect_[i_group]->GetScalParam(i_neuron, n_node, param_name);
  }
  else if (node_vect_[i_group]->IsPortParam(param_name)) {
    return node_vect_[i_group]->GetPortParam(i_neuron, n_node, param_name);
  }
  else if (node_vect_[i_group]->IsArrayParam(param_name)) {
    if (n_node != 1) {
      throw ngpu_exception("Cannot get array parameters for more than one node"
			   "at a time");
    }
    return node_vect_[i_group]->GetArrayParam(i_neuron, param_name);
  }
  else {
    throw ngpu_exception(std::string("Unrecognized parameter ")
			 + param_name);
  }
}

float *NeuronGPU::GetNeuronParam( int *i_node, int n_node,
				  std::string param_name)
{
  int i_group;
  std::vector<int> nodes = GetNodeArrayWithOffset(i_node, n_node,
						  i_group);
  if (node_vect_[i_group]->IsScalParam(param_name)) {
    return node_vect_[i_group]->GetScalParam(nodes.data(), n_node,
					     param_name);
  }
  else if (node_vect_[i_group]->IsPortParam(param_name)) {  
    return node_vect_[i_group]->GetPortParam(nodes.data(), n_node,
					     param_name);
  }
  else if (node_vect_[i_group]->IsArrayParam(param_name)) {
    if (n_node != 1) {
      throw ngpu_exception("Cannot get array parameters for more than one node"
			   "at a time");
    }
    return node_vect_[i_group]->GetArrayParam(nodes[0], param_name);
  }
  else {
    throw ngpu_exception(std::string("Unrecognized parameter ")
			 + param_name);
  }
}

float *NeuronGPU::GetArrayParam(int i_node, std::string param_name)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, 1, i_group);

  return node_vect_[i_group]->GetArrayParam(i_neuron, param_name);
}

int *NeuronGPU::GetNeuronIntVar(int i_node, int n_node,
				std::string var_name)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, n_node, i_group);
  if (node_vect_[i_group]->IsIntVar(var_name)) {
    return node_vect_[i_group]->GetIntVar(i_neuron, n_node, var_name);
  }
  else {
    throw ngpu_exception(std::string("Unrecognized integer variable ")
			 + var_name);
  }
}

int *NeuronGPU::GetNeuronIntVar(int *i_node, int n_node,
			       std::string var_name)
{
  int i_group;
  std::vector<int> nodes = GetNodeArrayWithOffset(i_node, n_node,
						  i_group);
  if (node_vect_[i_group]->IsIntVar(var_name)) {
    return node_vect_[i_group]->GetIntVar(nodes.data(), n_node,
					     var_name);
  }
  else {
    throw ngpu_exception(std::string("Unrecognized variable ")
			 + var_name);
  }
}

float *NeuronGPU::GetNeuronVar(int i_node, int n_node,
			       std::string var_name)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, n_node, i_group);
  if (node_vect_[i_group]->IsScalVar(var_name)) {
    return node_vect_[i_group]->GetScalVar(i_neuron, n_node, var_name);
  }
  else if (node_vect_[i_group]->IsPortVar(var_name)) {
    return node_vect_[i_group]->GetPortVar(i_neuron, n_node, var_name);
  }
  else if (node_vect_[i_group]->IsArrayVar(var_name)) {
    if (n_node != 1) {
      throw ngpu_exception("Cannot get array variables for more than one node"
			   "at a time");
    }
    return node_vect_[i_group]->GetArrayVar(i_neuron, var_name);
  }
  else {
    throw ngpu_exception(std::string("Unrecognized variable ")
			 + var_name);
  }
}

float *NeuronGPU::GetNeuronVar(int *i_node, int n_node,
			       std::string var_name)
{
  int i_group;
  std::vector<int> nodes = GetNodeArrayWithOffset(i_node, n_node,
						  i_group);
  if (node_vect_[i_group]->IsScalVar(var_name)) {
    return node_vect_[i_group]->GetScalVar(nodes.data(), n_node,
					     var_name);
  }
  else if (node_vect_[i_group]->IsPortVar(var_name)) {  
    return node_vect_[i_group]->GetPortVar(nodes.data(), n_node,
					   var_name);
  }
  else if (node_vect_[i_group]->IsArrayVar(var_name)) {
    if (n_node != 1) {
      throw ngpu_exception("Cannot get array variables for more than one node"
			   "at a time");
    }
    return node_vect_[i_group]->GetArrayVar(nodes[0], var_name);
  }
  else {
    throw ngpu_exception(std::string("Unrecognized variable ")
			 + var_name);
  }
}

float *NeuronGPU::GetArrayVar(int i_node, std::string var_name)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, 1, i_group);

  return node_vect_[i_group]->GetArrayVar(i_neuron, var_name);
}

int NeuronGPU::ConnectMpiInit(int argc, char *argv[])
{
#ifdef HAVE_MPI
  CheckUncalibrated("MPI connections cannot be initialized after calibration");
  int err = connect_mpi_->MpiInit(argc, argv);
  if (err==0) {
    mpi_flag_ = true;
  }
  
  return err;
#else
  throw ngpu_exception("MPI is not available in your build");
#endif
}

int NeuronGPU::MpiId()
{
#ifdef HAVE_MPI
  return connect_mpi_->mpi_id_;
#else
  throw ngpu_exception("MPI is not available in your build");
#endif
}

int NeuronGPU::MpiNp()
{
#ifdef HAVE_MPI
  return connect_mpi_->mpi_np_;
#else
  throw ngpu_exception("MPI is not available in your build");
#endif

}

int NeuronGPU::ProcMaster()
{
#ifdef HAVE_MPI
  return connect_mpi_->ProcMaster();
#else
  throw ngpu_exception("MPI is not available in your build");
#endif  
}

int NeuronGPU::MpiFinalize()
{
#ifdef HAVE_MPI
  if (mpi_flag_) {
    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized) {
      MPI_Finalize();
    }
  }
  
  return 0;
#else
  throw ngpu_exception("MPI is not available in your build");
#endif
}

std::string NeuronGPU::MpiRankStr()
{
#ifdef HAVE_MPI
  if (mpi_flag_) {
    return std::string("MPI rank ") + std::to_string(connect_mpi_->mpi_id_)
      + " : ";
  }
  else {
    return "";
  }
#else
  return "";
#endif
}

unsigned int *NeuronGPU::RandomInt(size_t n)
{
  return curand_int(*random_generator_, n);
}

float *NeuronGPU::RandomUniform(size_t n)
{
  return curand_uniform(*random_generator_, n);
}

float *NeuronGPU::RandomNormal(size_t n, float mean, float stddev)
{
  return curand_normal(*random_generator_, n, mean, stddev);
}

float *NeuronGPU::RandomNormalClipped(size_t n, float mean, float stddev,
				      float vmin, float vmax, float vstep)
{
  const float epsi = 1.0e-6;
  
  n = (n/4 + 1)*4; 
  int n_extra = n/10;
  n_extra = (n_extra/4 + 1)*4; 
  if (n_extra<1024) {
    n_extra=1024;
  }
  int i_extra = 0;
  float *arr = curand_normal(*random_generator_, n, mean, stddev);
  float *arr_extra = NULL;
  for (size_t i=0; i<n; i++) {
    while (arr[i]<vmin || arr[i]>vmax) {
      if (i_extra==0) {
	arr_extra = curand_normal(*random_generator_, n_extra, mean, stddev);
      }
      arr[i] = arr_extra[i_extra];
      i_extra++;
      if (i_extra==n_extra) {
	i_extra = 0;
	delete[](arr_extra);
	arr_extra = NULL;
      }
    }
  }
  if (arr_extra != NULL) {
    delete[](arr_extra);
  }
  if (vstep>stddev*epsi) {
    for (size_t i=0; i<n; i++) {
      arr[i] = vmin + vstep*round((arr[i] - vmin)/vstep);
    }
  }

  return arr; 
}

int NeuronGPU::BuildDirectConnections()
{
  for (unsigned int iv=0; iv<node_vect_.size(); iv++) {
    if (node_vect_[iv]->has_dir_conn_) {
      std::vector<DirectConnection> dir_conn_vect;
      int i0 = node_vect_[iv]->i_node_0_;
      int n = node_vect_[iv]->n_node_;
      for (int i_source=i0; i_source<i0+n; i_source++) {
	std::vector<ConnGroup> &conn = net_connection_->connection_[i_source];
	for (unsigned int id=0; id<conn.size(); id++) {
	  std::vector<TargetSyn> tv = conn[id].target_vect;
	  for (unsigned int i=0; i<tv.size(); i++) {
	    DirectConnection dir_conn;
	    dir_conn.irel_source_ = i_source - i0;
	    dir_conn.i_target_ = tv[i].node;
	    dir_conn.port_ = tv[i].port;
	    dir_conn.weight_ = tv[i].weight;
	    dir_conn.delay_ = time_resolution_*(conn[id].delay+1);
	    dir_conn_vect.push_back(dir_conn);
	  }
	}
      }
      uint64_t n_dir_conn = dir_conn_vect.size();
      node_vect_[iv]->n_dir_conn_ = n_dir_conn;
      
      DirectConnection *d_dir_conn_array;
      gpuErrchk(cudaMalloc(&d_dir_conn_array,
			   n_dir_conn*sizeof(DirectConnection )));
      gpuErrchk(cudaMemcpy(d_dir_conn_array, dir_conn_vect.data(),
			   n_dir_conn*sizeof(DirectConnection),
			   cudaMemcpyHostToDevice));
      node_vect_[iv]->d_dir_conn_array_ = d_dir_conn_array;
    }
  }

  return 0;
}

std::vector<std::string> NeuronGPU::GetIntVarNames(int i_node)
{
  if (i_node<0 || i_node>(int)node_group_map_.size()) {
    throw ngpu_exception("Unrecognized node in reading variable names");
  }
  int i_group = node_group_map_[i_node];
  
  return node_vect_[i_group]->GetIntVarNames();
}

std::vector<std::string> NeuronGPU::GetScalVarNames(int i_node)
{
  if (i_node<0 || i_node>(int)node_group_map_.size()) {
    throw ngpu_exception("Unrecognized node in reading variable names");
  }
  int i_group = node_group_map_[i_node];
  
  return node_vect_[i_group]->GetScalVarNames();
}

int NeuronGPU::GetNIntVar(int i_node)
{
  if (i_node<0 || i_node>(int)node_group_map_.size()) {
    throw ngpu_exception("Unrecognized node in reading number of "
			 "variables");
  }
  int i_group = node_group_map_[i_node];
  
  return node_vect_[i_group]->GetNIntVar();
}

int NeuronGPU::GetNScalVar(int i_node)
{
  if (i_node<0 || i_node>(int)node_group_map_.size()) {
    throw ngpu_exception("Unrecognized node in reading number of "
			 "variables");
  }
  int i_group = node_group_map_[i_node];
  
  return node_vect_[i_group]->GetNScalVar();
}

std::vector<std::string> NeuronGPU::GetPortVarNames(int i_node)
{
  if (i_node<0 || i_node>(int)node_group_map_.size()) {
    throw ngpu_exception("Unrecognized node in reading variable names");
  }
  int i_group = node_group_map_[i_node];
  
  return node_vect_[i_group]->GetPortVarNames();
}

int NeuronGPU::GetNPortVar(int i_node)
{
  if (i_node<0 || i_node>(int)node_group_map_.size()) {
    throw ngpu_exception("Unrecognized node in reading number of "
			 "variables");
  }
  int i_group = node_group_map_[i_node];
  
  return node_vect_[i_group]->GetNPortVar();
}


std::vector<std::string> NeuronGPU::GetScalParamNames(int i_node)
{
  if (i_node<0 || i_node>(int)node_group_map_.size()) {
    throw ngpu_exception("Unrecognized node in reading parameter names");
  }
  int i_group = node_group_map_[i_node];
  
  return node_vect_[i_group]->GetScalParamNames();
}

int NeuronGPU::GetNScalParam(int i_node)
{
  if (i_node<0 || i_node>(int)node_group_map_.size()) {
    throw ngpu_exception("Unrecognized node in reading number of "
			 "parameters");
  }
  int i_group = node_group_map_[i_node];
  
  return node_vect_[i_group]->GetNScalParam();
}

std::vector<std::string> NeuronGPU::GetPortParamNames(int i_node)
{
  if (i_node<0 || i_node>(int)node_group_map_.size()) {
    throw ngpu_exception("Unrecognized node in reading parameter names");
  }
  int i_group = node_group_map_[i_node];
  
  return node_vect_[i_group]->GetPortParamNames();
}

int NeuronGPU::GetNPortParam(int i_node)
{
  if (i_node<0 || i_node>(int)node_group_map_.size()) {
    throw ngpu_exception("Unrecognized node in reading number of "
			 "parameters");
  }
  int i_group = node_group_map_[i_node];
  
  return node_vect_[i_group]->GetNPortParam();
}


std::vector<std::string> NeuronGPU::GetArrayParamNames(int i_node)
{
  if (i_node<0 || i_node>(int)node_group_map_.size()) {
    throw ngpu_exception("Unrecognized node in reading array parameter names");
  }
  int i_group = node_group_map_[i_node];
  
  return node_vect_[i_group]->GetArrayParamNames();
}

int NeuronGPU::GetNArrayParam(int i_node)
{
  if (i_node<0 || i_node>(int)node_group_map_.size()) {
    throw ngpu_exception("Unrecognized node in reading number of array "
			 "parameters");
  }
  int i_group = node_group_map_[i_node];
  
  return node_vect_[i_group]->GetNArrayParam();
}


std::vector<std::string> NeuronGPU::GetArrayVarNames(int i_node)
{
  if (i_node<0 || i_node>(int)node_group_map_.size()) {
    throw ngpu_exception("Unrecognized node in reading array variable names");
  }
  int i_group = node_group_map_[i_node];
  
  return node_vect_[i_group]->GetArrayVarNames();
}

int NeuronGPU::GetNArrayVar(int i_node)
{
  if (i_node<0 || i_node>(int)node_group_map_.size()) {
    throw ngpu_exception("Unrecognized node in reading number of array "
			 "variables");
  }
  int i_group = node_group_map_[i_node];
  
  return node_vect_[i_group]->GetNArrayVar();
}

ConnectionStatus NeuronGPU::GetConnectionStatus(ConnectionId conn_id) {
  ConnectionStatus conn_stat = net_connection_->GetConnectionStatus(conn_id);
  if (calibrate_flag_ == true) {
    int i_source = conn_id.i_source_;
    int i_group = conn_id.i_group_;
    int i_conn = conn_id.i_conn_;
    int n_spike_buffer = net_connection_->connection_.size();
    conn_stat.weight = 0;
    float *d_weight_pt
      = h_ConnectionGroupTargetWeight[i_group*n_spike_buffer+i_source] + i_conn;
    gpuErrchk(cudaMemcpy(&conn_stat.weight, d_weight_pt, sizeof(float),
			 cudaMemcpyDeviceToHost));
  }
  return conn_stat;
}

std::vector<ConnectionStatus> NeuronGPU::GetConnectionStatus(std::vector
							     <ConnectionId>
							     &conn_id_vect) {
  std::vector<ConnectionStatus> conn_stat_vect;
  for (unsigned int i=0; i<conn_id_vect.size(); i++) {
    ConnectionStatus conn_stat = GetConnectionStatus(conn_id_vect[i]);
    conn_stat_vect.push_back(conn_stat);
  }
  return conn_stat_vect;
}
  
std::vector<ConnectionId> NeuronGPU::GetConnections(int i_source, int n_source,
						    int i_target, int n_target,
						    int syn_group) {
  if (n_source<=0) {
    i_source = 0;
    n_source = net_connection_->connection_.size();
  }
  if (n_target<=0) {
    i_target = 0;
    n_target = net_connection_->connection_.size();
  }

  return net_connection_->GetConnections<int>(i_source, n_source, i_target,
					      n_target, syn_group);    
}

std::vector<ConnectionId> NeuronGPU::GetConnections(int *i_source, int n_source,
						    int i_target, int n_target,
						    int syn_group) {
  if (n_target<=0) {
    i_target = 0;
    n_target = net_connection_->connection_.size();
  }
    
  return net_connection_->GetConnections<int*>(i_source, n_source, i_target,
					       n_target, syn_group);
  
}


std::vector<ConnectionId> NeuronGPU::GetConnections(int i_source, int n_source,
						    int *i_target, int n_target,
						    int syn_group) {
  if (n_source<=0) {
    i_source = 0;
    n_source = net_connection_->connection_.size();
  }
  
  return net_connection_->GetConnections<int>(i_source, n_source, i_target,
					      n_target, syn_group);    
}

std::vector<ConnectionId> NeuronGPU::GetConnections(int *i_source, int n_source,
						    int *i_target, int n_target,
						    int syn_group) {
  
  return net_connection_->GetConnections<int*>(i_source, n_source, i_target,
					       n_target, syn_group);
  
}


std::vector<ConnectionId> NeuronGPU::GetConnections(NodeSeq source,
						    NodeSeq target,
						    int syn_group) {
  return net_connection_->GetConnections<int>(source.i0, source.n, target.i0,
					      target.n, syn_group);
}

std::vector<ConnectionId> NeuronGPU::GetConnections(std::vector<int> source,
						    NodeSeq target,
						    int syn_group) {
  return net_connection_->GetConnections<int*>(source.data(), source.size(),
					       target.i0, target.n,
					       syn_group);
}


std::vector<ConnectionId> NeuronGPU::GetConnections(NodeSeq source,
						    std::vector<int> target,
						    int syn_group) {
  return net_connection_->GetConnections<int>(source.i0, source.n,
					      target.data(), target.size(),
					      syn_group);
}

std::vector<ConnectionId> NeuronGPU::GetConnections(std::vector<int> source,
						    std::vector<int> target,
						    int syn_group) {
  return net_connection_->GetConnections<int*>(source.data(), source.size(),
					       target.data(), target.size(),
					       syn_group);
}

int NeuronGPU::ActivateSpikeCount(int i_node, int n_node)
{
  CheckUncalibrated("Spike count must be activated before calibration");
  int i_group;
  int i_node_0 = GetNodeSequenceOffset(i_node, n_node, i_group);
  if (i_node_0!=i_node || node_vect_[i_group]->n_node_!=n_node) {
    throw ngpu_exception("Spike count must be activated for all and only "
			 " the nodes of the same group");
  }
  node_vect_[i_group]->ActivateSpikeCount();

  return 0;
}

int NeuronGPU::ActivateRecSpikeTimes(int i_node, int n_node,
				     int max_n_rec_spike_times)
{
  CheckUncalibrated("Spike time recording must be activated "
		    "before calibration");
  int i_group;
  int i_node_0 = GetNodeSequenceOffset(i_node, n_node, i_group);
  if (i_node_0!=i_node || node_vect_[i_group]->n_node_!=n_node) {
    throw ngpu_exception("Spike count must be activated for all and only "
			 " the nodes of the same group");
  }
  node_vect_[i_group]->ActivateRecSpikeTimes(max_n_rec_spike_times);

  return 0;
}

int NeuronGPU::GetNRecSpikeTimes(int i_node)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, 1, i_group);
  return node_vect_[i_group]->GetNRecSpikeTimes(i_neuron);
}

std::vector<float> NeuronGPU::GetRecSpikeTimes(int i_node)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, 1, i_group);
  return node_vect_[i_group]->GetRecSpikeTimes(i_neuron);
}

int NeuronGPU::PushSpikesToNodes(int n_spikes, int *node_id,
				 float *spike_height)
{
  int *d_node_id;
  float *d_spike_height;
  gpuErrchk(cudaMalloc(&d_node_id, n_spikes*sizeof(int)));
  gpuErrchk(cudaMalloc(&d_spike_height, n_spikes*sizeof(float)));
  gpuErrchk(cudaMemcpy(d_node_id, node_id, n_spikes*sizeof(int),
		       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_spike_height, spike_height, n_spikes*sizeof(float),
		       cudaMemcpyHostToDevice));
  PushSpikeFromRemote<<<(n_spikes+1023)/1024, 1024>>>(n_spikes, d_node_id,
						     d_spike_height);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  gpuErrchk(cudaFree(d_node_id));
  gpuErrchk(cudaFree(d_spike_height));

  return 0;
}

int NeuronGPU::PushSpikesToNodes(int n_spikes, int *node_id)
{
  //std::cout << "n_spikes: " << n_spikes << "\n";
  //for (int i=0; i<n_spikes; i++) {
  //  std::cout << node_id[i] << " ";
  //}
  //std::cout << "\n";

  int *d_node_id;
  gpuErrchk(cudaMalloc(&d_node_id, n_spikes*sizeof(int)));
  gpuErrchk(cudaMemcpy(d_node_id, node_id, n_spikes*sizeof(int),
		       cudaMemcpyHostToDevice));  
  PushSpikeFromRemote<<<(n_spikes+1023)/1024, 1024>>>(n_spikes, d_node_id);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  gpuErrchk(cudaFree(d_node_id));

  return 0;
}

int NeuronGPU::GetExtNeuronInputSpikes(int *n_spikes, int **node, int **port,
				       float **spike_height, bool include_zeros)
{
  ext_neuron_input_spike_node_.clear();
  ext_neuron_input_spike_port_.clear();
  ext_neuron_input_spike_height_.clear();
  
  for (unsigned int i=0; i<node_vect_.size(); i++) {
    if (node_vect_[i]->IsExtNeuron()) {
      int n_node;
      int n_port;
      float *sh = node_vect_[i]->GetExtNeuronInputSpikes(&n_node, &n_port);
      for (int i_neur=0; i_neur<n_node; i_neur++) {
	int i_node = i_neur + node_vect_[i]->i_node_0_;
	for (int i_port=0; i_port<n_port; i_port++) {
	  int j = i_neur*n_port + i_port;
	  if (sh[j] != 0.0 || include_zeros) {
	    ext_neuron_input_spike_node_.push_back(i_node);
	    ext_neuron_input_spike_port_.push_back(i_port);
	    ext_neuron_input_spike_height_.push_back(sh[j]);
	  }
	}
      }	
    }
  }
  *n_spikes = ext_neuron_input_spike_node_.size();
  *node = ext_neuron_input_spike_node_.data();
  *port = ext_neuron_input_spike_port_.data();
  *spike_height = ext_neuron_input_spike_height_.data();
  
  return 0;
}

int NeuronGPU::SetNeuronGroupParam(int i_node, int n_node,
				   std::string param_name, float val)
{
  int i_group;
  int i_node_0 = GetNodeSequenceOffset(i_node, n_node, i_group);
  if (i_node_0!=i_node || node_vect_[i_group]->n_node_!=n_node) {
    throw ngpu_exception(std::string("Group parameter ") + param_name
			 + " can only be set for all and only "
			 " the nodes of the same group");
  }
  return node_vect_[i_group]->SetGroupParam(param_name, val);
}

int NeuronGPU::IsNeuronGroupParam(int i_node, std::string param_name)
{
  int i_group;
  int i_node_0 = GetNodeSequenceOffset(i_node, 1, i_group);

  return node_vect_[i_group]->IsGroupParam(param_name);
}

float NeuronGPU::GetNeuronGroupParam(int i_node, std::string param_name)
{
  int i_group;
  int i_node_0 = GetNodeSequenceOffset(i_node, 1, i_group);

  return node_vect_[i_group]->GetGroupParam(param_name);
}

std::vector<std::string> NeuronGPU::GetGroupParamNames(int i_node)
{
  if (i_node<0 || i_node>(int)node_group_map_.size()) {
    throw ngpu_exception("Unrecognized node in reading group parameter names");
  }
  int i_group = node_group_map_[i_node];
  
  return node_vect_[i_group]->GetGroupParamNames();
}

int NeuronGPU::GetNGroupParam(int i_node)
{
  if (i_node<0 || i_node>(int)node_group_map_.size()) {
    throw ngpu_exception("Unrecognized node in reading number of "
			 "group parameters");
  }
  int i_group = node_group_map_[i_node];
  
  return node_vect_[i_group]->GetNGroupParam();
}

// Connect spike buffers of remote source nodes to local target nodes
// Maybe move this in connect_rules.cpp ? And parallelize with OpenMP?
int NeuronGPU::ConnectRemoteNodes()
{
  if (n_remote_node_>0) {
    i_remote_node_0_ = node_group_map_.size();
    BaseNeuron *bn = new BaseNeuron;
    node_vect_.push_back(bn);  
    CreateNodeGroup(n_remote_node_, 0);       
    for (unsigned int i=0; i<remote_connection_vect_.size(); i++) {
      RemoteConnection rc = remote_connection_vect_[i];
      net_connection_->Connect(i_remote_node_0_ + rc.i_source_rel, rc.i_target,
			       rc.port, rc.syn_group, rc.weight, rc.delay);

    }
  }
  
  return 0;
}

int NeuronGPU::GetNFloatParam()
{
  return N_KERNEL_FLOAT_PARAM;
}

std::vector<std::string> NeuronGPU::GetFloatParamNames()
{
  std::vector<std::string> param_name_vect;
  for (int i=0; i<N_KERNEL_FLOAT_PARAM; i++) {
    param_name_vect.push_back(kernel_float_param_name[i]);
  }
  
  return param_name_vect;
}

bool NeuronGPU::IsFloatParam(std::string param_name)
{
  int i_param;
  for (i_param=0; i_param<N_KERNEL_FLOAT_PARAM; i_param++) {
    if (param_name == kernel_float_param_name[i_param]) return true;
  }
  return false;
}

int NeuronGPU::GetFloatParamIdx(std::string param_name)
{
  int i_param;
  for (i_param=0; i_param<N_KERNEL_FLOAT_PARAM; i_param++) {
    if (param_name == kernel_float_param_name[i_param]) break;
  }
  if (i_param == N_KERNEL_FLOAT_PARAM) {
    throw ngpu_exception(std::string("Unrecognized kernel float parameter ")
			 + param_name);
  }
  
  return i_param;
}

float NeuronGPU::GetFloatParam(std::string param_name)
{
  int i_param =  GetFloatParamIdx(param_name);
  switch (i_param) {
  case i_time_resolution:
    return time_resolution_;
  case i_max_spike_num_fact:
    return max_spike_num_fact_;
  case i_max_spike_per_host_fact:
    return max_spike_per_host_fact_;
  default:
    throw ngpu_exception(std::string("Unrecognized kernel float parameter ")
			 + param_name);
  }
}

int NeuronGPU::SetFloatParam(std::string param_name, float val)
{
  int i_param =  GetFloatParamIdx(param_name);

  switch (i_param) {
  case i_time_resolution:
    time_resolution_ = val;
    break;
  case i_max_spike_num_fact:
    max_spike_num_fact_ = val;
    break;
  case i_max_spike_per_host_fact:
    max_spike_per_host_fact_ = val;
    break;
  default:
    throw ngpu_exception(std::string("Unrecognized kernel float parameter ")
			 + param_name);
  }
  
  return 0;
}

int NeuronGPU::GetNIntParam()
{
  return N_KERNEL_INT_PARAM;
}

std::vector<std::string> NeuronGPU::GetIntParamNames()
{
  std::vector<std::string> param_name_vect;
  for (int i=0; i<N_KERNEL_INT_PARAM; i++) {
    param_name_vect.push_back(kernel_int_param_name[i]);
  }
  
  return param_name_vect;
}

bool NeuronGPU::IsIntParam(std::string param_name)
{
  int i_param;
  for (i_param=0; i_param<N_KERNEL_INT_PARAM; i_param++) {
    if (param_name == kernel_int_param_name[i_param]) return true;
  }
  return false;
}

int NeuronGPU::GetIntParamIdx(std::string param_name)
{
  int i_param;
  for (i_param=0; i_param<N_KERNEL_INT_PARAM; i_param++) {
    if (param_name == kernel_int_param_name[i_param]) break;
  }
  if (i_param == N_KERNEL_INT_PARAM) {
    throw ngpu_exception(std::string("Unrecognized kernel int parameter ")
			 + param_name);
  }
  
  return i_param;
}

int NeuronGPU::GetIntParam(std::string param_name)
{
  int i_param =  GetIntParamIdx(param_name);
  switch (i_param) {
  case i_rnd_seed:
    return kernel_seed_ - 12345; // see neurongpu.cu
  case i_verbosity_level:
    return verbosity_level_;
  case i_max_spike_buffer_size:
    return max_spike_buffer_size_;
  case i_remote_spike_height_flag:
#ifdef HAVE_MPI
    if (connect_mpi_->remote_spike_height_) {
      return 1;
    }
    else {
      return 0;
    }
#else
    return 0;
#endif
  default:
    throw ngpu_exception(std::string("Unrecognized kernel int parameter ")
			 + param_name);
  }
}

int NeuronGPU::SetIntParam(std::string param_name, int val)
{
  int i_param =  GetIntParamIdx(param_name);
  switch (i_param) {
  case i_rnd_seed:
    SetRandomSeed(val);
    break;
  case i_verbosity_level:
    SetVerbosityLevel(val);
    break;
  case i_max_spike_per_host_fact:
    SetMaxSpikeBufferSize(val);
    break;
  case i_remote_spike_height_flag:
#ifdef HAVE_MPI
    if (val==0) {
      connect_mpi_->remote_spike_height_ = false;
    }
    else if (val==1) {
      connect_mpi_->remote_spike_height_ = true;
    }
    else {
      throw ngpu_exception("Admissible values of remote_spike_height_flag are only 0 or 1");
    }
    break;
#else
    throw ngpu_exception("remote_spike_height_flag cannot be changed in an installation without MPI support");
#endif
  default:
    throw ngpu_exception(std::string("Unrecognized kernel int parameter ")
			 + param_name);
  }
  
  return 0;
}

RemoteNodeSeq NeuronGPU::RemoteCreate(int i_host, std::string model_name,
				      int n_node /*=1*/, int n_port /*=1*/)
{
#ifdef HAVE_MPI
  if (i_host<0 || i_host>=MpiNp()) {
    throw ngpu_exception("Invalid host index in RemoteCreate");
  }
  NodeSeq node_seq;
  if (i_host == MpiId()) {
    node_seq = Create(model_name, n_node, n_port);
  }
  MPI_Bcast(&node_seq, sizeof(NodeSeq), MPI_BYTE, i_host, MPI_COMM_WORLD);
  return RemoteNodeSeq(i_host, node_seq);
#else
  throw ngpu_exception("MPI is not available in your build");
#endif
}

__global__
void SpikeBufferUp() {
   InternSpikeBufferUpdate(0);
}

__global__
void RevSpikeBufferUp(int conn_size,int internal_loop) {
   InternRevSpikeBufferUpdate(conn_size, 0);
}

__global__
void node_vect_update(int n_nodes) {
  for (unsigned int i=0; i<(n_nodes-1); i++) {
    InternUpdate(0, 0, d_node_vect[i], 0);
  }
}

__global__
void ClearGetSpikeArraysKernel(int n_nodes) {
  InternClearGetSpikeArrays(n_nodes);
}

__global__
void InternSpikeResetKernel(){
  InternSpikeReset();
}

__global__
void InternRevSpikeResetKernel(){
  InternRevSpikeReset();
}

__global__
void InternGetSpikesKernel(int n_nodes)
{
  for (unsigned int i=0; i<n_nodes; i++) {
    if (d_node_vect[i]->n_port_>0) {
      InternGetSpikes
	(d_node_vect[i]->intern_get_spike_array_, d_node_vect[i]->n_node_,
	 d_node_vect[i]->n_port_,
	 d_node_vect[i]->n_var_,
	 d_node_vect[i]->port_weight_arr_,
	 d_node_vect[i]->port_weight_arr_step_,
	 d_node_vect[i]->port_weight_port_step_,
	 d_node_vect[i]->port_input_arr_,
	 d_node_vect[i]->port_input_arr_step_,
	 d_node_vect[i]->port_input_port_step_);
    }
  }
}

__global__
void InternRunKernel()
{
  if (InternSpikeNum[blockIdx.x] > 0) {
    NestedLoop::InternRun(InternSpikeNum[blockIdx.x], &InternSpikeTargetNum[InternMaxSpikeNum * blockIdx.x], 0, 0);
  }
}

__global__
void InternRevRunKernel()
{
  if (InternRevSpikeNum[blockIdx.x] > 0) {
    NestedLoop::InternRun(InternRevSpikeNum[blockIdx.x], &InternRevSpikeNConn[blockIdx.x * *nodes_per_block], 1, 0);  
  }
}

__global__
void NewInternSimulationStepKernel(int n_nodes, int conns, bool rev_conns, double t, float time_step){
  for (int internal_loop=0;internal_loop<GroupResolution;++internal_loop){
  InternSpikeBufferUpdate(internal_loop);
  __syncthreads();
  InternUpdate(0, 0, d_node_vect[0], internal_loop);
  __syncthreads(); 
  InternClearGetSpikeArrays(n_nodes);

  __syncthreads(); 
  if (InternSpikeNum[blockIdx.x] > 0) {
    NestedLoop::InternRun(InternSpikeNum[blockIdx.x], &InternSpikeTargetNum[InternMaxSpikeNum * blockIdx.x], 0, internal_loop);
  }
  __syncthreads();

  InternSendDirectSpikes(t+internal_loop*time_step, time_step/1000.0);
  __syncthreads();
  
  for (unsigned int i=0; i<n_nodes; i++) {
    if (d_node_vect[i]->n_port_>0) {
      InternGetSpikes
	(d_node_vect[i]->intern_get_spike_array_, d_node_vect[i]->n_node_,
	 d_node_vect[i]->n_port_,
	 d_node_vect[i]->n_var_,
	 d_node_vect[i]->port_weight_arr_,
	 d_node_vect[i]->port_weight_arr_step_,
	 d_node_vect[i]->port_weight_port_step_,
	 d_node_vect[i]->port_input_arr_,
	 d_node_vect[i]->port_input_arr_step_,
	 d_node_vect[i]->port_input_port_step_);
    }
  }
  __syncthreads();
  InternSpikeReset();
  __syncthreads(); 
  if (rev_conns) {
    InternRevSpikeReset();
    __syncthreads(); 
    InternRevSpikeBufferUpdate(conns, 0);
    __syncthreads(); 
    if (InternRevSpikeNum[blockIdx.x] > 0) {
      NestedLoop::InternRun(InternRevSpikeNum[blockIdx.x], &InternRevSpikeNConn[blockIdx.x * *nodes_per_block], 1, internal_loop);  
    }
  }
  __syncthreads();
  }
}

int NeuronGPU::NewInternSimulationStep(){
  NewInternSimulationStepKernel<<<NUM_BLOCKS, 1024>>>(node_vect_.size(), net_connection_->connection_.size(), net_connection_->NRevConnections()>0, neural_time_, time_resolution_); 
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  return 0;
}

int NeuronGPU::SimulationStep()
{
  if (first_simulation_flag_) {
    StartSimulation();
  }

  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  neural_time_ = neur_t0_ + (double)time_resolution_*(it_+1);
  gpuErrchk(cudaMemcpyToSymbol(NeuronGPUTime, &neural_time_, sizeof(double)));
  long long time_idx = (int)round(neur_t0_/time_resolution_) + it_ + 1;
  gpuErrchk(cudaMemcpyToSymbol(NeuronGPUTimeIdx, &time_idx, sizeof(long long)));

  if (ConnectionSpikeTimeFlag) {
    if ( (time_idx & 0xffff) == 0x8000) {
      ResetConnectionSpikeTimeUp(net_connection_);
    }
    else if ( (time_idx & 0xffff) == 0) {
      ResetConnectionSpikeTimeDown(net_connection_);
    }
  }
  ClearGetSpikeArrays();
  NewInternSimulationStep();

  multimeter_->WriteRecords(neural_time_);
    
  int n_spikes;
  gpuErrchk(cudaMemcpy(&n_spikes, d_SpikeNum, sizeof(int),
		       cudaMemcpyDeviceToHost));


  for (unsigned int i=0; i<node_vect_.size(); i++) {
    if (node_vect_[i]->n_port_>0) {

      int grid_dim_x = (node_vect_[i]->n_node_+1023)/1024;
      int grid_dim_y = node_vect_[i]->n_port_;
      dim3 grid_dim(grid_dim_x, grid_dim_y);
      //dim3 block_dim(1024,1);
					    
      GetSpikes<<<grid_dim, 1024>>> //block_dim>>>
	(node_vect_[i]->get_spike_array_, node_vect_[i]->n_node_,
	 node_vect_[i]->n_port_,
	 node_vect_[i]->n_var_,
	 node_vect_[i]->port_weight_arr_,
	 node_vect_[i]->port_weight_arr_step_,
	 node_vect_[i]->port_weight_port_step_,
	 node_vect_[i]->port_input_arr_,
	 node_vect_[i]->port_input_arr_step_,
	 node_vect_[i]->port_input_port_step_);
    }
  }
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  SpikeReset<<<1, 1>>>();
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  it_+=GroupResolution;
  
  return 0;
}