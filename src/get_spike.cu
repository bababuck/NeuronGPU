/*
Copyright (C) 2020 Bruno Golosio
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

#include "neurongpu.h"
#include "node_group.h"
#include "send_spike.h"
#include "spike_buffer.h"
#include "cuda_error.h"

extern __constant__ long long NeuronGPUTimeIdx;
extern __constant__ float NeuronGPUTimeResolution;
extern __constant__ NodeGroupStruct NodeGroupArray[];
extern __device__ signed char *NodeGroupMap;

extern __device__ int InternMaxSpikeNum;
extern __device__ int *InternSpikeNum;
extern __device__ int *InternSpikeSourceIdx;
extern __device__ int *InternSpikeConnIdx;
extern __device__ float *InternSpikeHeight;
extern __device__ int *InternSpikeTargetNum;
extern __device__ int *nodes_per_block;

extern __device__ void SynapseUpdate(int syn_group, float *w, float Dt);

__device__ double atomicAddDouble(double* address, double val)
{
    unsigned long long int* address_as_ull =
                                          (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, 
                        __double_as_longlong(val + 
                        __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

//////////////////////////////////////////////////////////////////////
// This is the function called by the nested loop
// that collects the spikes
__device__ void NestedLoopFunction0(int i_spike, int i_syn, bool all_connections, int _it)
{
  int i_source;
  int i_conn;
  float height;
  if (all_connections) {
     i_source = SpikeSourceIdx[i_spike];
     i_conn = SpikeConnIdx[i_spike];
     height = SpikeHeight[i_spike];
  } else {
     i_spike += InternMaxSpikeNum * blockIdx.x;
     i_source = InternSpikeSourceIdx[i_spike];
     i_conn = InternSpikeConnIdx[i_spike];
     height = InternSpikeHeight[i_spike];
  }
  unsigned int target_port
    = ConnectionGroupTargetNode[i_conn*NSpikeBuffer + i_source][i_syn];
  int i_target = target_port & PORT_MASK;
  unsigned char port = (unsigned char)(target_port >> (PORT_N_SHIFT + 24));
  unsigned char syn_group
    = ConnectionGroupTargetSynGroup[i_conn*NSpikeBuffer + i_source][i_syn];
  float weight = ConnectionGroupTargetWeight[i_conn*NSpikeBuffer+i_source]
    [i_syn];
  //printf("handles spike %d src %d conn %d syn %d target %d"
  //" port %d weight %f\n",
  //i_spike, i_source, i_conn, i_syn, i_target,
  //port, weight);
  
  /////////////////////////////////////////////////////////////////
  int i_group=NodeGroupMap[i_target];
  int i = port*NodeGroupArray[i_group].n_node_ + i_target
    - NodeGroupArray[i_group].i_node_0_;
  double d_val = (double)(height*weight);
  if (blockIdx.x != (i_target / *nodes_per_block)) {
    atomicAddDouble(&NodeGroupArray[i_group].get_spike_array_[i], d_val);    
  } else {
    atomicAddDouble(&NodeGroupArray[i_group].intern_get_spike_array_[i], d_val);
  }

  if (syn_group>0) {
    ConnectionGroupTargetSpikeTime[i_conn*NSpikeBuffer+i_source][i_syn]
      = (unsigned short)((NeuronGPUTimeIdx+_it) & 0xffff);
    
    long long Dt_int = (NeuronGPUTimeIdx+_it) - LastRevSpikeTimeIdx[i_target];
     if (Dt_int>0 && Dt_int<MAX_SYN_DT) {
       SynapseUpdate(syn_group, &ConnectionGroupTargetWeight
		    [i_conn*NSpikeBuffer+i_source][i_syn],
		     -NeuronGPUTimeResolution*Dt_int);
    }
  }
  ////////////////////////////////////////////////////////////////
}
///////////////

// improve using a grid
/*
__global__ void GetSpikes(double *spike_array, int array_size, int n_port,
			  int n_var,
			  float *port_weight_arr,
			  int port_weight_arr_step,
			  int port_weight_port_step,
			  float *port_input_arr,
			  int port_input_arr_step,
			  int port_input_port_step)
{
  int i_array = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_array < array_size*n_port) {
     int i_target = i_array % array_size;
     int port = i_array / array_size;
     int port_input = i_target*port_input_arr_step
       + port_input_port_step*port;
     int port_weight = i_target*port_weight_arr_step
       + port_weight_port_step*port;
     double d_val = (double)port_input_arr[port_input]
       + spike_array[i_array]
       * port_weight_arr[port_weight];

     port_input_arr[port_input] = (float)d_val;
  }
}
*/

__global__ void GetSpikes(double *spike_array, int array_size, int n_port,
			  int n_var,
			  float *port_weight_arr,
			  int port_weight_arr_step,
			  int port_weight_port_step,
			  float *port_input_arr,
			  int port_input_arr_step,
			  int port_input_port_step)
{
  int i_target = blockIdx.x*blockDim.x+threadIdx.x;
  int port = blockIdx.y*blockDim.y+threadIdx.y;

  if (i_target < array_size && port<n_port) {
    int i_array = port*array_size + i_target;
    int port_input = i_target*port_input_arr_step
      + port_input_port_step*port;
    int port_weight = i_target*port_weight_arr_step
      + port_weight_port_step*port;
    double d_val = (double)port_input_arr[port_input]
      + spike_array[i_array]
      * port_weight_arr[port_weight];
//    if (spike_array[i_array] * port_weight_arr[port_weight] > 0 && !threadIdx.x && !blockIdx.x) printf("Extern\n");
    port_input_arr[port_input] = (float)d_val;
  }
}

#define NUM_BLOCKS 30

__device__ void InternGetSpikes(double *spike_array, int nodes, int ports,
			  int n_var,
			  float *port_weight_arr,
			  int port_weight_arr_step,
			  int port_weight_port_step,
			  float *port_input_arr,
			  int port_input_arr_step,
			  int port_input_port_step)
{
      int nodes_p_block = ((nodes + NUM_BLOCKS-1)/NUM_BLOCKS);
      int start_node = blockIdx.x * nodes_p_block;
      int finish_node = start_node + nodes_p_block; 
      for (int i_target = threadIdx.x + start_node; i_target<finish_node;i_target+=blockDim.x) {
        for (int port=0;port<ports;++port){
          if (i_target < nodes) {
            int i_array = port*nodes + i_target;
            int port_input = i_target*port_input_arr_step
              + port_input_port_step*port;
            int port_weight = i_target*port_weight_arr_step
              + port_weight_port_step*port;
            double d_val = (double)port_input_arr[port_input]
              + spike_array[i_array]
              * port_weight_arr[port_weight];
            port_input_arr[port_input] = (float)d_val;
          }
        }
      }
}

int NeuronGPU::ClearGetSpikeArrays()
{
  for (unsigned int i=0; i<node_vect_.size(); i++) {
    BaseNeuron *bn = node_vect_[i];
    if (bn->get_spike_array_ != NULL) {
      gpuErrchk(cudaMemset(bn->get_spike_array_, 0, bn->n_node_*bn->n_port_
			   *sizeof(double)));
    }
  } 
  return 0;
}

int NeuronGPU::FreeGetSpikeArrays()
{
  for (unsigned int i=0; i<node_vect_.size(); i++) {
    BaseNeuron *bn = node_vect_[i];
    if (bn->get_spike_array_ != NULL) {
      gpuErrchk(cudaFree(bn->get_spike_array_));
    }
    if (bn->intern_get_spike_array_ != NULL) {
      gpuErrchk(cudaFree(bn->intern_get_spike_array_));
    }
  }
  
  return 0;
}
