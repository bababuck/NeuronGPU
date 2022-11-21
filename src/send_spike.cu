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
#include "send_spike.h"
#include "cuda_error.h"

int *d_SpikeNum;
int *d_SpikeSourceIdx;
int *d_SpikeConnIdx;
float *d_SpikeHeight;
int *d_SpikeTargetNum;

int *d_InternSpikeNum;
int *d_InternSpikeSourceIdx;
int *d_InternSpikeConnIdx;
float *d_InternSpikeHeight;
int *d_InternSpikeTargetNum;

__device__ int MaxSpikeNum;
__device__ int *SpikeNum;
__device__ int *SpikeSourceIdx;
__device__ int *SpikeConnIdx;
__device__ float *SpikeHeight;
__device__ int *SpikeTargetNum;

__device__ int InternMaxSpikeNum;
__device__ int *InternSpikeNum;
__device__ int *InternSpikeSourceIdx;
__device__ int *InternSpikeConnIdx;
__device__ float *InternSpikeHeight;
__device__ int *InternSpikeTargetNum;
__device__ int *nodes_per_block;

__device__ void SendSpike(int i_source, int i_conn, float height,
			  int target_num)
{
//  printf("SPIKED\n");
  int pos = atomicAdd(SpikeNum, 1);
  if (pos>=MaxSpikeNum) {
    printf("Number of spikes larger than MaxSpikeNum: %d\n", MaxSpikeNum);
    *SpikeNum = MaxSpikeNum;
    return;
  }
  SpikeSourceIdx[pos] = i_source;
  SpikeConnIdx[pos] = i_conn;
  SpikeHeight[pos] = height;
  SpikeTargetNum[pos] = target_num;

  pos = atomicAdd(&InternSpikeNum[blockIdx.x], 1);
  if (pos>=InternMaxSpikeNum) {
    printf("Number of spikes larger than InternMaxSpikeNum: %d\n", InternMaxSpikeNum);
    *InternSpikeNum = InternMaxSpikeNum;
    return;
  }
  pos = blockIdx.x * InternMaxSpikeNum + pos;
  InternSpikeSourceIdx[pos] = i_source;
  InternSpikeConnIdx[pos] = i_conn;
  InternSpikeHeight[pos] = height;
  InternSpikeTargetNum[pos] = target_num;

}

__global__ void DeviceSpikeInit(int *spike_num, int *spike_source_idx,
				int *spike_conn_idx, float *spike_height,
				int *spike_target_num,
				int max_spike_num)
{
  SpikeNum = spike_num;
  SpikeSourceIdx = spike_source_idx;
  SpikeConnIdx = spike_conn_idx;
  SpikeHeight = spike_height;
  SpikeTargetNum = spike_target_num;
  MaxSpikeNum = max_spike_num;
  *SpikeNum = 0;
}

__global__ void DeviceInternSpikeInit(int *spike_num, int *spike_source_idx,
				      int *spike_conn_idx, float *spike_height,
				      int *spike_target_num,
				      int max_spike_num, int num_blocks)
{
  InternSpikeNum = spike_num;
  InternSpikeSourceIdx = spike_source_idx;
  InternSpikeConnIdx = spike_conn_idx;
  InternSpikeHeight = spike_height;
  InternSpikeTargetNum = spike_target_num;
  InternMaxSpikeNum = max_spike_num;
  for (int i=0;i<num_blocks;++i){
    InternSpikeNum[i] = 0;
  }
}

void SpikeInit(int max_spike_num, int num_blocks)
{
  //h_SpikeTargetNum = new int[PrefixScan::AllocSize];

  gpuErrchk(cudaMalloc(&d_SpikeNum, sizeof(int)));
  gpuErrchk(cudaMalloc(&d_SpikeSourceIdx, max_spike_num*sizeof(int)));
  gpuErrchk(cudaMalloc(&d_SpikeConnIdx, max_spike_num*sizeof(int)));
  gpuErrchk(cudaMalloc(&d_SpikeHeight, max_spike_num*sizeof(float)));
  gpuErrchk(cudaMalloc(&d_SpikeTargetNum, max_spike_num*sizeof(int)));

  gpuErrchk(cudaMalloc(&d_InternSpikeNum, sizeof(int)*num_blocks));
  gpuErrchk(cudaMalloc(&d_InternSpikeSourceIdx, max_spike_num*sizeof(int)*num_blocks));
  gpuErrchk(cudaMalloc(&d_InternSpikeConnIdx, max_spike_num*sizeof(int)*num_blocks));
  gpuErrchk(cudaMalloc(&d_InternSpikeHeight, max_spike_num*sizeof(float)*num_blocks));
  gpuErrchk(cudaMalloc(&d_InternSpikeTargetNum, max_spike_num*sizeof(int)*num_blocks));
  
  //printf("here: SpikeTargetNum size: %d", max_spike_num);
  DeviceSpikeInit<<<1,1>>>(d_SpikeNum, d_SpikeSourceIdx, d_SpikeConnIdx,
			   d_SpikeHeight, d_SpikeTargetNum, max_spike_num);
  DeviceInternSpikeInit<<<1,1>>>(d_InternSpikeNum, d_InternSpikeSourceIdx,
  			         d_InternSpikeConnIdx, d_InternSpikeHeight,
			   	 d_InternSpikeTargetNum, max_spike_num, num_blocks);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
}

__global__ void SpikeReset()
{
  *SpikeNum = 0;
}

__device__ void InternSpikeReset()
{
  if (!(threadIdx.x ||
        threadIdx.y)) {
    InternSpikeNum[blockIdx.x] = 0;
  }
}
