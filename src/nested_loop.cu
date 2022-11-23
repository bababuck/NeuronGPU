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
#include <stdlib.h>

				    //#include "cuda_error_nl.h"
#include "cuda_error.h"
#include "nested_loop.h"

//TMP
#include "getRealTime.h"
//

//////////////////////////////////////////////////////////////////////
// declare here the functions called by the nested loop 
__device__ void NestedLoopFunction0(int ix, int iy);
__device__ void NestedLoopFunction1(int ix, int iy);
//////////////////////////////////////////////////////////////////////

namespace NestedLoop
{
  PrefixScan prefix_scan_;
  int *d_Ny_cumul_sum_;   
}

__device__ int locate(int val, int *data, int n)
{
  int i_left = 0;
  int i_right = n-1;
  int i = (i_left+i_right)/2;
  while(i_right-i_left>1) {
    if (data[i] > val) i_right = i;
    else if (data[i]<val) i_left = i;
    else break;
    i=(i_left+i_right)/2;
  }

  return i;
}

__device__ void CumulSumNestedLoopKernel0(int Nx, int *Ny_cumul_sum,
					 int Ny_sum)
{
  int stride = blockDim.x;
  for (int array_idx=threadIdx.x;i<Ny_sum;array_idx+=stride) {
    if (array_idx<Ny_sum) {
      int ix = locate(array_idx, Ny_cumul_sum, Nx + 1);
      int iy = (int)(array_idx - Ny_cumul_sum[ix]);
      NestedLoopFunction0(ix, iy);
    }
  }
}

__device__ void CumulSumNestedLoopKernel1(int Nx, int *Ny_cumul_sum,
					 int Ny_sum)
{
  int stride = blockDim.x;
  for (int array_idx=threadIdx.x;i<Ny_sum;array_idx+=stride) {
    if (array_idx<Ny_sum) {
      int ix = locate(array_idx, Ny_cumul_sum, Nx + 1);
      int iy = (int)(array_idx - Ny_cumul_sum[ix]);
      NestedLoopFunction1(ix, iy);
    }
  }
}

//////////////////////////////////////////////////////////////////////
int NestedLoop::Init()
{
  //prefix_scan_.Init();
  gpuErrchk(cudaMalloc(&d_Ny_cumul_sum_,
			  PrefixScan::AllocSize*sizeof(int)));
  
  return 0;
}

//////////////////////////////////////////////////////////////////////
__device__
int NestedLoop::Run(int Nx, int *d_Ny, int i_func)
{
  return CumulSumNestedLoop(Nx, d_Ny, i_func);
}


//////////////////////////////////////////////////////////////////////
__device__
int NestedLoop::CumulSumNestedLoop(int Nx, int *d_Ny, int i_func)
{
  simple_scan(d_Ny_cumul_sum_, d_Ny, Nx+1);

  int Ny_sum;
  gpuErrchk(cudaMemcpy(&Ny_sum, &d_Ny_cumul_sum_[Nx],
			  sizeof(int), cudaMemcpyDeviceToHost));

  ////
  if(Ny_sum>0) {
    switch (i_func) {
    case 0:
      CumulSumNestedLoopKernel0
	(Nx, d_Ny_cumul_sum_, Ny_sum);
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());
      break;
    case 1:
      CumulSumNestedLoopKernel1
	(Nx, d_Ny_cumul_sum_, Ny_sum);
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());
      break;
    default:
      throw ngpu_exception("unknown nested loop function");
    }

    //TMP
    //printf("cst: %lf\n", getRealTime()-time_mark);
    //
  }
    
  return 0;
}

__device__
void simple_scan(int *out, int *data, int length)
{
  if (!threadIdx.x) {
    out[0] = 0;
    for (int i=1;i<length;++i) {
      out[i] = data[i-1] + out[i-1];
    }
  }
  __thread_sync();
  return;
}