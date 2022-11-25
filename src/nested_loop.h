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

#ifndef NESTED_LOOP_H
#define  NESTED_LOOP_H

#include "prefix_scan.h"

namespace NestedLoop
{
  extern PrefixScan prefix_scan_;
  
  int Init();
  int Run(int Nx, int *d_Ny, int i_func);
  __device__ int InternRun(int Nx, int *d_Ny, int i_func);
  int CumulSumNestedLoop(int Nx, int *d_Ny, int i_func);  
  __device__ int InternCumulSumNestedLoop(int Nx, int *d_Ny, int i_func);  
  
  int Free();
}

#endif
