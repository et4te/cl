%%------------------------------------------------------------------------------
%% Author: Edward Tate <edward.tate@erlang-solutions.com>
%%------------------------------------------------------------------------------
-module(cl_mult_kernel).

-export([source/0, data_descriptors/0]).

-define(WA, 128).
-define(HA, 128).
-define(WB, 128).
-define(HB, 128).
-define(WC, 128).
-define(HC, 128).

matA() ->
  << <<X:32/native-float>> || X <- lists:seq(1, ?WA * ?HA) >>.
matB() ->
  << <<X:32/native-float>> || X <- lists:seq(1, ?WB * ?HB) >>.
matC() ->
  << <<X:32/native-float>> || X <- lists:seq(1, ?WC * ?HC) >>.
rowA() ->
  <<"0">>.
colC() ->
  <<"0">>.

source() ->
  "__kernel void mmult(
     __global float* C, 
     __global float* A, 
     __global float* B, 
     int widthA, int widthB) 
   {
     int i = get_global_id(0);
     int j = get_global_id(1);
     float value = 0;
     for (int k = 0; k < widthA; k++)
     {
       value += A[k + j * widthA] * B[k * widthB + i];
     }
     C[i + widthA * j] = value;
   }".

data_descriptors() ->
  [cl_util:data_desc(matC(), 4, [read_write]),
   cl_util:data_desc(matA(), 4, [read_write]),
   cl_util:data_desc(matB(), 4, [read_write]),
   cl_util:data_desc(rowA(), 4, [read_write]),
   cl_util:data_desc(colC(), 4, [read_write])].
