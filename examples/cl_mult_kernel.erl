%%------------------------------------------------------------------------------
%% Author: Edward Tate <edward.tate@erlang-solutions.com>
%%------------------------------------------------------------------------------
-module(cl_mult_kernel).

-export([source/0, input/0, output/0]).

%%------------------------------------------------------------------------------
%% Sample dimensions
%%------------------------------------------------------------------------------
-define(A_DIM, 4 * 4).
-define(B_DIM, 4 * 4).
-define(C_DIM, 4 * 4).

%%------------------------------------------------------------------------------
%% Sample data
%%------------------------------------------------------------------------------
matA() ->
  << <<X:32/native-float>> || X <- lists:seq(1, ?A_DIM) >>.
matB() ->
  << <<X:32/native-float>> || X <- lists:seq(1, ?B_DIM) >>.

matC() ->
  << <<X:32/native-float>> || X <- lists:seq(1, ?C_DIM) >>.

%%------------------------------------------------------------------------------
%% Kernel parameters
%%------------------------------------------------------------------------------
input() ->
  [{matA(), 4, [read_write]},
   {matB(), 4, [read_write]},
   4,
   4].

output() ->
  [{matC(), 4, [read_write]}].

%%------------------------------------------------------------------------------
%% Kernel source
%%------------------------------------------------------------------------------
source() ->
  "__kernel void mmult(
     /* Input */
     __global float* A, 
     __global float* B, 
     int widthA, int widthB,
     /* Output */
     __global float* C     
   ) 
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

