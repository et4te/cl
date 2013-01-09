%%------------------------------------------------------------------------------
%% Author: Edward Tate <edward.tate@erlang-solutions.com>
%%------------------------------------------------------------------------------
-module(cl_mult_kernel).

-export([source/0, input/0, output/0]).

%%------------------------------------------------------------------------------
%% Sample dimensions
%%------------------------------------------------------------------------------
-define(WA, 128).
-define(HA, 128).
-define(WB, 128).
-define(HB, 128).
-define(WC, 128).
-define(HC, 128).

%%------------------------------------------------------------------------------
%% Sample data
%%------------------------------------------------------------------------------
matA() ->
  << <<X:32/native-float>> || X <- lists:seq(1, ?WA * ?HA) >>.
matB() ->
  << <<X:32/native-float>> || X <- lists:seq(1, ?WB * ?HB) >>.
matC() ->
  << <<X:32/native-float>> || X <- lists:seq(1, ?WC * ?HC) >>.
widthA() ->
  << <<X:32/integer>> || X <- lists:seq(0, 1) >>.
widthB() ->
  << <<X:32/integer>> || X <- lists:seq(0, 1) >>.

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

%%------------------------------------------------------------------------------
%% Kernel parameters
%%------------------------------------------------------------------------------
input() ->
  [{matA(), 4, [read_only]},
   {matB(), 4, [read_only]},
   {widthA(), 4, [read_only]},
   {widthB(), 4, [read_only]}].

output() ->
  [{matC(), 4, [write_only]}].
