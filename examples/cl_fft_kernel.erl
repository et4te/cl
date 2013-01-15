-module(cl_fft_kernel).

-export([source/0, input/0, output/0]).

-define(X_DIM, 16*2).
-define(Y_DIM, 16*2).

%%------------------------------------------------------------------------------
%% Sample data
%%------------------------------------------------------------------------------
x() ->
  << <<X:32/native-float>> || X <- lists:seq(1, ?X_DIM) >>.

y() ->
  << <<Y:32/native-float>> || Y <- lists:seq(1, ?Y_DIM) >>.

%%------------------------------------------------------------------------------
%% Inputs / Outputs
%%------------------------------------------------------------------------------
input() ->
  [{x(), 4, [read_write]},
   4].

output() ->
  [{y(), 4, [read_write]}].

%%------------------------------------------------------------------------------
%% Kernel source
%%------------------------------------------------------------------------------
source() ->
  "
typedef float real_t;
typedef float2 real2_t;

#define FFT_PI 3.14159265359f
#define FFT_SQRT_1_2 0.707106781187f

// return A * B
real2_t mul(real2_t a, real2_t b)
{
#if USE_MAD
  return (real2_t)(mad(a.x,b.x,-a.y*b.y),mad(a.x,b.y,a.y*b.x));
#else
  return (real2_t)(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x);
#endif
}

// return A * exp(K*ALPHA*i)
real2_t twiddle(real2_t a, int k, real_t alpha)
{
  real_t cs,sn;
  sn = sincos((real_t)k*alpha,&cs);
  return mul(a,(real2_t)(cs,sn));
}

#define DFT2(a,b) { real2_t tmp = a - b; a += b; b = tmp; }

__kernel void fftRadix2Kernel(__global const real_t* x, int p, __global real_t* y)
{
  int t = get_global_size(0); // thread count
  int i = get_global_id(0);   // thread index
  int k = i&(p-1);            // index in input sequence, in 0..P-1
  int j = ((i-k)<<1) + k;     // output index
  real_t alpha = -FFT_PI*(real_t)k/(real_t)p;
  
  // read and twiddle input
  x += i * 2;
  real2_t u0 = (x[0],x[1]);
  real2_t u1 = (x[t],x[t+1]);
  real2_t u2 = twiddle(u1,1,alpha);

  // in-place DFT-2
  DFT2(u0,u2);

  y += j * 2;
  y[0] = u0.x;
  y[1] = u0.y;
  y[p] = u2.x;
  y[p+1] = u2.y;
}".
