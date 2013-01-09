-module(cl_sum).

-include("../include/cl.hrl").

-export([test/0]).

-define(GLOBAL_WORK_SIZE, 16).
-define(LOCAL_WORK_SIZE, 1).

test() ->
  Cl = clu:setup(all),
  
  KernelSource = cl_sum_kernel:source(),
  KernelI = cl_sum_kernel:input(),
  KernelO = cl_sum_kernel:output(),
  
  io:format("Building kernel from source... ~n"),
  {ok, Program, Kernel} =
    cl_util:build_kernel("mat_sum", Cl#cl.context, KernelSource, Cl#cl.devices),
  
  I = cl_util:create_buffers(Cl#cl.context, KernelI),
  O = cl_util:create_buffers(Cl#cl.context, KernelO),

  %% You can select devices by using cl_util:gpus(), cl_util:cpus()
  Devices = cl_util:devices(),

  Queues = cl_util:one_queue_per_device(Cl#cl.context, Devices, []),
  
  lists:foreach(
    fun (Queue) ->
        execute_kernel_directives(Queue, Kernel, I, O)
    end, Queues
   ),
  
  cl_util:release(Program, Kernel, Queues, I, O),
  
  clu:teardown(Cl).

execute_kernel_directives(Queue, Kernel, InDescriptors, OutDescriptors) ->
  KernelIArgs = cl_data:kernel_args(InDescriptors),
  KernelOArgs = cl_data:kernel_args(OutDescriptors),
 
  {ok, E1} = cl_util:enqueue_write_buffers(Queue, InDescriptors, []),
  
  cl_util:set_kernel_args(Kernel, KernelIArgs ++ KernelOArgs),
  
  {ok, E2} = cl:enqueue_nd_range_kernel(Queue, Kernel,
                                        [?GLOBAL_WORK_SIZE, ?GLOBAL_WORK_SIZE],
                                        [?LOCAL_WORK_SIZE, ?LOCAL_WORK_SIZE],
                                        [E1]),

  {ok, E3} = cl_util:enqueue_read_buffers(Queue, OutDescriptors, [E2]),

  ok = cl:flush(Queue),

  io:format("Queue flushed... ~n"),
  
  io:format("E1 = ~p~n", [cl:wait(E1)]),
  io:format("E2 = ~p~n", [cl:wait(E2)]),

  {ok, E3Bin} = cl:wait(E3),

  io:format("E3 Data = ~p~n", [cl_data:to_list(E3Bin)]).
