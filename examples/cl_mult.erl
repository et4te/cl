-module(cl_mult).

-compile(export_all).

-include("../include/cl.hrl").

-define(GLOBAL_WORK_SIZE, 4).
-define(LOCAL_WORK_SIZE, 1).

%%------------------------------------------------------------------------------
%% Sets up the OpenCL context, creates input and output buffers, ...
%%------------------------------------------------------------------------------
test() ->
  Cl = clu:setup(all),

  %%------------------------------------------------------------------------------
  %% Note: Each kernel may have a source(), input() and output() parameters or
  %%       a similar API for a parse transform to be created.
  %%------------------------------------------------------------------------------
  KernelSource = cl_mult_kernel:source(),
  KernelI = cl_mult_kernel:input(),
  KernelO = cl_mult_kernel:output(),

  io:format("Building kernel from source...~n"),
  {ok, Program, Kernel} = 
    cl_util:build_kernel("mmult", Cl#cl.context, KernelSource, Cl#cl.devices),

  I = cl_util:create_buffers(Cl#cl.context, KernelI),
  O = cl_util:create_buffers(Cl#cl.context, KernelO),

  Queues = 
    lists:map(
      fun (Device) ->
          {ok, Queue} = cl:create_queue(Cl#cl.context, Device, []),
          Queue
      end, Cl#cl.devices
     ),
  
  lists:foreach(
    fun (Queue) ->
        execute_kernel_directives(Queue, Kernel, I, O)
    end, Queues
   ),

  cl_util:release(Program, Kernel, Queues, I, O),

  clu:teardown(Cl).

execute_kernel_directives(Queue, Kernel, InDescriptors, OutDescriptors) ->  
  io:format("Executing kernel directives...~n"),

  KernelIArgs = cl_data:kernel_args(InDescriptors),
  KernelOArgs = cl_data:kernel_args(OutDescriptors),

  io:format("KernelIArgs = ~p~n", [KernelIArgs]),
  io:format("KernelOArgs = ~p~n", [KernelOArgs]),

  io:format("Calling enqueue write buffers...~n"),
  {ok, E1} = cl_util:enqueue_write_buffers(Queue, InDescriptors, []),

  io:format("Setting kernel arguments... (~p)~n", [KernelIArgs ++ KernelOArgs]),
  cl_util:set_kernel_args(Kernel, KernelIArgs ++ KernelOArgs),

  io:format("Calling EnqueueNDRange...~n"),
  {ok, E2} = cl:enqueue_nd_range_kernel(Queue, Kernel,
                                        [?GLOBAL_WORK_SIZE, ?GLOBAL_WORK_SIZE],
                                        [?LOCAL_WORK_SIZE, ?LOCAL_WORK_SIZE],
                                        [E1]),

  io:format("Calling enqueue read buffers...~n"),
  {ok, E3} = cl_util:enqueue_read_buffers(Queue, OutDescriptors, [E2]),

  ok = cl:flush(Queue),

  io:format("Queue flushed...~n"),

  io:format("E1 = ~p~n", [cl:wait(E1)]),
  io:format("E2 = ~p~n", [cl:wait(E2)]),

  {ok, E3Bin} = cl:wait(E3),

  io:format("E3 Data = ~p~n", [cl_util:to_list(E3Bin)]).

