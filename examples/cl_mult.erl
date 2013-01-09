-module(cl_mult).

-compile(export_all).

-include("../include/cl.hrl").

-define(GLOBAL_WORK_SIZE, 1024).
-define(LOCAL_WORK_SIZE, 16).

%%------------------------------------------------------------------------------
test() ->
  Cl = clu:setup(all),

  %%
  %% Note: each kernel must have a source() and parameters() where parse_trans
  %%       is concerned.
  %%
  KernelSource = cl_mult_kernel:source(),
  KernelI = cl_mult_kernel:input(),
  KernelO = cl_mult_kernel:output(),

  io:format("Building kernel from source...~n"),
  Kernel = cl_util:build_kernel("mmult", Cl#cl.context, KernelSource, Cl#cl.devices),

  I = cl_util:create_buffers(Cl#cl.context, KernelI),
  O = cl_util:create_buffers(Cl#cl.context, KernelO),

  CommandQueues = 
    lists:map(
      fun (Device) ->
	  {ok, CommandQueue} = cl:create_queue(Cl#cl.context, Device, []),
	  CommandQueue
      end, Cl#cl.devices
     ),
  
  lists:foreach(
    fun (CommandQueue) ->
	{ok, E} = execute_kernel_directives(CommandQueue, Kernel, I, O),
	cl:wait(E)
    end, CommandQueues
   ).

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

  io:format("Calling enqueue nd range...~n"),
  {ok, E2} = cl:enqueue_nd_range_kernel(Queue, Kernel,
					[?GLOBAL_WORK_SIZE, ?GLOBAL_WORK_SIZE],
					[?LOCAL_WORK_SIZE, ?LOCAL_WORK_SIZE],
					[]),

  io:format("Calling enqueue read buffers...~n"),
  {ok, E3} = cl_util:enqueue_read_buffers(Queue, OutDescriptors, [E2]),

  ok = cl:flush(Queue),

  io:format("Queue flushed...~n"),

  io:format("E1 = ~p~n", [cl:wait(E1)]),
  io:format("E2 = ~p~n", [cl:wait(E2)]),
  io:format("E3 = ~p~n", [cl:wait(E3)]),

  {ok, E3}.

  

