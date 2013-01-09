-module(cl_mult).

-compile(export_all).

-include("../include/cl.hrl").

-define(GLOBAL_DIM, 1024).
-define(LOCAL_DIM, 16).

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
	execute_kernel_directives(CommandQueue, Kernel, I, O)
    end, CommandQueues
   ).

execute_kernel_directives(Queue, Kernel, InDescriptors, OutDescriptors) ->  
  io:format("Executing kernel directives...~n"),
  MemI = cl_data:buffer_mem(InDescriptors),
  MemO = cl_data:buffer_mem(OutDescriptors),

  io:format("MemI = ~p~n", [MemI]),
  io:format("MemO = ~p~n", [MemO]),

  io:format("Calling enqueue write buffers...~n"),
  {ok, Event1} = cl_util:enqueue_write_buffers(Queue, InDescriptors, []),
  io:format("Setting kernel arguments...~n"),
  cl_util:set_kernel_args(Kernel, MemI ++ MemO),
  io:format("Calling enqueue nd range...~n"),
  {ok, Event2} = cl:enqueue_nd_range_kernel(Queue, Kernel, 
					    [?GLOBAL_DIM, ?GLOBAL_DIM], 
					    [?LOCAL_DIM, ?LOCAL_DIM], [Event1]),
  io:format("Calling enqueue read buffers...~n"),
  {ok, Event3} = cl_util:enqueue_read_buffers(Queue, OutDescriptors, [Event2]),
  cl:flush(Queue).
  

