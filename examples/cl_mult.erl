-module(cl_mult).

-compile(export_all).

-include("../include/cl.hrl").

-define(GLOBAL_DIM, 1024).
-define(LOCAL_DIM, 128).

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
  MemI = cl_data:buffer_mem(InDescriptors),
  MemO = cl_data:buffer_mem(OutDescriptors),
  {ok, Event1} = cl_util:enqueue_write_buffers(Queue, InDescriptors),
  cl_util:set_kernel_args(Kernel, MemI ++ MemO),
  {ok, Event2} = cl:enqueue_nd_range_kernel(Queue, Kernel, [?GLOBAL_DIM], [?LOCAL_DIM], []),
  {ok, Event3} = cl_util:enqueue_read_buffers(Queue, OutDescriptors).
  

