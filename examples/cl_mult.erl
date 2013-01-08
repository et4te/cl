-module(cl_mult).

-compile(export_all).

-include("../include/cl.hrl").

test() ->
  {cl, Platform, Devices, Context} = clu:setup(all),
  KernelSource = cl_mult_kernel:source(),
  KernelDataDescriptors = cl_mult_kernel:data_descriptors(),
  {ok, Kernel} = cl_util:build_kernel("mmult", Context, KernelSource, Devices),
  BufferDescriptors = cl_util:create_buffers(Context, KernelDataDescriptors),

  CommandQueues = 
    lists:map(
      fun (Device) ->
	  {ok, Queue} = cl:create_queue(Context, Device, []),
	  Queue
      end, Devices
     ),
  
  lists:foreach(
    fun (CommandQueue) ->
	cl_util:enqueue_write_buffers(CommandQueue, BufferDescriptors)
    end, CommandQueues
   ).
  

  
