%%------------------------------------------------------------------------------
%% Author: Edward Tate <edward.tate@erlang-solutions.com>
%%------------------------------------------------------------------------------
-module(cl_util).

-compile(export_all).

%%------------------------------------------------------------------------------
%% @doc Used to find out information about the current machine in terms of 
%%      OpenCL.
%%------------------------------------------------------------------------------
current_machine() ->
  {ok, PlatformIds} = cl:get_platform_ids(),
  lists:foreach(
    fun (PlatformId) ->
	io:format("Platform ~p~n", [PlatformId])
    end, PlatformIds
   ),
  
  DeviceIds = 
    lists:map(
      fun (PlatformId) ->
	  {ok, DeviceId} = cl:get_device_ids(PlatformId, all),
	  DeviceId
      end, PlatformIds
     ),

  lists:foreach(
    fun (DeviceId) ->
	io:format("Device ~p~n", [DeviceId])
    end, DeviceIds
   ),

  Contexts = [cl:create_context(Id) || Id <- DeviceIds],
  lists:foreach(
    fun (Context) ->
	io:format("Context ~p~n", [Context])
    end, Contexts
   ).

%%------------------------------------------------------------------------------
%% @doc Convenience function to represent generic data to be provided to OpenCL.
%%------------------------------------------------------------------------------
data_desc(Data, TypeSize, Opt) ->
  DataSize = byte_size(Data),
  {data_desc, Data, DataSize, DataSize div TypeSize, Opt}.

%%------------------------------------------------------------------------------
%% @doc Create a set of buffers in main mem from a list of data descriptors.
%%------------------------------------------------------------------------------
create_buffer_desc(Context, Data, DataSize, Opt) ->
  {ok, Buffer} = cl:create_buffer(Context, Data, Opt),
  {buffer_desc, Buffer,  Data, DataSize}.

create_buffers(Context, DataDescriptors) ->
  [create_buffer_desc(Context, Data, DataSize, Opt) ||
    {data_desc, Data, DataSize, ByteSize, Opt} <- DataDescriptors].

%%------------------------------------------------------------------------------
%% @doc Enqueues a list of write buffers to a command queue.
%%------------------------------------------------------------------------------
enqueue_write_buffers(Queue, BufferDescriptors) ->
  lists:foreach(
    fun ({buffer_desc, Buffer, Data, DataSize}) ->
	cl:enqueue_write_buffer(Queue, Buffer, 0, DataSize, Data, [])
    end, BufferDescriptors
   ),
  cl:flush(Queue).

%%------------------------------------------------------------------------------
%% @doc Builds a kernel.
%%------------------------------------------------------------------------------
build_kernel(Label, Context, Source, Devices) ->
  {ok, Program} = cl:create_program_with_source(Context, Source),

  {ok, Info} = cl:get_program_info(Program),
  io:format("Program Info: ~p\n", [Info]),

  case cl:build_program(Program, Devices, "") of
    ok ->
      lists:foreach(
	fun (Device) ->
	    {ok, BuildInfo} = cl:get_program_build_info(Program, Device),
	    io:format("Build Info @ ~w: ~p\n", [Device, BuildInfo])
	end, Devices
       );
    Error ->
      io:format("\n\nKernel Build Error: ~p\n\n", [Error]),
      lists:foreach(
	fun (Device) ->
	    {ok, BuildInfo} = cl:get_program_build_info(Program, Device),
	    io:format("Build Info @ ~w: ~p\n", [Device, BuildInfo])
	end, Devices
       )
  end,
  
  cl:create_kernel(Program, Label).

%%------------------------------------------------------------------------------
%% @doc Sets the arguments passed to the kernel before it is executed.
%%------------------------------------------------------------------------------
set_kernel_args(Kernel, [], N) ->
  ok;
set_kernel_args(Kernel, [Arg|Args], N) ->
  cl:set_kernel_arg(Kernel, N, Arg),
  set_kernel_args(Kernel, Args, N+1).

