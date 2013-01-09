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
%% @doc Create a set of buffers in main mem from a list of kernel parameters.
%%------------------------------------------------------------------------------
create_buffers(Context, KernelParameters) ->
  [cl_data:create_buffer(Context, Data, TypeSize, Opt) ||
    {Data, TypeSize, Opt} <- KernelParameters].

%%------------------------------------------------------------------------------
%% @doc Enqueues a list of write buffers to a command queue.
%%------------------------------------------------------------------------------
enqueue_write_buffers(Queue, BufferDescriptors, Events) ->
  WriteBuffers =
    lists:map(
      fun ({buffer_desc, Buffer, Data, DataSize}) ->
	  cl:enqueue_write_buffer(Queue, Buffer, 0, DataSize, Data, Events)
      end, BufferDescriptors
     ),
  lists:last(WriteBuffers).

%%------------------------------------------------------------------------------
%% @doc Enqueues a list of read buffers to a command queue.
%%------------------------------------------------------------------------------
enqueue_read_buffers(Queue, BufferDescriptors, Events) ->
  lists:foreach(
    fun ({buffer_desc, Buffer, Data, DataSize}) ->
	cl:enqueue_read_buffer(Queue, Buffer, 0, DataSize, Events)
    end, BufferDescriptors
   ).

%%------------------------------------------------------------------------------
%% @doc Builds a kernel.
%%------------------------------------------------------------------------------
build_kernel(Label, Context, Source, Devices) ->
  {ok, Program} = cl:create_program_with_source(Context, Source),
  io:format("Program created successfully, building...~n"),

  case cl:build_program(Program, Devices, "") of
    ok ->
      lists:foreach(
	fun (Device) ->
	    {ok, BuildInfo} = cl:get_program_build_info(Program, Device),
	    io:format(" * Built on device ~w:~n  ~p~n", [Device, BuildInfo])
	end, Devices
       );
    Error ->
      lists:foreach(
	fun (Device) ->
	    {ok, BuildInfo} = cl:get_program_build_info(Program, Device),
	    io:format(" * Build error ~w:~n  ~p~n", [Device, BuildInfo])
	end, Devices
       )
  end,

  io:format("Build succeeded, creating kernel...~n"),
  {ok, Kernel} = cl:create_kernel(Program, Label),
  Kernel.

%%------------------------------------------------------------------------------
%% @doc Sets the arguments passed to the kernel before it is executed.
%%------------------------------------------------------------------------------
set_kernel_args(Kernel, Args) ->
  set_kernel_args(Kernel, Args, 0).

set_kernel_args(_Kernel, [], _N) ->
  ok;
set_kernel_args(Kernel, [Arg|Args], N) ->
  cl:set_kernel_arg(Kernel, N, Arg),
  set_kernel_args(Kernel, Args, N+1).

