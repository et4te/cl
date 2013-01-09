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
        io:format("Platform~n  ~p~n", [PlatformId])
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
  create_buffers(Context, KernelParameters, []).

create_buffers(_Context, [], Acc) ->
  lists:reverse(Acc);
create_buffers(Context, [KernelParameter|Rest], Acc) ->
  case KernelParameter of
    {Data, TypeSize, Opt} ->
      A = cl_data:create_buffer(Context, Data, TypeSize, Opt),
      create_buffers(Context, Rest, [A|Acc]);
    Atom when is_integer(Atom) ->
      A = {ok, Atom},
      create_buffers(Context, Rest, [A|Acc]);
    _ ->
      A = [{error, erroneous_type}],
      create_buffers(Context, Rest, [A|Acc])
  end.

%%------------------------------------------------------------------------------
%% @doc Enqueues a list of write buffers to a command queue.
%%------------------------------------------------------------------------------
enqueue_write_buffers(Queue, BufferDescriptors, Events) ->
  enqueue_write_buffers(Queue, BufferDescriptors, Events, []).

enqueue_write_buffers(_Queue, [], _Es, E) ->
  {ok, E};
enqueue_write_buffers(Queue, [{buffer_desc, Mem, Data, DataSize}|Rest], Es, _E0) ->
  {ok, E1} = cl:enqueue_write_buffer(Queue, Mem, 0, DataSize, Data, Es),
  enqueue_write_buffers(Queue, Rest, Es, E1);
enqueue_write_buffers(Queue, [{ok, _Atom}|Rest], Es, E) ->
  enqueue_write_buffers(Queue, Rest, Es, E).

%%------------------------------------------------------------------------------
%% @doc Enqueues a list of read buffers to a command queue.
%%------------------------------------------------------------------------------
enqueue_read_buffers(Queue, BufferDescriptors, Events) ->
  enqueue_read_buffers(Queue, BufferDescriptors, Events, []).

enqueue_read_buffers(_Queue, [], _Es, E) ->
  {ok, E};
enqueue_read_buffers(Queue, [{buffer_desc, Mem, _, DataSize}|Rest], Es, _E0) ->
  {ok, E1} = cl:enqueue_read_buffer(Queue, Mem, 0, DataSize, Es),
  enqueue_read_buffers(Queue, Rest, Es, E1).

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
       ),

      io:format("Build succeeded, creating kernel ~p... ~n", [Label]),
      {ok, Kernel} = cl:create_kernel(Program, Label),
      {ok, Program, Kernel};
    Error ->
      lists:foreach(
        fun (Device) ->
            {ok, BuildInfo} = cl:get_program_build_info(Program, Device),
            io:format(" * Build error ~w:~n  ~p~n", [Device, BuildInfo])
        end, Devices
       ),
      Error
  end.

%%------------------------------------------------------------------------------
%% @doc Releases the kernel, buffers, queues etc
%%------------------------------------------------------------------------------
release(Program, Kernel, Queues, I, O) ->
  [cl:release_mem_object(X) || {buffer_desc, X, _, _} <- I, not is_atom(X)],
  [cl:release_mem_object(X) || {buffer_desc, X, _, _} <- O],
  [cl:release_queue(X) || X <- Queues],
  cl:release_kernel(Kernel),
  cl:release_program(Program).

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

