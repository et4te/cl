-module(cl_data).

-export([create_buffer/4, to_list/1, kernel_args/1]).

%%------------------------------------------------------------------------------
%% Generic type descriptors
%%------------------------------------------------------------------------------
create_buffer(Context, Data, TypeSize, Opt) ->
  DataSize = byte_size(Data),
  {ok, Mem} = cl:create_buffer(Context, Opt, DataSize),
  {buffer_desc, Mem, Data, DataSize}.

%%------------------------------------------------------------------------------
%% @doc Convert a binary containing native float elements to a list of floats.
%%------------------------------------------------------------------------------
to_list(Binary) when is_binary(Binary) ->
  to_list(Binary, []).

to_list(<<>>, Acc) ->
  lists:reverse(Acc);
to_list(<<X:32/native-float, Rest/binary >>, Acc) ->
  to_list(Rest, [X|Acc]).

%%------------------------------------------------------------------------------
%% @doc Convert a list of buffer_desc elements to kernel arguments. The 
%%      following function returns a list which contains only atoms such as ints
%%      and memory elements which can be used as arguments to an OpenCL kernel.
%%------------------------------------------------------------------------------
kernel_args(BufferDescriptors) ->
  kernel_args(BufferDescriptors, []).

kernel_args([], Acc) ->
  lists:reverse(Acc);
kernel_args([{ok, Atom}|Rest], Acc) ->
  kernel_args(Rest, [Atom|Acc]);
kernel_args([{buffer_desc, Mem, _, _}|Rest], Acc) ->
  kernel_args(Rest, [Mem|Acc]).

  


