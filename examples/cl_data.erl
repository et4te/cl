-module(cl_data).

-export([create_buffer/4, kernel_args/1]).

%%------------------------------------------------------------------------------
%% Generic type descriptors
%%------------------------------------------------------------------------------
create_buffer(Context, Data, TypeSize, Opt) ->
  DataSize = byte_size(Data),
  {ok, Mem} = cl:create_buffer(Context, Opt, DataSize),
  {buffer_desc, Mem, Data, DataSize}.


kernel_args(BufferDescriptors) ->
  kernel_args(BufferDescriptors, []).

kernel_args([], Acc) ->
  lists:reverse(Acc);
kernel_args([{ok, Atom}|Rest], Acc) ->
  kernel_args(Rest, [Atom|Acc]);
kernel_args([{buffer_desc, Mem, _, _}|Rest], Acc) ->
  kernel_args(Rest, [Mem|Acc]).

  


