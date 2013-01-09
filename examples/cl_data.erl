-module(cl_data).

-export([create_buffer/4, buffer_mem/1]).

%%------------------------------------------------------------------------------
%% Generic type descriptors
%%------------------------------------------------------------------------------
create_buffer(Context, Data, TypeSize, Opt) ->
  DataSize = byte_size(Data),
  {ok, Mem} = cl:create_buffer(Context, Opt, DataSize),
  {buffer_desc, Mem, Data, DataSize}.

buffer_mem({buffer_desc, Mem, _, _}) ->
  Mem;
buffer_mem(BufferDescriptors) when is_list(BufferDescriptors) ->
  [Mem || {buffer_desc, Mem, _, _} <- BufferDescriptors].  

  


