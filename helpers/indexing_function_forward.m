function forward_overlap = indexing_function_forward(chunk_end, locality, num_inputs)
% indexing_function_forward - returns forward local overlap indices for
% systems with periodic boundary conditions.
% 
% Inputs:
%   chunk_end - index for end of data chunk
%
%   locality - local overlap on forward side of data chunk
%
%   num_inputs - total size of data
%
% Outputs: forward_overlap - indices of forward overlap region
if locality == 0
    forward_overlap = [];
elseif chunk_end + locality <= num_inputs
    forward_overlap = chunk_end+1:chunk_end+locality;
elseif chunk_end+locality>num_inputs && chunk_end == num_inputs
    forward_overlap = 1:mod(chunk_end + locality, num_inputs);
elseif chunk_end+locality>num_inputs && chunk_end < num_inputs
    forward_overlap = horzcat(chunk_end+1:num_inputs, 1:mod(chunk_end+locality, num_inputs));
else forward_overlap = -NaN;
end