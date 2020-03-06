function  rear_overlap = indexing_function_rear(chunk_begin, locality, num_inputs)
% indexing_function_rear- returns rear local overlap indices for
% systems with periodic boundary conditions.
% 
% Inputs:
%   chunk_begin - index for beginning of data chunk
%
%   locality - local overlap on forward side of data chunk
%
%   num_inputs - total size of data
%
% Outputs: rear_overlap - indices of rear overlap region
if locality == 0
    rear_overlap = [];
elseif chunk_begin - locality > 0
    rear_overlap = chunk_begin-locality:chunk_begin-1;
elseif chunk_begin == locality
    rear_overlap = [num_inputs,1:(chunk_begin-1)];
elseif chunk_begin-locality <= 0 && chunk_begin > 1
    i1 = mod(chunk_begin - locality, num_inputs);
    rear_overlap = horzcat(i1:num_inputs, 1:chunk_begin-1);
elseif chunk_begin-locality <= 0 && chunk_begin == 1
    if locality == 1
        rear_overlap = num_inputs;
    else
        i1 = mod(chunk_begin - locality, num_inputs);
        rear_overlap = i1:num_inputs;
    end
else rear_overlap = -NaN; %throw error, hopefully
end
    