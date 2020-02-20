function  rear_overlap = indexing_function_rear(chunk_begin, locality, num_inputs)

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
else rear_overlap = -NaN; %throw error, hopefully :P
end
    