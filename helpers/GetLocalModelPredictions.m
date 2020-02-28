function model_states = GetLocalModelPredictions(init_cond_indices,...
    filehandle,datafield,noisefield,datamean,datavar,resnoise,ModelParams,...
    pool_size,chunk_size,locality,chunk_begin,chunk_end,rear_overlap,...
    forward_overlap)

train_input = datavar.*(filehandle.(datafield)(init_cond_indices,:)'+...
                resnoise*filehandle.(noisefield)(init_cond_indices,:)')+ datamean;

train_size = numel(init_cond_indices);
model_forecast = zeros(size(train_input));

for j = 1:train_size
    model_forecast(:,j) = ModelParams.predict(train_input(:,j), ModelParams);
end

model_states = Composite(pool_size);

for j = 1:pool_size
    

    v = zeros(chunk_size+2*locality, train_size);
    rear_overlap_temp = rear_overlap{j};
    forward_overlap_temp = forward_overlap{j};
    chunk_begin_temp = chunk_begin{j};
    chunk_end_temp = chunk_end{j};

    if locality > 0

        v(1:locality,:) = model_forecast(rear_overlap_temp(1,:), 1:end);

        v(locality+chunk_size+1:2*locality+chunk_size,:) = model_forecast(forward_overlap_temp(end,:), 1:end);

    end

    v(locality+1:locality+chunk_size,:) = model_forecast(chunk_begin_temp(1):chunk_end_temp(end), 1:end);

    model_states{j} = v;

end