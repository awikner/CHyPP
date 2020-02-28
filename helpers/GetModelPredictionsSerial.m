function model_states = GetModelPredictionsSerial(init_cond_indices,...
    filehandle,datafield,noisefield,datamean,datavar,resnoise,ModelParams)

train_input = datavar.*(filehandle.(datafield)(init_cond_indices,:)'+...
                resnoise*filehandle.(noisefield)(init_cond_indices,:)')+ datamean;

train_size = numel(init_cond_indices);
model_states = zeros(size(train_input));

for j = 1:train_size
    model_states(:,j) = ModelParams.predict(train_input(:,j), ModelParams);
end