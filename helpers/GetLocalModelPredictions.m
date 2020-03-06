function model_states = GetLocalModelPredictions(init_cond_indices,...
    filehandle,datafield,noisefield,datamean,datavar,resnoise,ModelParams,...
    pool_size,chunk_size,locality,chunk_begin,chunk_end,rear_overlap,...
    forward_overlap)
% GetLocalModelPredictions - function for obtaining model predictions given
% a set of training data and distributing the local regions of these
% predictions to the corresponding workers.
%
% Inputs:
%   init_cond_indices - indices specifying time points in time series for
%                       initial conditions
%
%   filehandle - full file path to training data file
%
%   datafield - field in training data file containing the training data
%
%   noisefield - field in training data file containing the noise vectors
%
%   datamean - global training data mean
%
%   datavar - global training data variance
%
%   resnoise - noise scaling
%
%   ModelParams - model parameter struct for evaluating model
%
%   pool_size - total number of workers being used
%
%   chunk_size - size of local spatial region predicted by each worker
%
%   locality - local overlap on either side of each worker's region
%
%   chunk_begin - composite variable containing the starting data index for
%                 each local region being predicted by each reservoir for
%                 each worker. The composite indice ({}) specifies which
%                 worker the indices correspond to, while the column index
%                 of the contained vector specfies the corresponding
%                 reservoir.
%
%   chunk_end - composite variable containing the ending data index for
%               each local region being predicted by each reservoir for
%               each worker. The composite indice ({}) specifies which
%               worker the indices correspond to, while the column index of
%               the contained vector specfies the corresponding reservoir.
%
%   rear_overlap - composite variable containing the rear overlap indices 
%                  for each local region being predicted by each reservoir
%                  for each worker. The composite indice ({}) specifies
%                  which worker the indices correspond to, while the row
%                  index of the contained matrix specfies the corresponding
%                  reservoir.
%
%   forward_overlap - composite variable containing the forward overlap 
%                     indices for each local region being predicted by each
%                     reservoir for each worker. The composite indice ({})
%                     specifies which worker the indices correspond to,
%                     while the row index of the contained matrix specfies
%                     the corresponding reservoir.
% Outputs:
%   model_states - composite variable containing the local model
%                  predictions for each worker (including those in the
%                  overlap regions)

% Load training data and additional noise into initial conditions matrix
train_input = datavar.*(filehandle.(datafield)(init_cond_indices,:)'+...
                resnoise*filehandle.(noisefield)(init_cond_indices,:)')+ datamean;

% Obtain model 1-step global forecasts
train_size = numel(init_cond_indices);
model_forecast = zeros(size(train_input));

for j = 1:train_size
    model_forecast(:,j) = ModelParams.predict(train_input(:,j), ModelParams);
end

% Distribute local model predictions to each worker
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