function model_states = GetModelPredictionsSerial(init_cond_indices,...
    filehandle,datafield,noisefield,datamean,datavar,resnoise,ModelParams)
% GetLocalModelPredictions - function for obtaining model predictions given
% a set of training data.
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
% Outputs:
%   model_states - composite variable containing the model predictions

% Load in training data with scaled noise added on
train_input = datavar.*(filehandle.(datafield)(init_cond_indices,:)'+...
                resnoise*filehandle.(noisefield)(init_cond_indices,:)')+ datamean;

% Obtain model 1-step forecasts
train_size = numel(init_cond_indices);
model_states = zeros(size(train_input));

for j = 1:train_size
    model_states(:,j) = ModelParams.predict(train_input(:,j), ModelParams);
end