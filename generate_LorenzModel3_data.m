function generate_LorenzModel3_data(ModelParams,Nskip,predict_length)
% generate_LorenzModel3_data - generates training and testing time series
% data from Lorenz Model 3.
% Inputs:
%   ModelParams - a struct containing all of the model parameters necessary
%                 to solve the KS equation numerically. See the script
%                 CHyPP_test_LorenzModel3.m for an example of the necessary
%                 parameters when using the RK4 integrator.
%
%   Nskip - number of spatial grid points to skip when outputing training
%           and testing time series. For example, Nskip=2 will cause the
%           function to only output every second spatial gridpoint at each
%           time.
%
%   predict_length - length of the predictions to made from this data set.
%                    This parameter is used to specify the limit of start 
%                    indices that are saved in the pred_start_indices.mat
%                    file.
% Outputs:
%   M3_train_input_sequence.mat - contains the training data set, the
%       random gaussian noise vectors, and the overall mean and variance of
%       the training data.
%
%   M3_test_input_sequence.mat - contains the test data set and the overall
%       mean and variance of the training data.
%
%   M3_pred_start_indices.mat - contains the indices where synchronization
%       and prediction will begin during the testing phase.

% Set the lengths of the training and testing time series and determine a
% random initial condition.
seed = 5;
rng(seed,'twister');
Z = randn(ModelParams.N, 1);
sync_length = 100;
ModelParams.nstep = 1000;

data = ModelParams.prediction(Z, ModelParams);
Z = data(:,end);
ModelParams.nstep = 130000;

% Obtain the training and testing data
data = ModelParams.prediction(Z, ModelParams);

test_input_sequence = transpose(data(:, 100001:end));
train_input_sequence = transpose(data(:,1:100000));

% Coursen the data depending on Nskip
test_input_sequence = test_input_sequence(Nskip:Nskip:end,:);
train_input_sequence = train_input_sequence(Nskip:Nskip:end,:);

% Normalize the training and testing data
datamean = mean(train_input_sequence(:));

datavar = std(train_input_sequence(:));

train_input_sequence = train_input_sequence - datamean;

train_input_sequence = train_input_sequence./datavar;

test_input_sequence = test_input_sequence - datamean;

test_input_sequence= test_input_sequence./datavar;

% Determine random nosie for use during training
noise = randn(size(train_input_sequence,1),size(train_input_sequence,2));

% Determine random points for starting predictions
rng(seed,'twister')
start_iter = randi([1,size(test_input_sequence,1)-predict_length - sync_length - 1],1,1000);

% Save the data
mkdir('LorenzModel3_Data')
addpath LorenzModel3_Data
save('LorenzModel3_Data/M3_train_input_sequence.mat', 'train_input_sequence', 'noise','datamean', 'datavar', '-v7.3')
save('LorenzModel3_Data/M3_test_input_sequence.mat', 'test_input_sequence', 'datamean', 'datavar', '-v7.3')
save('LorenzModel3_Data/M3_pred_start_indices.mat','start_iter','-v7.3')