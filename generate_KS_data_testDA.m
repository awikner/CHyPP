function generate_KS_data_testDA(ModelParams,predict_length)
% generate_KS_data - generates training and testing time series data from
% the Kuramoto-Sivashinsky equation.
% Inputs:
%   ModelParams - a struct containing all of the model parameters necessary
%                 to solve the KS equation numerically. See the script
%                 CHyPP_test_KS.m for an example of the necessary
%                 parameters when using the ETDRK4 integrator.
%
%   predict_length - length of the predictions to made from this data set.
%                    This parameter is used to specify the limit of start 
%                    indices that are saved in the pred_start_indices.mat
%                    file.
% Outputs:
%   KS_train_input_sequence.mat - contains the training data set, the
%       random gaussian noise vectors, and the overall mean and variance of
%       the training data.
%
%   KS_test_input_sequence.mat - contains the test data set and the overall
%       mean and variance of the training data.
%
%   KS_pred_start_indices.mat - contains the indices where synchronization
%       and prediction will begin during the testing phase.

% Set the lengths of the training and testing time series and determine a
% random initial condition.
dataseed = 5;
rng(dataseed);
nsteps = 150000;
nstepstest = 30000;
sync_length = 100;
x = 0.6*(-1+2*rand(ModelParams.N,1));

% Begin solving the equation, throw away the transient.
transient = 1000;

for i = 1:transient
    x = ModelParams.predict(x,ModelParams);
end

% Obtain the training data set
data = x;
train_input_sequence = zeros(nsteps,ModelParams.N);
train_input_sequence(1,:) = data';
for i = 1:nsteps-1
    data = ModelParams.predict(data,ModelParams);
    train_input_sequence(i+1,:) = data';
end

% Obtain the testing data set
data = ModelParams.predict(data,ModelParams);
test_input_sequence = zeros(nstepstest,ModelParams.N);
test_input_sequence(1,:) = data';
for i = 1:nstepstest-1
    data = ModelParams.predict(data,ModelParams);
    test_input_sequence(i+1,:) = data';
end

% Normalize both sets by the mean and variance of the training data
datamean = mean(train_input_sequence(:));
da_datamean = datamean;
datavar = std(train_input_sequence(:));
da_datavar = datavar;

train_input_sequence = train_input_sequence - datamean;

train_input_sequence = train_input_sequence./datavar;

test_input_sequence = test_input_sequence - datamean;

test_input_sequence= test_input_sequence./datavar;

da_test_input_sequence = test_input_sequence;
da_train_input_sequence = train_input_sequence;

% Generate random starting points for prediction
iterseed = 10;
rng(iterseed);

start_iter = randi([1,size(test_input_sequence,1)-predict_length - sync_length - 1],1,1000);

% Generate noise vectors to be used during training
noise_seed = 20;
rng(noise_seed);
noise = randn(size(train_input_sequence,1),size(train_input_sequence,2));

% Save all data
mkdir('KS_Data')
addpath KS_Data
save('KS_Data/KS_train_input_sequence.mat', 'train_input_sequence', 'da_train_input_sequence','noise','datamean','da_datamean', 'datavar','da_datavar', '-v7.3')
save('KS_Data/KS_test_input_sequence.mat', 'test_input_sequence', 'da_test_input_sequence','datamean', 'da_datamean','datavar','da_datavar', '-v7.3')
save('KS_Data/KS_pred_start_indices.mat','start_iter','-v7.3')