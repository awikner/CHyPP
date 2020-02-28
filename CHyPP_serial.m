function [avg_pred_length,filename,filename_rms] = CHyPP_serial(varargin)
% CHyPP (Combined Hybrid Parallel Prediction) - executes the CHyPP
% algorithm (training and prediction) for a 1-Dimensional system using only
% one processor core. If you have access to the MATLAB parallel computing
% toolbox, running the parallel version of this code (CHyPP) will be
% faster. This function takes inputs in the form of string-input pairs. If
% an input value is not specified, then CHyPP uses the default value
% specified below. Ex. CHyPP('NumRes',4,'Noise',1e-3)
% NOTE ON RANDOM SEEDS: Due to the nature of default random number
% generation in MATLAB, each worker uses its own random number
%
% Inputs:
%   NumRes - total number of reservoirs to be used in the prediction (P).
%            Default: 1
%
%   TrainLength - number of system measurements in training time series
%                 (t_0/(\Delta t)). Default: Length of data sequence given
%                 by TrainFile.
%
%   ReservoirSize - number of nodes in each reservoir (D_r). Default: 1000
%
%   AvgDegree - average in-degree of each reservoir adjacency matrix (<d>).
%               Default: 3
%
%   LocalOverlap - number of grid points in each local overlap region (l).
%                  Default: 0
%
%   InputWeight - scaling of nonzero elements in input matrices (\sigma).
%                 Default: 1
%
%   SpectralRadius - spectral radius (maximum absolute eigenvalue) of each
%                    reservoir adjacency matrix (\rho). Default: 0.6
%
%   Leakage - mixing parameter between advanced reservoir state and current
%             reservoir state using when calculating subsequent states (see
%             A Practical Guide to Applying Echo States Networks,
%             Lukosevicius 2012). This parameter is typically set to
%             between 0 and 1, where 0 fully weights the previous reservoir
%             state (no dynamics), and 1 fully weights the activated
%             reservoir state (as described in the CHyPP paper). The effect
%             of decreasing the leakage is to "slow down" the reservoir
%             dynamics. Default: 1
%
%   Noise - added training noise standard deviation (s). Default: 0
%
%   RidgeReg - L^2 regularization parameter used during Ridge Regression to
%              determine each output matrix W_{out}. Should be kept small.
%              Default: 1e-4
%              NOTE: CHyPP allows for separate regularization parameters
%              for the knowledge-based model variables and reservoir node
%              states. While the current codes set both to the same value,
%              this could be easily modified without changing any of the
%              computation.
%
%   Predictions - number of predictions to be made after training. 
%                 Default: 10
%
%   PredictLength - length of each prediction in t/(\Delta t). 
%                   Default: 1000
%
%   TrainSteps - number of steps over which the outer product matrices used
%                to compute each W_{out} are calculated. This number must 
%                divide TrainLength. Default: 50
%
%   ResOnly - boolean value indicating whether to perform a CHyPP or
%             "reservoirs-only" prediction (Pathak et. al. 2018). To use 
%             CHyPP, set to true; to use reservoirs-only, set to false.
%             Default: true
%             NOTE: If set to true, you MUST input a ModelParams struct as
%             well.
%
%   RunIter - seed for random generation of reservoir adjacency matrices.
%             Default: 1
%
%   OutputData - boolean indicating whether data from the predictions
%                should be output in the form of two files whose names are 
%                given by filename and filename_rms. See Outputs for more 
%                information. Default: true
%
%   TrainFile - name of full path to .mat file containing the variables
%               train_input_sequence, noise, datamean, and datavar. These
%               variables correspond to a (training time series length) by
%               (system size) matrix containing the training data to be
%               used with overall mean 0 and standard deviation 1, a matrix
%               of equal size containing independent gaussian random
%               variables of standard deviation 1, the mean of the original
%               data, and the standard deviation of the original data.
%               Default: 'KS_data/KS_train_input_sequence.mat' 
%               NOTE: To ensure proper loading during parallel execution,
%               make sure to save all .mat files using the '-v7.3' flag.
%
%   TestFile - name of full path to .mat file containing the variables
%              test_input_sequence, datamean, and datavar. These variables
%              correspond to a (test time series length) by (system size)
%              matrix containing the data that predictions are tested
%              against with mean 0 and standard deviation 1, the mean of
%              the original training data, and the standard deviation of
%              the original training data.
%              Default: 'KS_Data/KS_test_input_sequence.mat'
%              NOTE: To ensure proper loading during parallel execution,
%              make sure to save all .mat files using the '-v7.3' flag.
%
%   StartFile - name of full path to .mat file containing the variable
%               start_iter. This variable corresponds to the indices in the
%               test data set that synchonization and prediction begin at.
%               Default: 'KS_Data/KS_pred_start_indices.mat'
%               NOTES: To ensure proper loading during parallel execution,
%               make sure to save all .mat files using the '-v7.3' flag.
%               start_iter should be generated such that there are an equal
%               or greater number of elements than Prediction and such that
%               no element exceeds the length of the test data set - the
%               synchronization time - the prediction time.
%
%   ModelParams - struct whose fields contain the parameters necessary to
%                 execute the knowledge-based predictor component of
%                 CHyPP. The following fields are required by CHyPP:
%
%                 ModelParams.predict - contains a function handle to the
%                 imperfect knowledge-based predictor. This function must
%                 have two inputs, the initial state of the system and the
%                 ModelParams struct, and must produce one output, the
%                 prediction of the full system state after time \Delta t.
%
%                 ModelParams.prediction - contains a function handle to
%                 the iterated knowledge-based predictor. This function
%                 must have two inputs, the initial state of the system
%                 and the ModelParams struct, and must produce one output,
%                 a matrix whose j+1^th column contains the knowledge-based
%                 prediction of the system state at time j\Delta t after
%                 the initial condition (first column is just the initial
%                 condition). In addition, this output matrix must have
%                 PredictLength+1 columns for the comparison to work
%                 correctly.
%                 Default: 0 (no struct)
%
%   TestModelParams - a ModelParams type struct that, if given, is used
%                     in place of the CHyPP ModelParams struct when
%                     comparing CHyPP predictions to the knowledge-based
%                     model.
%                     Default: 0 (no struct)
%
%   OutputLocation - specifies the location where the output files should
%                    be saved (if applicable). Default: KS_Data
%
%
%   ErrorCutoff - value specifying the cutoff in normalized RMS error below
%                 which the prediction is considered valid. Used in
%                 determining the valid time of prediction. RMSE saturates
%                 at sqrt(2). Default: 0.2
% Outputs:
%   avg_pred_length - (average valid time of prediction)/(\Delta t) over
%                     all of the predictions
%
%   filename - full path to output file containing full predictions and
%              CHyPP parameters after training.
%
%   filename_rms - full path to output file containing RMS prediction 
%                  error, valid time for each prediction, and similar
%                  output using just the knowledge-based model (when
%                  specified).

%% Set default parameter values
num_res = 1;
train_length = 0;
reservoir_size = 1000;
avg_degree = 3;
locality = 0;
sigma_res = 1;
radius = 0.6;
leakage = 1;
beta_res = 1e-4;
beta_model = 1e-4;
resnoise = 0;
runiter = 1;
ifsavepred = true;
ifsaverms = true;
trainfilename = 'KS_Data/KS_train_input_sequence.mat';
testfilename = 'KS_Data/KS_test_input_sequence.mat';
startfilename = 'KS_Data/KS_pred_start_indices.mat';
outputlocation = 'KS_Data';
typeflag = 'reservoir';
ModelParams = 0;
TestModelParams = 0;
predictions = 10;
predict_length = 1000;
train_steps = 50;
error_cutoff = 0.2;


%% Parse inputs, assign to parameters, and check parameter compatibility
for arg = 1:numel(varargin)/2
    switch varargin{2*(arg-1)+1}
        case 'NumRes'
            num_res = varargin{2*arg};
        case 'TrainLength'
            train_length = varargin{2*arg};
        case 'ReservoirSize'
            reservoir_size = varargin{2*arg};
        case 'AvgDegree'
            avg_degree = varargin{2*arg};
        case 'LocalOverlap'
            locality = varargin{2*arg};
        case 'InputWeight'
            sigma_res = varargin{2*arg};
        case 'SpectralRadius'
            radius = varargin{2*arg};
        case 'Leakage'
            leakage = varargin{2*arg};
        case 'Noise'
            resnoise = varargin{2*arg};
        case 'RidgeReg'
            beta_res = varargin{2*arg};
            beta_model = varargin{2*arg};
        case 'Predictions'
            predictions = varargin{2*arg};
        case 'PredictLength'
            predict_length = varargin{2*arg};
        case 'TrainSteps'
            train_steps = varargin{2*arg};
        case 'ResOnly'
            if varargin{2*arg}
                typeflag = 'reservoir';
            elseif ~varargin{2*arg}
                typeflag = 'hybrid';
            end
        case 'RunIter'
            runiter = varargin{2*arg};
        case 'OutputData'
            ifsavepred = varargin{2*arg};
            ifsaverms = varargin{2*arg};
        case 'TrainFile'
            trainfilename = varargin{2*arg};
        case 'TestFile'
            testfilename = varargin{2*arg};
        case 'StartFile'
            startfilename = varargin{2*arg};
        case 'ModelParams'
            ModelParams = varargin{2*arg};
        case 'TestModelParams'
            TestModelParams = varargin{2*arg};
        case 'OutputLocation'
            outputlocation = varargin{2*arg};
        case 'ErrorCutoff'
            error_cutoff = varargin{2*arg};
        otherwise
            error(['Input variable ',num2str(arg),' not recognized.'])
    end
end

assert(~(strcmp(typeflag,'hybrid') & ~isstruct(ModelParams)),...
    'CHyPP has been told to run in a hybrid configuration, but no ModelParams struct has been given as input.')

%% Send parameters to each worker
%% Create handles for data files and load parameters from them
m = matfile(trainfilename);
tm = matfile(testfilename);
startfile = load(startfilename);
start_iter = startfile.start_iter;

resparams.sigma_data = sigma_res;

[len, num_inputs] = size(m, 'train_input_sequence');

[test_len,~] = size(tm,'test_input_sequence');

datamean = m.datamean;

datavar = m.datavar;

% Determine the the size of the data for each reservoir
res_chunk_size = num_inputs/num_res;

% Determine the beginning, end, and overlaps for each reservoir
chunk_begin = zeros(num_res,1);
chunk_end = zeros(num_res,1);
rear_overlap = zeros(num_res,locality);
forward_overlap = zeros(num_res,locality);

for res = 1:num_res

    chunk_begin(res) = res_chunk_size*(res-1)+1;
    chunk_end(res) = res_chunk_size*res;
    rear_overlap(res,:) = indexing_function_rear(chunk_begin(res), locality, num_inputs);  %spatial overlap on the one side
    forward_overlap(res,:) = indexing_function_forward(chunk_end(res), locality, num_inputs);  %spatial overlap on the other side
end

overlap_size = size(rear_overlap,2) + size(forward_overlap,2);

% Set resparams struct parameters
resparams.sparsity = avg_degree/reservoir_size;

resparams.degree = avg_degree;

resparams.N = reservoir_size;

resparams.discard_length = 1000;

resparams.predict_length = predict_length;

resparams.sync_length = 100;

if train_length == 0    %number of time steps used for training
    resparams.train_length = len - resparams.discard_length;
else
    resparams.train_length = train_length;  
end

assert(mod(resparams.train_length/train_steps,1)==0,...
    'train_steps must divide train_length. Change train_steps to a whole number that divides train_length.')

resparams.predictions = predictions; % define number of predictions (a prediction and synchronization is of length predict_length + sync_length)

resparams.radius = radius;

resparams.beta_model = beta_model; % ridge regression regularization parameter for the model 

resparams.beta_reservoir = beta_res; % ridge regression regularization parameter for the reservoir states

% Load transient data & noise
u = GetDataChunk(1:resparams.discard_length,m,...
    'train_input_sequence',num_inputs,0, ...
    1,num_inputs,[], []);
noise = GetDataChunk(1:resparams.discard_length,m,...
    'noise',num_inputs,0,1,num_inputs,[], []);
% Load test data used during synchronization
testu = GetDataChunk(1:test_len,tm,'test_input_sequence',...
    num_inputs,0, 1,num_inputs,[], []);

%% Generate reservoirs
A = cell(num_res,1);
for res = 1:num_res
    A{res} = generate_reservoir(resparams.N, resparams.radius, resparams.degree, 1, runiter, num_res, 1, res);
end

%% Generate input matrices
input_size = (res_chunk_size + overlap_size);

q = floor(resparams.N/(input_size));

win = cell(num_res,1);
for res = 1:num_res
    win{res} = zeros(resparams.N, input_size);

    for i=1:input_size
        rng(i,'twister')
        ip = (-1 + 2*rand(q,1));
        win{res}((i-1)*q+1:i*q,i) = resparams.sigma_data*ip;
    end

    occupied_nodes = floor(resparams.N/(input_size))*input_size;
    leftover_nodes = resparams.N - occupied_nodes;

    for i=1:leftover_nodes
        rng(i+input_size,'twister')
        ip = (-1 + 2*rand);
        win{res}(occupied_nodes+i,randi([1,input_size])) = resparams.sigma_data*ip;
    end
end

% Set the length of each training period

train_size  = resparams.train_length/train_steps;

states = cell(num_res,1);
x = cell(num_res,1);
for res = 1:num_res
    states{res} = zeros(resparams.N, train_size);
    x{res} = zeros(resparams.N,1);
end

% Create variables to store outer product matrices

data_trstates = cell(num_res,1);
states_trstates = cell(num_res,1);
if strcmp(typeflag, 'hybrid')
    for i=1:num_res
        data_trstates{i} = zeros(res_chunk_size,resparams.N+res_chunk_size);
        states_trstates{i} = zeros(resparams.N+res_chunk_size);
    end
elseif strcmp(typeflag,'reservoir')
    for i=1:num_res
        data_trstates{i} = zeros(res_chunk_size,resparams.N);
        states_trstates{i} = zeros(resparams.N);
    end
end



%% Feed in initial training data transient but do not train on resulting states
res_chunk = cell(num_res,1);
for res = 1:num_res
    res_chunk{res} = [rear_overlap(res,:),chunk_begin(res):chunk_end(res),forward_overlap(res,:)];
    for i = 1:resparams.discard_length-1
        x{res} = (1-leakage)*x{res}+leakage*tanh(A{res}*x{res} + win{res}*(u(res_chunk{res},i)+resnoise*noise(res_chunk{res},i)));
    end
    states{res}(:,1) = x{res};
end


for k = 1:train_steps

    %% Perform training over a number of train_steps, only importing training data for each section each time
    % For hybrid, first use knowledge-based predictor to get single step
    % predictions and send local regions to each worker.
    if strcmp(typeflag,'hybrid')
        if k == train_steps
            init_cond_indices = resparams.discard_length+(k-1)*train_size:resparams.discard_length+k*train_size;
        else
            init_cond_indices = resparams.discard_length+(k-1)*train_size:resparams.discard_length+k*train_size-1;
        end
        model_states = GetModelPredictionsSerial(init_cond_indices,...
            m,'train_input_sequence','noise',datamean,datavar,...
            resnoise,ModelParams);
    end

    % Load training data
    input_data_indices = resparams.discard_length+...
        (k-1)*train_size+1:resparams.discard_length+k*train_size;

    u = GetDataChunk(input_data_indices,m,'train_input_sequence',...
        num_inputs,0,1,num_inputs,[],[]);

    noise = GetDataChunk(input_data_indices,m,'noise',...
        num_inputs,0,1,num_inputs,[],[]);

    % For each reservoir, input the corresponding local region state
    % and record the resulting reservoir state after evolution
    augmented_states = cell(num_res,1);

    for res = 1:num_res
        for i = 1:train_size-1
            states{res}(:,i+1) = (1-leakage)*states{res}(:,i)+...
                leakage*tanh(A{res}*states{res}(:,i) + win{res}*...
                (u(res_chunk{res},i)+resnoise*noise(res_chunk{res},i)));
        end

        x{res} = states{res}(:, end);

        states{res}(2:2:resparams.N,:) = states{res}(2:2:resparams.N,:).^2;        
        % For hybrid, form augmented state using knowldedge-based
        % prediction
        if strcmp(typeflag,'hybrid')
            if k == train_steps
                augmented_states{res} = vertcat((model_states(res_chunk{res}(locality+1:...
                    locality+res_chunk_size),1:end-1)-datamean)./datavar, states{res});
                local_model = model_states(:,end);
            else
                augmented_states{res} = vertcat((model_states(res_chunk{res}(locality+1:...
                    locality+res_chunk_size),:)-datamean)./datavar, states{res});
            end
        elseif strcmp(typeflag, 'reservoir')
            augmented_states{res} = states{res};
        end
        % Use resulting states and training to form outer product
        % matrices.
        data_trstates{res} = data_trstates{res} + u(res_chunk{res}(locality+1:...
                    locality+res_chunk_size), :)*augmented_states{res}';
        states_trstates{res} = states_trstates{res} + augmented_states{res}*augmented_states{res}';

        states{res}(:,1) = (1-leakage)*x{res}+leakage*tanh(A{res}*x{res} + win{res}*(u(res_chunk{res},end)+resnoise*noise(res_chunk{res},end)));
    end
end
%% Train each reservoir using Ridge Regression
if strcmp(typeflag, 'hybrid')
    idenmat = sparse(diag([resparams.beta_model*ones(1,res_chunk_size),resparams.beta_reservoir*ones(1,resparams.N)]));
elseif strcmp(typeflag, 'reservoir')
    idenmat = sparse(diag(resparams.beta_reservoir*ones(1,resparams.N)));
end
wout = cell(num_res,1);
for res = 1:num_res
    wout{res} = data_trstates{res}/(states_trstates{res}+idenmat);
end
%% Use full system to make predictions
savepred = cell(resparams.predictions,1);
for j = 1:resparams.predictions
    savepred{j} =  zeros(num_inputs, resparams.predict_length);
end

for j = 1:resparams.predictions
    for res = 1:num_res
        x{res} = zeros(size(x{res}));
    end
    test_input = datavar.*tm.test_input_sequence(start_iter(j):start_iter(j)+resparams.sync_length-1,:)'+datamean;
    %% First, we synchronize the reservoir system to the test data using
    % a short synchronization sequence
    for i=1:resparams.sync_length
        feedback = testu(:,start_iter(j)+(i-1));

        if strcmp(typeflag, 'hybrid') && i==resparams.sync_length
            forecast_out = ModelParams.predict( test_input(:,i), ModelParams);
        end
        for res = 1:num_res
            x{res} = (1-leakage)*x{res}+leakage*tanh(A{res}*x{res} + win{res}*feedback(res_chunk{res}));
        end
    end
    % Obtain initial condition for prediction
    x_ = cell(num_res,1);
    augmented_x = cell(num_res,1);
    concatenated_out = zeros(num_inputs,1);
    for res = 1:num_res
        x_{res} = x{res};
        x_{res}(2:2:resparams.N) = x_{res}(2:2:resparams.N).^2;

        if strcmp(typeflag, 'hybrid')
            augmented_x{res} = vertcat((forecast_out(res_chunk{res}(locality+1:...
                    locality+res_chunk_size)) - datamean)./datavar, x_{res});
        elseif strcmp(typeflag, 'reservoir')
            augmented_x{res} = x_{res};
        end
        concatenated_out(res_chunk_size*(res-1)+1:res_chunk_size*res) = wout{res}*augmented_x{res};
    end

    %% Predict out to a time PredictLength*(\Delta t)
    for pred_idx = 1:resparams.predict_length
        % Get local region from global prediction
        feedback = concatenated_out;

        % If CHyPP, then obtain the local knowledge-based prediction
        if strcmp(typeflag, 'hybrid')
            forecast_out = ModelParams.predict( datavar.*concatenated_out + datamean, ModelParams);
        end

        % Obtain prediction from each reservoir and concatenate all of
        % regions for each worker together.
        for res = 1:num_res
            x{res} = (1-leakage)*x{res}+leakage*tanh(A{res}*x{res} + win{res}*feedback(res_chunk{res}));
            x_{res} = x{res};
            x_{res}(2:2:resparams.N) = x_{res}(2:2:resparams.N).^2;

            if strcmp(typeflag, 'hybrid')
                augmented_x{res} = vertcat((forecast_out(res_chunk{res}(locality+1:...
                    locality+res_chunk_size)) - datamean)./datavar, x_{res});
            elseif strcmp(typeflag, 'reservoir')
                augmented_x{res} = x_{res};
            end
    %        out = local_model(locality+1:locality+res_chunk_size,:);
            concatenated_out(res_chunk_size*(res-1)+1:res_chunk_size*res) = wout{res}*augmented_x{res};
        end 

        % Record prediction
        savepred{j}(:,pred_idx) = concatenated_out;

    end
end


%% Get CHyPP parameters from each worker

Woutmat = {};
Amat = {};
winmat = {};
iter = 1;
for res = 1:num_res
    Woutmat{iter} = wout{res};
    Amat{iter} = A{res};
    winmat{iter} = win{res};
    iter = iter + 1;
end

% Set output file names
beta_reservoir = resparams.beta_reservoir;
if strcmp(typeflag, 'hybrid')
    filename = [outputlocation,'/','hybrid', '-numres', num2str(num_res), ...
        'res', num2str(reservoir_size), ...
        'localoverlap',num2str(locality),'trainlen',num2str(train_length),'sigma',...
        strrep(num2str(resparams.sigma_data),'.',''),'radius',strrep(num2str(resparams.radius),'.',''),...
        'leakage',strrep(num2str(leakage),'.',''),'betares',strrep(num2str(beta_reservoir),'.',''),...
        'runiter',num2str(runiter),'_wnoise',strrep(num2str(resnoise),'.',''),'_data_serial.mat'];
elseif strcmp(typeflag, 'reservoir')
    filename = [outputlocation,'/','reservoir', '-numres', num2str(num_res), ...
        'res', num2str(reservoir_size), ...
        'localoverlap',num2str(locality),'trainlen',num2str(train_length),'sigma',...
        strrep(num2str(resparams.sigma_data),'.',''),'radius',strrep(num2str(resparams.radius),'.',''),...
        'leakage',strrep(num2str(leakage),'.',''),'betares',strrep(num2str(beta_reservoir),'.',''),...
        'runiter',num2str(runiter),'_wnoise',strrep(num2str(resnoise),'.',''),'_data_serial.mat'];
end
%% Output predictions and CHyPP parameters
if ifsavepred
    if isstruct(ModelParams)
        save(filename, 'savepred','resparams','start_iter', 'Woutmat','Amat','winmat','ModelParams');
    else
        save(filename, 'savepred','resparams','start_iter', 'Woutmat','Amat','winmat');
    end
end

%% Evaluate CHyPP (or parallel reservoir only) and knowledge-based model (if applicable)
% For each prediction, calculate RMS error and valid time using the
% specified error cutoff (saturation at sqrt(2))
if isstruct(ModelParams) && ~isstruct(TestModelParams)
    TestModelParams = ModelParams;
end
if isstruct(TestModelParams)
    TestModelParams.nstep = resparams.predict_length;
    model_pred_length = zeros(1,resparams.predictions);
    model_average_diff = zeros(1,TestModelParams.nstep);
    model_err  = cell(1,resparams.predictions);
end

prediction_average_diff = zeros(1,resparams.predict_length);
pred_length = zeros(1,resparams.predictions);
prediction_err = cell(1,resparams.predictions);
prediction_err_sum = 0;

for i=1:resparams.predictions
    if isstruct(TestModelParams)
        xinit = savepred{i}(:,1).*datavar+datamean;
        data1 = TestModelParams.prediction( xinit, TestModelParams);

        model_sequence = transpose(data1);
        model_sequence = model_sequence - datamean;
        model_sequence = model_sequence./datavar;
        model_err{i}    = sqrt(mean((model_sequence-...
            tm.test_input_sequence(start_iter(i)+resparams.sync_length+1:...
            start_iter(i)+resparams.sync_length+resparams.predict_length,:)).^2,2)');
        model_average_diff = model_average_diff + model_err{i}/(resparams.predictions);

        for j=1:length(model_err{i})
            if model_err{i}(j) > error_cutoff
                model_pred_length(i) = j-1;
                break
            end
        end
    end

    prediction_err{i} = sqrt(mean((savepred{i}'-...
        tm.test_input_sequence(start_iter(i)+...
        resparams.sync_length+1:start_iter(i)+resparams.sync_length...
        +resparams.predict_length,:)).^2,2)');
    for j=1:length(prediction_err{i})
        if prediction_err{i}(j) > error_cutoff
            pred_length(i) = j-1;
            break
        elseif j == length(prediction_err{i}) && prediction_err{i}(j) <= error_cutoff
            pred_length(i) = j;
        end
    end
    prediction_err_sum = prediction_err_sum + sum(prediction_err{i})/resparams.predictions;
    prediction_average_diff = prediction_average_diff + prediction_err{i};

end

prediction_average_diff = prediction_average_diff/resparams.predictions;
avg_pred_length = mean(pred_length);
std_pred_length = std(pred_length);
if isstruct(TestModelParams)
    model_avg_pred_length = mean(model_pred_length);
    model_std_pred_length = std(model_pred_length);
end
%% Set rms data output file name and safe it specified
if strcmp(typeflag, 'hybrid')
    filename_rms = [outputlocation,'/','hybrid', '-numres', num2str(num_res), ...
            'res', num2str(reservoir_size), ...
            'localoverlap',num2str(locality),'trainlen',num2str(train_length),'sigma',...
            strrep(num2str(resparams.sigma_data),'.',''),'radius',strrep(num2str(resparams.radius),'.',''),...
            'leakage',strrep(num2str(leakage),'.',''),'betares',strrep(num2str(beta_reservoir),'.',''),...
            'runiter',num2str(runiter),'_wnoise',strrep(num2str(resnoise),'.',''),'_rms_serial.mat'];
elseif strcmp(typeflag, 'reservoir')
    filename_rms = [outputlocation,'/','reservoir', '-numres', num2str(num_res), ...
            'res', num2str(reservoir_size), ...
            'localoverlap',num2str(locality),'trainlen',num2str(train_length),'sigma',...
            strrep(num2str(resparams.sigma_data),'.',''),'radius',strrep(num2str(resparams.radius),'.',''),...
            'leakage',strrep(num2str(leakage),'.',''),'betares',strrep(num2str(beta_reservoir),'.',''),...
            'runiter',num2str(runiter),'_wnoise',strrep(num2str(resnoise),'.',''),'_rms_serial.mat'];
end
if ifsaverms
    if isstruct(TestModelParams)
        save(filename_rms, 'prediction_err','prediction_average_diff','model_err','model_average_diff',...
            'avg_pred_length','std_pred_length','pred_length','model_avg_pred_length','model_std_pred_length','model_pred_length','TestModelParams')
    else
        save(filename_rms, 'prediction_err','prediction_average_diff',...
            'avg_pred_length','std_pred_length','pred_length')
    end
end