function [avg_pred_length,filename,filename_rms] = CHyPP_DA(varargin)
% CHyPP (Combined Hybrid Parallel Prediction) - executes the CHyPP
% algorithm (training and prediction) for a 1-Dimensional system using
% MATLAB's parallel computing toolbox. If you don't have access to this
% toolbox, you can run the slower serialized version of this code instead
% (CHyPP_serial). This function takes inputs in the form of string-input
% pairs. If an input value is not specified, then CHyPP uses the default
% value specified below. Ex. CHyPP('NumRes',4,'Noise',1e-3)
%
% Inputs:
%   PoolSize - number of CPU cores to be used for computation. This number
%              divide NumRes (total number of reservoirs). Default: 1
%
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
%   ErrorCutoff - value specifying the cutoff in normalized RMS error below
%                 which the prediction is considered valid. Used in
%                 determining the valid time of prediction. RMSE saturates
%                 at sqrt(2). Default: 0.2
%
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
pool_size = 1;
num_res_in = 1;
train_lengthin = 0;
reservoir_sizein = 1000;
avg_degreein = 3;
localityin = 0;
sigma_resin = 1;
radiusin = 0.6;
leakage_in = 1;
betain_res = 1e-4;
betain_model = 1e-4;
resnoisein = 0;
runiterin = 1;
ifsavepred = false;
ifsaverms = false;
trainfilename_in = 'KS_Data/KS_train_input_sequence.mat';
testfilename_in = 'KS_Data/KS_test_input_sequence.mat';
startfilename_in = 'KS_Data/KS_pred_start_indices.mat';
outputlocation = 'KS_Data';
typeflag_in = 'reservoir';
ModelParams = 0;
TestModelParams = 0;
predictions_in = 10;
predict_length_in = 1000;
train_steps_in = 50;
error_cutoff = 0.2;
covariance_inflation = 1;


%% Parse inputs, assign to parameters, and check parameter compatibility
for arg = 1:numel(varargin)/2
    switch varargin{2*(arg-1)+1}
        case 'PoolSize'
            pool_size = varargin{2*arg};
        case 'NumRes'
            num_res_in = varargin{2*arg};
        case 'TrainLength'
            train_lengthin = varargin{2*arg};
        case 'ReservoirSize'
            reservoir_sizein = varargin{2*arg};
        case 'AvgDegree'
            avg_degreein = varargin{2*arg};
        case 'LocalOverlap'
            localityin = varargin{2*arg};
        case 'InputWeight'
            sigma_resin = varargin{2*arg};
        case 'SpectralRadius'
            radiusin = varargin{2*arg};
        case 'Leakage'
            leakage_in = varargin{2*arg};
        case 'Noise'
            resnoisein = varargin{2*arg};
        case 'RidgeReg'
            betain_res = varargin{2*arg};
            betain_model = varargin{2*arg};
        case 'Predictions'
            predictions_in = varargin{2*arg};
        case 'PredictLength'
            predict_length_in = varargin{2*arg};
        case 'TrainSteps'
            train_steps_in = varargin{2*arg};
        case 'ResOnly'
            if varargin{2*arg}
                typeflag_in = 'reservoir';
            elseif ~varargin{2*arg}
                typeflag_in = 'hybrid';
            end
        case 'RunIter'
            runiterin = varargin{2*arg};
        case 'OutputData'
            ifsavepred = varargin{2*arg};
        case 'OutputRMS'
            ifsaverms = varargin{2*arg};
        case 'TrainFile'
            trainfilename_in = varargin{2*arg};
        case 'TestFile'
            testfilename_in = varargin{2*arg};
        case 'StartFile'
            startfilename_in = varargin{2*arg};
        case 'ModelParams'
            ModelParams = varargin{2*arg};
        case 'TestModelParams'
            TestModelParams = varargin{2*arg};
        case 'OutputLocation'
            outputlocation = varargin{2*arg};
        case 'ErrorCutoff'
            error_cutoff = varargin{2*arg};
        case 'CovarianceInflation'
            covariance_inflation = varargin{2*arg};
        otherwise
            error(['Input variable ',num2str(arg),' not recognized.'])
    end
end

assert(~(strcmp(typeflag_in,'hybrid') & ~isstruct(ModelParams)),...
    'CHyPP has been told to run in a hybrid configuration, but no ModelParams struct has been given as input.')
assert(mod(num_res_in/pool_size,1)==0, 'Number of cores must divide number of reservoirs used.')

%% Send parameters to each worker
num_res = Composite(pool_size);
avg_degree = Composite(pool_size);
sigma_res = Composite(pool_size);
radius = Composite(pool_size);
leakage = Composite(pool_size);
beta = Composite(pool_size);
beta_model = Composite(pool_size);
resnoise = Composite(pool_size);
reservoir_size = Composite(pool_size);
runiter = Composite(pool_size);
train_length = Composite(pool_size);
locality = Composite(pool_size);
typeflag = Composite(pool_size);
trainfilename = Composite(pool_size);
testfilename  = Composite(pool_size);
startfilename = Composite(pool_size);
predictions = Composite(pool_size);
predict_length = Composite(pool_size);
train_steps = Composite(pool_size);

for i = 1:pool_size
    num_res{i} = num_res_in;
    avg_degree{i} = avg_degreein;
    sigma_res{i} = sigma_resin;
    radius{i} = radiusin;
    leakage{i} = leakage_in;
    beta{i} = betain_res;
    beta_model{i} = betain_model;
    resnoise{i} = resnoisein;
    reservoir_size{i} = reservoir_sizein;
    runiter{i} = runiterin;
    train_length{i} = train_lengthin;
    locality{i} = localityin;
    typeflag{i} = typeflag_in;
    trainfilename{i} = trainfilename_in;
    testfilename{i}  = testfilename_in;
    startfilename{i} = startfilename_in;
    predictions{i} = predictions_in;
    predict_length{i} = predict_length_in;
    train_steps{i} = train_steps_in;
end

spmd(pool_size)
    %% Create handles for data files and load parameters from them
    m = matfile(trainfilename);
    tm = matfile(testfilename);
    startfile = load(startfilename);
    start_iter = startfile.start_iter;

    resparams.sigma_data = sigma_res;

    [len, num_inputs] = size(m, 'train_input_sequence');
        
    [test_len,~] = size(tm,'test_input_sequence');

    datamean = m.datamean;
    da_datamean = m.da_datamean;
    datavar = m.datavar;
    da_datavar = m.da_datavar;

    num_workers = numlabs; % Determines number of workers (pool_size)
    
    % Determine the the size of the data for each worker and each reservoir
    core_chunk_size = num_inputs/num_workers; 
    res_chunk_size = num_inputs/num_res; 

    l = labindex; % labindex is a matlab function that returns the worker index
    
    % Determine the beginning, end, and overlaps for each reservoir
    res_per_core = num_res/num_workers;
    chunk_begin = zeros(res_per_core,1);
    chunk_end = zeros(res_per_core,1);
    rear_overlap = zeros(res_per_core,locality);
    forward_overlap = zeros(res_per_core,locality);
    
    for res = 1:res_per_core
        
        chunk_begin(res) = res_chunk_size*(res-1)+core_chunk_size*(l-1)+1;
        chunk_end(res) = core_chunk_size*(l-1)+res_chunk_size*res;
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

    resparams.beta_reservoir = beta; % ridge regression regularization parameter for the reservoir states
    
    % Load transient data & noise
    u = GetDataChunk(1:resparams.discard_length,m,...
        'train_input_sequence',core_chunk_size,locality, ...
        chunk_begin(1),chunk_end(end),rear_overlap(1,:), forward_overlap(end,:));
    dau = GetDataChunk(1:resparams.discard_length,m,'da_train_input_sequence',...
        core_chunk_size,locality,chunk_begin(1),chunk_end(end),...
        rear_overlap(1,:), forward_overlap(end,:));
    noise = GetDataChunk(1:resparams.discard_length,m,...
        'noise',core_chunk_size,locality,chunk_begin(1),chunk_end(end),...
        rear_overlap(1,:), forward_overlap(end,:));
    % Load test data used during synchronization
%     testu = GetDataChunk(1:test_len,tm,'test_input_sequence',...
%         core_chunk_size,locality,chunk_begin(1),chunk_end(end),...
%         rear_overlap(1,:), forward_overlap(end,:));
    
    %% Generate reservoirs
    A = cell(res_per_core,1);
    for res = 1:res_per_core
        A{res} = generate_reservoir(resparams.N, resparams.radius, resparams.degree, l, runiter, num_res, num_workers, res);
    end

    %% Generate input matrices
    input_size = 2*(res_chunk_size + overlap_size);

    q = floor(resparams.N/(input_size));
    
    win = cell(res_per_core,1);
    for res = 1:res_per_core
        win{res} = zeros(resparams.N, input_size);

        for i=1:input_size
            rng(i)
            ip = (-1 + 2*rand(q,1));
            win{res}((i-1)*q+1:i*q,i) = resparams.sigma_data*ip;
        end

        occupied_nodes = floor(resparams.N/(input_size))*input_size;
        leftover_nodes = resparams.N - occupied_nodes;

        for i=1:leftover_nodes
            rng(i+input_size)
            ip = (-1 + 2*rand);
            win{res}(occupied_nodes+i,randi([1,input_size])) = resparams.sigma_data*ip;
        end
    end
    
    % Set the length of each training period
    
    train_size  = resparams.train_length/train_steps;

    states = cell(res_per_core,1);
    x = cell(res_per_core,1);
    for res = 1:res_per_core
        states{res} = zeros(resparams.N, train_size);
        x{res} = zeros(resparams.N,1);
    end

end

% Create variables to store outer product matrices
rp = resparams{1};

train_file = m{1};

data_trstates = Composite(num_workers{1});
states_trstates = Composite(num_workers{1});
for i=1:num_workers{1}
    data_trstates{i} = cell(res_per_core{1},1);
    states_trstates{i} = cell(res_per_core{1},1);
end


spmd(pool_size)
    if strcmp(typeflag, 'hybrid')
        for i=1:res_per_core
            data_trstates{i} = zeros(2*res_chunk_size,resparams.N+res_chunk_size);
            states_trstates{i} = zeros(resparams.N+res_chunk_size);
        end
    elseif strcmp(typeflag,'reservoir')
        for i=1:res_per_core
            data_trstates{i} = zeros(2*res_chunk_size,resparams.N);
            states_trstates{i} = zeros(resparams.N);
        end
    end
    %% Feed in initial training data transient but do not train on resulting states
    res_chunk = cell(res_per_core,1);
    for res = 1:res_per_core
        res_chunk{res} = res_chunk_size*(res-1)+1:res_chunk_size*res+2*locality;
        for i = 1:resparams.discard_length-1
            x{res} = (1-leakage)*x{res}+leakage*tanh(A{res}*x{res} + win{res}*...
                ([u(res_chunk{res},i);dau(res_chunk{res},i)]+resnoise*[noise(res_chunk{res},i);noise(res_chunk{res},i)]));
        end
        states{res}(:,1) = x{res};
    end
end

for k = 1:train_steps{1}

    %% Perform training over a number of train_steps, only importing training data for each section each time
    % For hybrid, first use knowledge-based predictor to get single step
    % predictions and send local regions to each worker.
    if strcmp(typeflag{1},'hybrid')
        if k == train_steps{1}
            init_cond_indices = rp.discard_length+(k-1)*train_size{1}:rp.discard_length+k*train_size{1};
        else
            init_cond_indices = rp.discard_length+(k-1)*train_size{1}:rp.discard_length+k*train_size{1}-1;
        end
        model_states = GetLocalModelPredictions(init_cond_indices,...
            train_file,'da_train_input_sequence','noise',da_datamean{1},da_datavar{1},...
            resnoise{1},ModelParams,pool_size,core_chunk_size{1},...
            locality{1},chunk_begin,chunk_end,rear_overlap,...
            forward_overlap);
    end

    kc = Composite(num_workers{1});
    for i=1:num_workers{1}
        kc{i} = k;
    end
    k = kc;
    % Load training data
    spmd(pool_size)
        input_data_indices = resparams.discard_length+...
            (k-1)*train_size+1:resparams.discard_length+k*train_size;
        
        u = GetDataChunk(input_data_indices,m,'train_input_sequence',...
            core_chunk_size,locality,chunk_begin(1),...
            chunk_end(end),rear_overlap(1,:),forward_overlap(end,:));
        
        dau = GetDataChunk(input_data_indices,m,'da_train_input_sequence',...
            core_chunk_size,locality,chunk_begin(1),...
            chunk_end(end),rear_overlap(1,:),forward_overlap(end,:));
        
        noise = GetDataChunk(input_data_indices,m,'noise',...
            core_chunk_size,locality,chunk_begin(1),...
            chunk_end(end),rear_overlap(1,:),forward_overlap(end,:));
        
        % For each reservoir, input the corresponding local region state
        % and record the resulting reservoir state after evolution
        augmented_states = cell(res_per_core,1);
        for res = 1:res_per_core
            for i = 1:train_size-1
                states{res}(:,i+1) = (1-leakage)*states{res}(:,i)+...
                    leakage*tanh(A{res}*states{res}(:,i) + win{res}*...
                    ([u(res_chunk{res},i);dau(res_chunk{res},i)]+...
                    resnoise*[noise(res_chunk{res},i);noise(res_chunk{res},i)]));
            end

            x{res} = states{res}(:, end);

            states{res}(2:2:resparams.N,:) = states{res}(2:2:resparams.N,:).^2;        
            % For hybrid, form augmented state using knowldedge-based
            % prediction
            if strcmp(typeflag,'hybrid')
                if k == train_steps
                    augmented_states{res} = vertcat((model_states(locality+res_chunk_size*(res-1)+1:...
                        locality+res_chunk_size*res,1:end-1)-da_datamean)./da_datavar, states{res});
                    local_model = model_states(:,end);
                else
                    augmented_states{res} = vertcat((model_states(locality+res_chunk_size*(res-1)+1:...
                        locality+res_chunk_size*res,:)-da_datamean)./da_datavar, states{res});
                end
            elseif strcmp(typeflag, 'reservoir')
                augmented_states{res} = states{res};
            end
            % Use resulting states and training to form outer product
            % matrices.
            data_trstates{res} = data_trstates{res} + [u(locality+res_chunk_size*(res-1)+1:...
                        locality+res_chunk_size*res, :);dau(locality+res_chunk_size*(res-1)+1:...
                        locality+res_chunk_size*res, :)]*augmented_states{res}';
            states_trstates{res} = states_trstates{res} + augmented_states{res}*augmented_states{res}';

            states{res}(:,1) = (1-leakage)*x{res}+leakage*tanh(A{res}*x{res} + win{res}*([u(res_chunk{res},end);dau(res_chunk{res},end)]+resnoise*[noise(res_chunk{res},end);noise(res_chunk{res},end)]));
        end
    end
end
spmd(pool_size)
%% Train each reservoir using Ridge Regression
    if strcmp(typeflag, 'hybrid')
        idenmat = sparse(diag([resparams.beta_model*ones(1,res_chunk_size),resparams.beta_reservoir*ones(1,resparams.N)]));
    elseif strcmp(typeflag, 'reservoir')
        idenmat = sparse(diag(resparams.beta_reservoir*ones(1,resparams.N)));
    end
    wout = cell(res_per_core,1);
    for res = 1:res_per_core
        wout{res} = data_trstates{res}/(states_trstates{res}+idenmat);
    end
%% Use full system to make predictions
    if labindex == 1
        prediction = cell(resparams.predictions,1);
        prediction_da = cell(resparams.predictions,1);
        for j = 1:resparams.predictions
            prediction{j} =  zeros(num_inputs, resparams.predict_length);
            prediction_da{j} = zeros(num_inputs, resparams.predict_length);
        end
    end

    for j = 1:resparams.predictions
        for res = 1:res_per_core
            x{res} = zeros(size(x{res}));
        end
        if labindex == 1
            test_input = da_datavar.*tm.da_test_input_sequence(start_iter(j):start_iter(j)+resparams.sync_length-1,:)'+da_datamean;
        end
        %% First, we synchronize the reservoir system to the test data using
        % a short synchronization sequence
        for i=1:resparams.sync_length
            feedback_u = GetDataChunk(start_iter(j)+(i-1),tm,'test_input_sequence',...
                core_chunk_size,locality,chunk_begin(1),chunk_end(end),...
                rear_overlap(1,:), forward_overlap(end,:));
            feedback_dau = GetDataChunk(start_iter(j)+(i-1),tm,'da_test_input_sequence',...
                core_chunk_size,locality,chunk_begin(1),chunk_end(end),...
                rear_overlap(1,:), forward_overlap(end,:));

            if strcmp(typeflag, 'hybrid') && i==resparams.sync_length
                if labindex == 1
                    forecast_out = ModelParams.predict( test_input(:,i), ModelParams);
                    global_model_forecast = labBroadcast(1,forecast_out);
                else
                    global_model_forecast = labBroadcast(1);
                end
                % LabBarrier tells MATLAB to wait until all workers reach
                % this point
                labBarrier;

                local_model = zeros(input_size,1);

                if locality > 0

                    local_model(1:locality) = global_model_forecast(rear_overlap(1,:),1);

                    local_model(locality+core_chunk_size+1:2*locality+core_chunk_size) =...
                        global_model_forecast(forward_overlap(end,:),1);
                end

                local_model(locality+1:locality+core_chunk_size) = global_model_forecast(chunk_begin(1):chunk_end(end),1);
            end
            for res = 1:res_per_core
                x{res} = (1-leakage)*x{res}+leakage*tanh(A{res}*x{res} + win{res}*[feedback_u(res_chunk{res});feedback_dau(res_chunk{res})]);
            end
        end
        % Obtain initial condition for prediction
        x_ = cell(res_per_core,1);
        augmented_x = cell(res_per_core,1);
        out_u = zeros(core_chunk_size,1);
        out_dau = zeros(core_chunk_size,1);
        for res = 1:res_per_core
            x_{res} = x{res};
            x_{res}(2:2:resparams.N) = x_{res}(2:2:resparams.N).^2;

            if strcmp(typeflag, 'hybrid')
                augmented_x{res} = vertcat((local_model(locality+res_chunk_size*(res-1)+1:...
                    locality+res_chunk_size*res) - da_datamean)./da_datavar, x_{res});
            elseif strcmp(typeflag, 'reservoir')
                augmented_x{res} = x_{res};
            end
            temp_out = wout{res}*augmented_x{res};
            out_u(res_chunk_size*(res-1)+1:res_chunk_size*res) = temp_out(1:end/2);
            out_dau(res_chunk_size*(res-1)+1:res_chunk_size*res) = temp_out(end/2+1:end);
        end
        labBarrier;
        concatenated_out_u = gcat(out_u, 1);
        concatenated_out_dau = gcat(out_dau, 1);
        
        %% Predict out to a time PredictLength*(\Delta t)
        for pred_idx = 1:resparams.predict_length
            % Get local region from global prediction
            feedback_u = zeros(input_size,1);
            feedback_dau = zeros(input_size,1);
            if locality > 0
                feedback_u(1:locality) = concatenated_out_u(rear_overlap(1,:));

                feedback_u(locality+core_chunk_size+1:2*locality+core_chunk_size) = concatenated_out_u(forward_overlap(end,:));
                
                feedback_dau(1:locality) = concatenated_out_dau(rear_overlap(1,:));

                feedback_dau(locality+core_chunk_size+1:2*locality+core_chunk_size) = concatenated_out_dau(forward_overlap(end,:));
            end

            feedback_u(locality+1:locality+core_chunk_size) = concatenated_out_u(chunk_begin(1):chunk_end(end));
            feedback_dau(locality+1:locality+core_chunk_size) = concatenated_out_dau(chunk_begin(1):chunk_end(end));
            
            % If CHyPP, then obtain the local knowledge-based prediction
            if strcmp(typeflag, 'hybrid')
                if labindex == 1
                    forecast_out = ModelParams.predict( da_datavar.*concatenated_out_dau + da_datamean, ModelParams);
                    global_model_forecast = labBroadcast(1,forecast_out);

                else
                    global_model_forecast = labBroadcast(1);
                end
                labBarrier;

                local_model = zeros(input_size,1);

                if locality > 0
                    local_model(1:locality) = global_model_forecast(rear_overlap(1,:),1);

                    local_model(locality+core_chunk_size+1:2*locality+core_chunk_size) = global_model_forecast(forward_overlap(end,:),1);
                end

                local_model(locality+1:locality+core_chunk_size) = global_model_forecast(chunk_begin(1):chunk_end(end),1);
            end
            
            % Obtain prediction from each reservoir and concatenate all of
            % regions for each worker together.
            for res = 1:res_per_core
                x{res} = (1-leakage)*x{res}+leakage*tanh(A{res}*x{res} + win{res}*[feedback_u(res_chunk{res});feedback_dau(res_chunk{res})]);
                x_{res} = x{res};
                x_{res}(2:2:resparams.N) = x_{res}(2:2:resparams.N).^2;

                if strcmp(typeflag, 'hybrid')
                    augmented_x{res} = vertcat((local_model(locality+res_chunk_size*(res-1)+1:...
                        locality+res_chunk_size*res) - datamean)./datavar, x_{res});
                elseif strcmp(typeflag, 'reservoir')
                    augmented_x{res} = x_{res};
                end
        %        out = local_model(locality+1:locality+res_chunk_size,:);
                temp_out = wout{res}*augmented_x{res};
                out_u(res_chunk_size*(res-1)+1:res_chunk_size*res) = temp_out(1:end/2);
                out_dau(res_chunk_size*(res-1)+1:res_chunk_size*res) = temp_out(end/2+1:end);
            end
            labBarrier;
            
            % Send full state to all workers
            concatenated_out_u = gcat(out_u, 1);  
            concatenated_out_dau = gcat(out_dau, 1); 
            % Record prediction
            if labindex == 1
                prediction{j}(:,pred_idx) = concatenated_out_u;
                prediction_da{j}(:,pred_idx) = concatenated_out_dau;
            end

        end
    end
end

%% Get CHyPP parameters from each worker
reservoir_size = reservoir_size{1};

savepred = prediction{1};
savepred_da = prediction_da{1};
resparams = resparams{1};
Woutmat = {};
Amat = {};
winmat = {};
iter = 1;
for i=1:pool_size
    wout_temp = wout{i};
    A_temp = A{i};
    win_temp = win{i};
    for res = 1:res_per_core{1}
        Woutmat{iter} = wout_temp{res};
        Amat{iter} = A_temp{res};
        winmat{iter} = win_temp{res};
        iter = iter + 1;
    end
end

% Set output file names
beta_reservoir = rp.beta_reservoir;
start_iter = start_iter{1};
if strcmp(typeflag_in, 'hybrid')
    filename = [outputlocation,'/','DA-hybrid', '-numres', num2str(num_res_in), ...
        'res', num2str(reservoir_size), ...
        'localoverlap',num2str(locality{1}),'trainlen',num2str(train_lengthin),'sigma',...
        strrep(num2str(resparams.sigma_data),'.',''),'radius',strrep(num2str(resparams.radius),'.',''),...
        'leakage',strrep(num2str(leakage_in),'.',''),'covinf',num2str(covariance_inflation),...
            'betares',strrep(num2str(beta_reservoir),'.',''),...
            'runiter',num2str(runiterin),'_wnoise',strrep(num2str(resnoise{1}),'.',''),'_data.mat'];
elseif strcmp(typeflag_in, 'reservoir')
    filename = [outputlocation,'/','DA-reservoir', '-numres', num2str(num_res_in), ...
        'res', num2str(reservoir_size), ...
        'localoverlap',num2str(locality{1}),'trainlen',num2str(train_lengthin),'sigma',...
        strrep(num2str(resparams.sigma_data),'.',''),'radius',strrep(num2str(resparams.radius),'.',''),...
        'leakage',strrep(num2str(leakage_in),'.',''),'covinf',num2str(covariance_inflation),...
            'betares',strrep(num2str(beta_reservoir),'.',''),...
            'runiter',num2str(runiterin),'_wnoise',strrep(num2str(resnoise{1}),'.',''),'_data.mat'];
end

tm = matfile(testfilename_in);

%% Output predictions and CHyPP parameters
if ifsavepred
    if isstruct(ModelParams)
        save(filename, 'savepred','savepred_da','resparams','start_iter', 'Woutmat','Amat','winmat','ModelParams');
    else
        save(filename, 'savepred','savepred_da','resparams','start_iter', 'Woutmat','Amat','winmat');
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

datavar = datavar{1};
datamean = datamean{1};
prediction_average_diff = zeros(1,resparams.predict_length);
pred_length = zeros(1,resparams.predictions);
prediction_err = cell(1,resparams.predictions);
prediction_err_sum = 0;

if ifsaverms
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
end
%% Set rms data output file name and safe it specified
if strcmp(typeflag_in, 'hybrid')
    filename_rms = [outputlocation,'/','DA-hybrid', '-numres', num2str(num_res_in), ...
            'res', num2str(reservoir_size), ...
            'localoverlap',num2str(locality{1}),'trainlen',num2str(train_lengthin),'sigma',...
            strrep(num2str(resparams.sigma_data),'.',''),'radius',strrep(num2str(resparams.radius),'.',''),...
            'leakage',strrep(num2str(leakage_in),'.',''),'covinf',num2str(covariance_inflation),...
            'betares',strrep(num2str(beta_reservoir),'.',''),...
            'runiter',num2str(runiterin),'_wnoise',strrep(num2str(resnoise{1}),'.',''),'_rms.mat'];
elseif strcmp(typeflag_in, 'reservoir')
    filename_rms = [outputlocation,'/','DA-reservoir', '-numres', num2str(num_res_in), ...
            'res', num2str(reservoir_size), ...
            'localoverlap',num2str(locality{1}),'trainlen',num2str(train_lengthin),'sigma',...
            strrep(num2str(resparams.sigma_data),'.',''),'radius',strrep(num2str(resparams.radius),'.',''),...
            'leakage',strrep(num2str(leakage_in),'.',''),'covinf',num2str(covariance_inflation),...
            'betares',strrep(num2str(beta_reservoir),'.',''),...
            'runiter',num2str(runiterin),'_wnoise',strrep(num2str(resnoise{1}),'.',''),'_rms.mat'];
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