function [pool_size,avg_pred_length] = CHyPP(varargin)
pool_size = 1;
train_lengthin = 0;
reservoir_sizein = 1000;
avg_degreein = 3;
localityin = 0;
sigma_resin = 1;
radiusin = 0.6;
betain_res = 1e-4;
betain_model = 1e-4;
resnoisein = 0;
runiterin = 1;
ifsavepred = true;
ifsaverms = true;
trainfilename_in = 'KS_Data/KS_train_input_sequence.mat';
testfilename_in = 'KS_Data/KS_test_input_sequence.mat';
startfilename_in = 'KS_Data/KS_pred_start_indices.mat';
outputlocation = 'KS_Data';
ModelParams = 0;
TestModelParams = 0;


for arg = 1:numel(varargin)/2
    switch varargin{2*(arg-1)+1}
        case 'PoolSize'
            pool_size = varargin{2*arg};
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
        case 'Noise'
            resnoisein = varargin{2*arg};
        case 'RidgeReg'
            betain_res = varargin{2*arg};
            betain_model = varargin{2*arg};
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
        otherwise
            error(['Input variable ',num2str(arg),' not recognized.'])
    end
end

assert(~(strcmp(typeflag_in,'hybrid') & ~isstruct(ModelParams)),...
    'CHyPP has been told to run in a hybrid configuration, but no ModelParams struct has been given as input.')

avg_degree = Composite(pool_size);
sigma_res = Composite(pool_size);
radius = Composite(pool_size);
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

for i = 1:pool_size
    avg_degree{i} = avg_degreein;
    sigma_res{i} = sigma_resin;
    radius{i} = radiusin;
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
end

spmd(pool_size)

    m = matfile(trainfilename);
    tm = matfile(testfilename);
    startfile = load(startfilename);
    start_iter = startfile.start_iter;

    resparams.sigma_data = sigma_res;

    [len, num_inputs] = size(m, 'train_input_sequence');
        
    [test_len,~] = size(tm,'test_input_sequence');

    datamean = m.datamean;

    datavar = m.datavar;

    num_workers = numlabs; %numlabs is a matlab func that returns the number of workers allocated. equal to request_pool_size

    chunk_size = num_inputs/numlabs; %%%%%%%%%% MUST DIVIDE (each reservoir responsible for this chunk)

    l = labindex; % labindex is a matlab function that returns the worker index

    chunk_begin = chunk_size*(l-1)+1;

    chunk_end = chunk_size*l;

    rear_overlap = indexing_function_rear(chunk_begin, locality, num_inputs);  %spatial overlap on the one side

    forward_overlap = indexing_function_forward(chunk_end, locality, num_inputs);  %spatial overlap on the other side

    overlap_size = length(rear_overlap) + length(forward_overlap);

    approx_reservoir_size = reservoir_size;  % number of nodes in an individual reservoir network (approximate upto the next whole number divisible by number of inputs)

    resparams.sparsity = avg_degree/approx_reservoir_size;

    resparams.degree = avg_degree;

    resparams.N = approx_reservoir_size;

    resparams.discard_length = 1000;

    resparams.predict_length = 2000;

    resparams.sync_length = 100;
    
    if train_length == 0    %number of time steps used for training
        resparams.train_length = len - resparams.discard_length - 1;
    else
        resparams.train_length = train_length;  
    end

    resparams.predictions = 10; % define number of predictions (a prediction and synchronization is of length predict_length + sync_length)

    resparams.radius = radius;

    resparams.beta_model = beta_model; % ridge regression regularization parameter for the model 

    resparams.beta_reservoir = beta; % ridge regression regularization parameter for the reservoir states

    u = zeros(chunk_size + overlap_size, resparams.discard_length); % this will be populated by the input data to the reservoir

    if locality > 0
        if rear_overlap(end)<rear_overlap(1)
            u(1:locality,:) = [m.train_input_sequence(1:resparams.discard_length, rear_overlap(rear_overlap>rear_overlap(end))),...
                m.train_input_sequence(1:resparams.discard_length, rear_overlap(rear_overlap<=rear_overlap(end)))]';
        else
            u(1:locality,:) = m.train_input_sequence(1:resparams.discard_length,rear_overlap)';
        end

        if forward_overlap(end) < forward_overlap(1)
            u(locality+chunk_size+1:2*locality+chunk_size,:) = [m.train_input_sequence(1:resparams.discard_length,forward_overlap(forward_overlap>forward_overlap(end))),...
                m.train_input_sequence(1:resparams.discard_length,forward_overlap(forward_overlap<=forward_overlap(end)))]';
        else
            u(locality+chunk_size+1:2*locality+chunk_size,:) = m.train_input_sequence(1:resparams.discard_length,forward_overlap)';
        end
    end

    u(locality+1:locality+chunk_size,:) = m.train_input_sequence(1:resparams.discard_length, chunk_begin:chunk_end)';

    noise = zeros(chunk_size + overlap_size, resparams.discard_length); % this will be populated by the input data to the reservoir

    if locality > 0
        if rear_overlap(end)<rear_overlap(1)
            noise(1:locality,:) = [m.noise(1:resparams.discard_length, rear_overlap(rear_overlap>rear_overlap(end))),...
                m.noise(1:resparams.discard_length, rear_overlap(rear_overlap<=rear_overlap(end)))]';
        else
            noise(1:locality,:) = m.noise(1:resparams.discard_length,rear_overlap)';
        end

        if forward_overlap(end) < forward_overlap(1)
            noise(locality+chunk_size+1:2*locality+chunk_size,:) = [m.noise(1:resparams.discard_length,forward_overlap(forward_overlap>forward_overlap(end))),...
                m.noise(1:resparams.discard_length,forward_overlap(forward_overlap<=forward_overlap(end)))]';
        else
            noise(locality+chunk_size+1:2*locality+chunk_size,:) = m.noise(1:resparams.discard_length,forward_overlap)';
        end
    end

    noise(locality+1:locality+chunk_size,:) = m.noise(1:resparams.discard_length, chunk_begin:chunk_end)';

    testu = zeros(chunk_size + overlap_size, test_len); % this will be populated by the test data used for synchronization

    if locality > 0
        if rear_overlap(end)<rear_overlap(1)
            testu(1:locality,:) = [tm.test_input_sequence(:, rear_overlap(rear_overlap>rear_overlap(end))),...
                tm.test_input_sequence(:, rear_overlap(rear_overlap<=rear_overlap(end)))]';
        else
            testu(1:locality,:) = tm.test_input_sequence(:,rear_overlap)';
        end

        if forward_overlap(end) < forward_overlap(1)
            testu(locality+chunk_size+1:2*locality+chunk_size,:) = [tm.test_input_sequence(:,forward_overlap(forward_overlap>forward_overlap(end))),...
                tm.test_input_sequence(:,forward_overlap(forward_overlap<=forward_overlap(end)))]';
        else
            testu(locality+chunk_size+1:2*locality+chunk_size,:) = tm.test_input_sequence(:,forward_overlap)';
        end
    end

    testu(locality+1:locality+chunk_size,:) = tm.test_input_sequence(:, chunk_begin:chunk_end)';



    A = generate_reservoir(resparams.N, resparams.radius, resparams.degree, labindex, runiter, num_workers);

    input_size = (chunk_size + overlap_size);

    q = floor(resparams.N/(input_size));
    
    win = zeros(resparams.N, input_size);

    for i=1:input_size
        rng(i)
        ip = (-1 + 2*rand(q,1));
        win((i-1)*q+1:i*q,i) = resparams.sigma_data*ip;
    end

    occupied_nodes = floor(resparams.N/(input_size))*input_size;
    leftover_nodes = resparams.N - occupied_nodes;

    for i=1:leftover_nodes
        rng(i+input_size)
        ip = (-1 + 2*rand);
        win(occupied_nodes+i,randi([1,input_size])) = resparams.sigma_data*ip;
    end
    
    % Define number of steps over which to train and the length of each
    % training section

    train_steps = 50;
    assert(mod(resparams.train_length/train_steps,1)==0,...
        'train_steps must divide train_length. Change train_steps to a whole number that divides train_length.')
    train_size  = resparams.train_length/train_steps;

    states = zeros(resparams.N, train_size);

    x = zeros(resparams.N,1);

end

rp = resparams{1};

train_file = m{1};

data_trstates = Composite(num_workers{1});
states_trstates = Composite(num_workers{1});
if strcmp(typeflag{1}, 'hybrid')
    for i=1:num_workers{1}
        data_trstates{i} = zeros(chunk_size{1},rp.N+chunk_size{1});
        states_trstates{i} = zeros(rp.N+chunk_size{1});
    end
elseif strcmp(typeflag{1},'reservoir')
    for i=1:num_workers{1}
        data_trstates{i} = zeros(chunk_size{1},rp.N);
        states_trstates{i} = zeros(rp.N);
    end
end

spmd(pool_size)
    for i = 1:resparams.discard_length-1
        x = tanh(A*x + win*(u(:,i)+resnoise*noise(:,i)));
    end
    states(:,1) = x;
end

for k = 1:train_steps{1}

    % Perform training over a number of train_steps, only importing
    % training data for each section each time
    if strcmp(typeflag{1},'hybrid')
        if k == train_steps{1}
            train_input = datavar{1}.*(train_file.train_input_sequence(rp.discard_length+(k-1)*train_size{1}:rp.discard_length+k*train_size{1},:)'+...
                resnoise{1}*train_file.noise(rp.discard_length+(k-1)*train_size{1}:rp.discard_length+k*train_size{1},:)')+ datamean{1};

            model_forecast = zeros(num_inputs{1}, train_size{1}+1);

            for j = 1:train_size{1}+1
                model_forecast(:,j) = ModelParams.predict(train_input(:,j), ModelParams);
            end

            model_states = Composite(num_workers{1});

            for j = 1:num_workers{1}

                v = zeros(input_size{1}, train_size{1}+1);

                if locality{1} > 0

                    v(1:locality{1},:) = model_forecast(rear_overlap{j}, 1:end);

                    v(locality{1}+chunk_size{1}+1:2*locality{1}+chunk_size{1},:) = model_forecast(forward_overlap{j}, 1:end);

                end

                v(locality{1}+1:locality{1}+chunk_size{1},:) = model_forecast(chunk_begin{j}:chunk_end{j}, 1:end);

                model_states{j} = v;

            end
        else
            train_input = datavar{1}.*(train_file.train_input_sequence(rp.discard_length+(k-1)*train_size{1}:rp.discard_length+k*train_size{1}-1,:)'+...
                resnoise{1}*train_file.noise(rp.discard_length+(k-1)*train_size{1}:rp.discard_length+k*train_size{1}-1,:)')+ datamean{1};

            model_forecast = zeros(num_inputs{1}, train_size{1});

            for j = 1:train_size{1}
                model_forecast(:,j) = ModelParams.predict(train_input(:,j), ModelParams);
            end

            model_states = Composite(num_workers{1});

            for j = 1:num_workers{1}

                v = zeros(input_size{1}, train_size{1});

                if locality{1} > 0

                    v(1:locality{1},:) = model_forecast(rear_overlap{j}, 1:end);

                    v(locality{1}+chunk_size{1}+1:2*locality{1}+chunk_size{1},:) = model_forecast(forward_overlap{j}, 1:end);

                end

                v(locality{1}+1:locality{1}+chunk_size{1},:) = model_forecast(chunk_begin{j}:chunk_end{j}, 1:end);

                model_states{j} = v;

            end
        end
    end

    kc = Composite(num_workers{1});
    for i=1:num_workers{1}
        kc{i} = k;
    end
    k = kc;

    spmd(pool_size)
        u = zeros(chunk_size + overlap_size, train_size); % this will be populated by the input data to the reservoir
        if locality > 0
            if rear_overlap(end)<rear_overlap(1)
                u(1:locality,:) = [m.train_input_sequence(resparams.discard_length+(k-1)*train_size+1:resparams.discard_length+k*train_size,...
                    rear_overlap(rear_overlap>rear_overlap(end))),...
                    m.train_input_sequence(resparams.discard_length+(k-1)*train_size+1:resparams.discard_length+k*train_size,...
                    rear_overlap(rear_overlap<=rear_overlap(end)))]';
            else
                u(1:locality,:) = m.train_input_sequence(resparams.discard_length+...
                    (k-1)*train_size+1:resparams.discard_length+k*train_size,rear_overlap)';
            end

            if forward_overlap(end) < forward_overlap(1)
                u(locality+chunk_size+1:2*locality+chunk_size,:) = [m.train_input_sequence(resparams.discard_length+...
                    (k-1)*train_size+1:resparams.discard_length+k*train_size,forward_overlap(forward_overlap>forward_overlap(end))),...
                    m.train_input_sequence(resparams.discard_length+(k-1)*train_size+1:resparams.discard_length+...
                    k*train_size,forward_overlap(forward_overlap<=forward_overlap(end)))]';
            else
                u(locality+chunk_size+1:2*locality+chunk_size,:) = m.train_input_sequence(resparams.discard_length+(k-1)*train_size+1:resparams.discard_length+k*train_size,forward_overlap)';
            end
        end
        u(locality+1:locality+chunk_size,:) = m.train_input_sequence(resparams.discard_length+...
            (k-1)*train_size+1:resparams.discard_length+k*train_size, chunk_begin:chunk_end)';

        noise = zeros(chunk_size + overlap_size, train_size); % this will be populated by the input data to the reservoir
        if locality > 0
            if rear_overlap(end)<rear_overlap(1)
                noise(1:locality,:) = [m.noise(resparams.discard_length+(k-1)*train_size+1:resparams.discard_length+k*train_size,...
                    rear_overlap(rear_overlap>rear_overlap(end))),...
                    m.noise(resparams.discard_length+(k-1)*train_size+1:resparams.discard_length+k*train_size,...
                    rear_overlap(rear_overlap<=rear_overlap(end)))]';
            else
                noise(1:locality,:) = m.noise(resparams.discard_length+...
                    (k-1)*train_size+1:resparams.discard_length+k*train_size,rear_overlap)';
            end

            if forward_overlap(end) < forward_overlap(1)
                noise(locality+chunk_size+1:2*locality+chunk_size,:) = [m.noise(resparams.discard_length+...
                    (k-1)*train_size+1:resparams.discard_length+k*train_size,forward_overlap(forward_overlap>forward_overlap(end))),...
                    m.noise(resparams.discard_length+(k-1)*train_size+1:resparams.discard_length+...
                    k*train_size,forward_overlap(forward_overlap<=forward_overlap(end)))]';
            else
                noise(locality+chunk_size+1:2*locality+chunk_size,:) = m.noise(resparams.discard_length+(k-1)*train_size+1:resparams.discard_length+k*train_size,forward_overlap)';
            end
        end
        noise(locality+1:locality+chunk_size,:) = m.noise(resparams.discard_length+...
            (k-1)*train_size+1:resparams.discard_length+k*train_size, chunk_begin:chunk_end)';


        for i = 1:train_size-1
            states(:,i+1) = tanh(A*states(:,i) + win*(u(:,i)+resnoise*noise(:,i)));
        end

        x = states(:, end);

        states(2:2:resparams.N,:) = states(2:2:resparams.N,:).^2;        

        if strcmp(typeflag,'hybrid')
            if k == train_steps
                augmented_states = vertcat((model_states(locality+1:locality+chunk_size,1:end-1)-datamean)./datavar, states);
                local_model = model_states(:,end);
            else
                augmented_states = vertcat((model_states(locality+1:locality+chunk_size,:)-datamean)./datavar, states);
            end
        elseif strcmp(typeflag, 'reservoir')
            augmented_states = states;
        end

        data_trstates = data_trstates + u(locality+1:locality+chunk_size, :)*augmented_states';
        states_trstates = states_trstates + augmented_states*augmented_states';


        if k == train_steps
            states(:,1) = tanh(A*x + win*(u(:,end)+resnoise*noise(:,end)));
        else
            states(:,1) = tanh(A*x + win*(u(:,end)+resnoise*noise(:,end)));
        end

    end
end
spmd(pool_size)
%%
    if strcmp(typeflag, 'hybrid')
        idenmat = sparse(diag([resparams.beta_model*ones(1,chunk_size),resparams.beta_reservoir*ones(1,resparams.N)]));
    elseif strcmp(typeflag, 'reservoir')
        idenmat = sparse(diag(resparams.beta_reservoir*ones(1,resparams.N)));
    end
    % Train output matrix
    wout = data_trstates/(states_trstates+idenmat);

    if labindex == 1
        prediction = cell(resparams.predictions,1);
        for j = 1:resparams.predictions
            prediction{j} =  zeros(num_inputs, resparams.predict_length);
        end
    end

    for j = 1:resparams.predictions
        x = zeros(size(x));
        if labindex == 1
            test_input = datavar.*tm.test_input_sequence(start_iter(j):start_iter(j)+resparams.sync_length-1,:)'+datamean;
        end
        for i=1:resparams.sync_length
            feedback = testu(:,start_iter(j)+(i-1));

            if strcmp(typeflag, 'hybrid') && i==resparams.sync_length
                if labindex == 1
                    forecast_out = ModelParams.predict( test_input(:,i), ModelParams);
                    global_model_forecast = labBroadcast(1,forecast_out);
                else
                    global_model_forecast = labBroadcast(1);
                end

                labBarrier;

                local_model = zeros(input_size,1);

                if locality > 0

                    local_model(1:locality) = global_model_forecast(rear_overlap,1);

                    local_model(locality+chunk_size+1:2*locality+chunk_size) = global_model_forecast(forward_overlap,1);
                end

                local_model(locality+1:locality+chunk_size) = global_model_forecast(chunk_begin:chunk_end,1);
            end

            x = tanh(A*x + win*feedback);
        end

        x_ = x;
        x_(2:2:resparams.N) = x_(2:2:resparams.N).^2;

        if strcmp(typeflag, 'hybrid')
            augmented_x = vertcat((local_model(locality+1:locality+chunk_size) - datamean)./datavar, x_);
        elseif strcmp(typeflag, 'reservoir')
            augmented_x = x_;
        end
%        out = local_model(locality+1:locality+chunk_size,:);
        out = wout*augmented_x;
        labBarrier;
        concatenated_out = gcat(out, 1);

        for pred_idx = 1:resparams.predict_length

            feedback = zeros(input_size,1);

            if locality > 0
                feedback(1:locality) = concatenated_out(rear_overlap);

                feedback(locality+chunk_size+1:2*locality+chunk_size) = concatenated_out(forward_overlap);
            end

            feedback(locality+1:locality+chunk_size) = concatenated_out(chunk_begin:chunk_end);

            if strcmp(typeflag, 'hybrid')
                if labindex == 1
                    forecast_out = ModelParams.predict( datavar.*concatenated_out + datamean, ModelParams);
                    global_model_forecast = labBroadcast(1,forecast_out);

                else
                    global_model_forecast = labBroadcast(1);
                end
                labBarrier;

                local_model = zeros(input_size,1);

                if locality > 0
                    local_model(1:locality) = global_model_forecast(rear_overlap,1);

                    local_model(locality+chunk_size+1:2*locality+chunk_size) = global_model_forecast(forward_overlap,1);
                end

                local_model(locality+1:locality+chunk_size) = global_model_forecast(chunk_begin:chunk_end,1);
            end

            x = tanh(A*x + win*feedback);
            x_ = x;
            x_(2:2:resparams.N) = x_(2:2:resparams.N).^2;

            if strcmp(typeflag, 'hybrid')
                augmented_x = vertcat((local_model(locality+1:locality+chunk_size) - datamean)./datavar, x_);
            elseif strcmp(typeflag, 'reservoir')
                augmented_x = x_;
            end
    %        out = local_model(locality+1:locality+chunk_size,:);
            out = wout*augmented_x;
            labBarrier;
            concatenated_out = gcat(out, 1);  

            if labindex == 1
                prediction{j}(:,pred_idx) = concatenated_out;
            end

        end
    end
end

approx_reservoir_size = approx_reservoir_size{1};

savepred = prediction{1};
resparams = resparams{1};
Woutmat = {};
for i=1:pool_size
    Woutmat{i} = wout{i};
end

beta_reservoir = rp.beta_reservoir;
start_iter = start_iter{1};
if strcmp(typeflag_in, 'hybrid')
    filename = [outputlocation,'/','hybrid', '-pool', num2str(pool_size), ...
        'res', num2str(approx_reservoir_size), ...
        'localoverlap',num2str(locality{1}),'trainlen',num2str(train_lengthin),'sigma',...
        strrep(num2str(resparams.sigma_data),'.',''),'radius',strrep(num2str(resparams.radius),'.',''),'betares',strrep(num2str(beta_reservoir),'.',''),...
        'runiter',num2str(runiterin),'_wnoise',strrep(num2str(resnoise{1}),'.',''),'_data.mat'];
elseif strcmp(typeflag_in, 'reservoir')
    filename = [outputlocation,'/','reservoir', '-pool', num2str(pool_size), ...
        'res', num2str(approx_reservoir_size), ...
        'localoverlap',num2str(locality{1}),'trainlen',num2str(train_lengthin),'sigma',...
        strrep(num2str(resparams.sigma_data),'.',''),'radius',strrep(num2str(resparams.radius),'.',''),'betares',strrep(num2str(beta_reservoir),'.',''),...
        'runiter',num2str(runiterin),'_wnoise',strrep(num2str(resnoise{1}),'.',''),'_data.mat'];
end

tm = matfile(testfilename_in);

if ifsavepred
    if isstruct(ModelParams)
        save(filename, 'savepred','resparams','start_iter', 'Woutmat','ModelParams');
    else
        save(filename, 'savepred','resparams','start_iter', 'Woutmat');
    end
end

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
error_cutoff = 0.2;
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

if strcmp(typeflag_in, 'hybrid')
    filename_rms = [outputlocation,'/','hybrid', '-pool', num2str(pool_size), ...
            'res', num2str(approx_reservoir_size), ...
            'localoverlap',num2str(locality{1}),'trainlen',num2str(train_lengthin),'sigma',...
            strrep(num2str(resparams.sigma_data),'.',''),'radius',strrep(num2str(resparams.radius),'.',''),'betares',strrep(num2str(beta_reservoir),'.',''),...
            'runiter',num2str(runiterin),'_wnoise',strrep(num2str(resnoise{1}),'.',''),'_rms.mat'];
elseif strcmp(typeflag_in, 'reservoir')
    filename_rms = [outputlocation,'/','reservoir', '-pool', num2str(pool_size), ...
            'res', num2str(approx_reservoir_size), ...
            'localoverlap',num2str(locality{1}),'trainlen',num2str(train_lengthin),'sigma',...
            strrep(num2str(resparams.sigma_data),'.',''),'radius',strrep(num2str(resparams.radius),'.',''),'betares',strrep(num2str(beta_reservoir),'.',''),...
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