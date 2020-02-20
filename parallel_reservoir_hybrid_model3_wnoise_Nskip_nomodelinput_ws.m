function [pool_size,avg_pred_length] = ...
    parallel_reservoir_hybrid_model3_wnoise_Nskip_nomodelinput_ws(pool_size,train_lengthin,...
    reservoir_sizein,localityin,sigma_resin, radiusin, betain, resnoisein, Nskipin, runiterin, ifsavepred, ifsaverms)
% pool_size = 5;
Iin = 0;
sigmain = 0;
% sigma_resin = 1;
% radiusin = 0.6;
betain_res = betain;
betain_model = 1e-4;
% resnoisein = 0;
% runiterin = 1;


good_pred_frac = zeros(1,5);
average_pred_length = zeros(1,5);
hybrid_average_diff_tot = {};
for jid = 1:1
    
    I = Composite(pool_size);
    sigma = Composite(pool_size);
    sigma_res = Composite(pool_size);
    radius = Composite(pool_size);
    beta = Composite(pool_size);
    beta_model = Composite(pool_size);
    resnoise = Composite(pool_size);
    reservoir_size = Composite(pool_size);
    runiter = Composite(pool_size);
    jobid = Composite(pool_size);
    train_length = Composite(pool_size);
    locality = Composite(pool_size);
    Nskip = Composite(pool_size);
    
    
    for i = 1:pool_size
        jobid{i} = jid;
        I{i} = Iin;
        sigma{i} = sigmain;
        sigma_res{i} = sigma_resin;
        radius{i} = radiusin;
        beta{i} = betain_res;
        beta_model{i} = betain_model;
        resnoise{i} = resnoisein;
        reservoir_size{i} = reservoir_sizein;
        runiter{i} = runiterin;
        train_length{i} = train_lengthin;
        locality{i} = localityin;
        Nskip{i} = Nskipin;
    end
    
    spmd(pool_size)
        
        runflag = 'cluster';
        
        typeflag = 'hybrid';
        
        bigflag = false;

        if strcmp(runflag, 'cluster')
            
            if bigflag
            
                datafilename = ['/lustre/awikner1/LorenzLorenzModel3/N240F8_' num2str(jobid) '/train_input_sequence_I',num2str(I),'_big.mat'];
                testfilename = ['/lustre/awikner1/LorenzLorenzModel3/N240F8_' num2str(jobid) '/test_input_sequence_I',num2str(I),'_big_wnoise.mat'];
            else
                resdatafilename  = ['/lustre/awikner1/LorenzModel3/N960K32I12F15wnoiseNskip',num2str(Nskip),'_',num2str(jobid),'/train_input_sequence.mat'];
                testfilename  = ['/lustre/awikner1/LorenzModel3/N960K32I12F15wnoiseNskip',num2str(Nskip),'_',num2str(jobid),'/test_input_sequence.mat'];
                startfilename = ['/lustre/awikner1/LorenzModel3/N960K32I12F15wnoiseNskip',num2str(Nskip),'_',num2str(jobid),'/pred_start_indices_200000.mat'];
%                 datafilename = ['/lustre/awikner1/LorenzLorenzModel3/N960F15_' num2str(jobid) '/train_input_sequence_I',num2str(I),'.mat'];
%                 testfilename = ['/lustre/awikner1/LorenzLorenzModel3/N960F15_' num2str(jobid) '/test_input_sequence_I',num2str(I),'.mat'];
            end
   
        elseif strcmp(runflag, 'local')
            
            if bigflag
                datafilename = ['N240F8_1/train_input_sequence_I',num2str(I),'_big.mat'];
                testfilename = ['N240F8_1/test_input_sequence_I',num2str(I),'_big_wnoise.mat'];
            else
%                 datafilename = ['Data/N960K32I12F15noise',strrep(num2str(noise),'.',''),'_',num2str(jobid),'/train_input_sequence.mat'];
%                 testfilename = ['Data/N960K32I12F15noise',strrep(num2str(noise),'.',''),'_',num2str(jobid),'/test_input_sequence.mat'];
%                 startfilename = ['Data/N960K32I12F15noise',strrep(num2str(noise),'.',''),'_' num2str(jobid) '/pred_start_indices_200000.mat'];
                resdatafilename = ['Data/N960K32I12F15wnoiseNskip',num2str(Nskip),'_',num2str(jobid),'/train_input_sequence.mat'];
                testfilename = ['Data/N960K32I12F15wnoiseNskip',num2str(Nskip),'_',num2str(jobid),'/test_input_sequence.mat'];
                startfilename = ['Data/N960K32I12F15wnoiseNskip',num2str(Nskip),'_' num2str(jobid) '/pred_start_indices_200000.mat'];
            end
            
            
                    
        end

        m = matfile(resdatafilename);
        tm = matfile(testfilename);
        startfile = load(startfilename);
        start_iter = startfile.start_iter;

        resparams.sigma_data = sigma_res;
        
        resparams.sigma_model = sigma; % Define separate input weights for the model and previous reservoir state

        [len, num_inputs] = size(m, 'train_input_sequence');
        
        [test_len,~] = size(tm,'test_input_sequence');
        
        datamean = m.datamean;
        
        datavar = m.datavar;

        num_workers = numlabs; %numlabs is a matlab func that returns the number of workers allocated. equal to request_pool_size

        chunk_size = num_inputs/numlabs; %%%%%%%%%% MUST DIVIDE (each reservoir responsible for this chunk)

        l = labindex; % labindex is a matlab function that returns the worker index

        chunk_begin = chunk_size*(l-1)+1;

        chunk_end = chunk_size*l;

        % locality = 20; % there are restrictions on the allowed range of this parameter. check documentation

        rear_overlap = indexing_function_rear(chunk_begin, locality, num_inputs);  %spatial overlap on the one side

        forward_overlap = indexing_function_forward(chunk_end, locality, num_inputs);  %spatial overlap on the other side

        overlap_size = length(rear_overlap) + length(forward_overlap);

        approx_reservoir_size = reservoir_size;  % number of nodes in an individual reservoir network (approximate upto the next whole number divisible by number of inputs)

        avg_degree = 3; %average connection degree

        resparams.sparsity = avg_degree/approx_reservoir_size;

        resparams.degree = avg_degree;

%         nodes_per_input = round(approx_reservoir_size/((chunk_size+overlap_size)));
% 
%         resparams.N = nodes_per_input*(chunk_size+overlap_size); % exact number of nodes in the network
        
        resparams.N = approx_reservoir_size;

        if bigflag
            resparams.train_length = 989000;
        else
            resparams.train_length = train_length;  %number of time steps used for training
        end

        resparams.discard_length = 1000;  %number of time steps used to discard transient (generously chosen)

        resparams.predict_length = 200000; %number of steps to be predicted

        resparams.sync_length = 100; % set length for synchronization before each prediction
        
        resparams.predictions = 1; % define number of predictions (a prediction and synchronization is of length predict_length + sync_length
        
        resparams.radius = radius; % spectral radius of the reservoir

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

%         q = resparams.N/(input_size);
        
%         win = zeros(resparams.N, input_size);
% 
%         for i=1:input_size
%             rng(i)
%             ip = (-1 + 2*rand(q,1));
%             win((i-1)*q+1:i*q,i) = resparams.sigma_data*ip;
%         end
%         win = resparams.sigma_data*(-1+2*rand(resparams.N,input_size));
        q = floor(resparams.N/(input_size));
%         
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
        train_size  = resparams.train_length/train_steps;
        
        states = zeros(resparams.N, train_size);

        x = zeros(resparams.N,1);
        
    end
    
    rp = resparams{1};
%     rp
    
    ModelParams.N = 960/Nskipin;
    ModelParams.K = ModelParams.N/30;
    ModelParams.F = 15;
    ModelParams.b = 10;
    ModelParams.c = 2.5;
    ModelParams.I = 12;
    ModelParams.tau = 0.005;
    ModelParams.noise = 0;

    ModelParams.alpha = (3*(ModelParams.I)^2 + 3)/(2*(ModelParams.I^3) + 4*ModelParams.I);
    ModelParams.beta = (2*(ModelParams.I)^2 + 1)/((ModelParams.I^4) + 2*(ModelParams.I)^2);
    
    Z2Xmat = sparse(Z2X(ModelParams));
    s_mat_k = sparse(getsmat(ModelParams.N, ModelParams.K));
    
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
    
    % Evaluate model 2 over the discard length to set an initial reservoir
    % state
    
%     train_input = datavar{1}.*train_file.train_input_sequence(1:rp.discard_length-1,:)' + datamean{1};
% 
%     model_forecast = zeros(num_inputs{1}, rp.discard_length-1);
% 
%     for j = 1:rp.discard_length-1
%         model_forecast(:,j) = rk4_m1_skip(@m1, train_input(:,j), ModelParams);
%     end
% 
%     model_states = Composite(num_workers{1});
% 
%     for j = 1:num_workers{1}
% 
%         v = zeros(input_size{1}, rp.discard_length-1);
%         
%         if locality{1} > 0
%             v(1:locality{1},:) = model_forecast(rear_overlap{j}, 1:end);
% 
%             v(locality{1}+chunk_size{1}+1:2*locality{1}+chunk_size{1},:) = model_forecast(forward_overlap{j}, 1:end);
%         end
%         
%         v(locality{1}+1:locality{1}+chunk_size{1},:) = model_forecast(chunk_begin{j}:chunk_end{j}, 1:end);
% 
%         model_states{j} = v;
% 
%     end
    
    spmd(pool_size)
        for i = 1:resparams.discard_length-1
            x = tanh(A*x + win*(u(:,i)+resnoise*noise(:,i)));
        end
        states(:,1) = x;
    end
    
    for k = 1:train_steps{1}
        % disp(k)
        % Perform training over a number of train_steps, only importing
        % training data for each section each time
        if strcmp(typeflag{1},'hybrid')
            if k == train_steps{1}
                train_input = datavar{1}.*(train_file.train_input_sequence(rp.discard_length+(k-1)*train_size{1}:rp.discard_length+k*train_size{1},:)'+...
                    resnoise{1}*train_file.noise(rp.discard_length+(k-1)*train_size{1}:rp.discard_length+k*train_size{1},:)')+ datamean{1};

                model_forecast = zeros(num_inputs{1}, train_size{1}+1);

                for j = 1:train_size{1}+1
                    model_forecast(:,j) = rk4m3(@m2,train_input(:,j), ModelParams, s_mat_k, Z2Xmat);
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
                    model_forecast(:,j) = rk4m3(@m2,train_input(:,j), ModelParams, s_mat_k, Z2Xmat);
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

            
            
%             fitu = zeros(chunk_size + overlap_size, train_size); % this will be populated by the input data to the reservoir
%             if locality > 0
%                 if rear_overlap(end)<rear_overlap(1)
%                     fitu(1:locality,:) = [m.train_input_sequence_nonoise(resparams.discard_length+(k-1)*train_size+1:resparams.discard_length+k*train_size,...
%                         rear_overlap(rear_overlap>rear_overlap(end))),...
%                         m.train_input_sequence_nonoise(resparams.discard_length+(k-1)*train_size+1:resparams.discard_length+k*train_size,...
%                         rear_overlap(rear_overlap<=rear_overlap(end)))]';
%                 else
%                     fitu(1:locality,:) = m.train_input_sequence(resparams.discard_length+...
%                         (k-1)*train_size+1:resparams.discard_length+k*train_size,rear_overlap)';
%                 end
% 
%                 if forward_overlap(end) < forward_overlap(1)
%                     fitu(locality+chunk_size+1:2*locality+chunk_size,:) = [m.train_input_sequence_nonoise(resparams.discard_length+...
%                         (k-1)*train_size+1:resparams.discard_length+k*train_size,forward_overlap(forward_overlap>forward_overlap(end))),...
%                         m.train_input_sequence_nonoise(resparams.discard_length+(k-1)*train_size+1:resparams.discard_length+...
%                         k*train_size,forward_overlap(forward_overlap<=forward_overlap(end)))]';
%                 else
%                     fitu(locality+chunk_size+1:2*locality+chunk_size,:) = m.train_input_sequence_nonoise(resparams.discard_length+(k-1)*train_size+1:resparams.discard_length+k*train_size,forward_overlap)';
%                 end
%             end
%             fitu(locality+1:locality+chunk_size,:) = m.train_input_sequence_nonoise(resparams.discard_length+...
%                     (k-1)*train_size+1:resparams.discard_length+k*train_size, chunk_begin:chunk_end)';

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
            
            % wout = fit(resparams, augmented_states, u(locality+1:locality+chunk_size, :));
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
%         test_input_sequence = tm.test_input_sequence;
        
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
                        forecast_out = rk4m3(@m2, test_input(:,i), ModelParams, s_mat_k, Z2Xmat);
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
                        forecast_out = rk4m3(@m2, datavar.*concatenated_out + datamean, ModelParams,s_mat_k, Z2Xmat);
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
    Amat = {};
    Woutmat = {};
    for i=1:pool_size
        Amat{i} = A{i};
        Woutmat{i} = wout{i};
    end
    sigma_data = rp.sigma_data;
    sigma_model = rp.sigma_model;
    
    beta_reservoir = rp.beta_reservoir;
    beta_model = rp.beta_model;
    start_iter = start_iter{1};
    

    if strcmp(runflag{1}, 'cluster')
        
%         filelocation = ['/lustre/awikner1/LorenzModel3/N960K32I12F15noise',strrep(num2str(resnoise{1}),'.',''),'_',num2str(jid),'/'];
        filelocation = ['/lustre/awikner1/LorenzModel3/N960K32I12F15wnoiseNskip',num2str(Nskipin),'_',num2str(jid),'/'];
        
    elseif strcmp(runflag{1}, 'local')
        
%         filelocation = ['./Data/N960K32I12F15noise',strrep(num2str(noise{1}),'.',''),'_',num2str(jid),'/'];
        filelocation = ['./Data/N960K32I12F15wnoiseNskip',num2str(Nskipin),'_',num2str(jid),'/'];
     
    end
    
    if bigflag{1}
        filename = [ filelocation, 'hybrid', '-pool', num2str(pool_size), ...
            'res', num2str(approx_reservoir_size), '10sigma_', num2str(10*sigma_data),'_',num2str(10*sigma_model), ...
            'locality',num2str(locality{1}),'jobid', num2str(jid), 'beta_',...
            strrep(num2str(beta_reservoir),'.',''),'_',strrep(num2str(beta_model),'.',''),'_I',num2str(Iin), '_bigsync_wnoise',strrep(num2str(resnoisein),'.',''),'.mat'];
    else
        if strcmp(typeflag{1}, 'hybrid')
            filename = [ filelocation, 'fixed_hybrid', '-pool', num2str(pool_size), ...
                'res', num2str(approx_reservoir_size), ...
                'locality',num2str(locality{1}),'trainlen',num2str(train_lengthin),'sigma',...
                strrep(num2str(resparams.sigma_data),'.',''),'radius',strrep(num2str(resparams.radius),'.',''),'betares',strrep(num2str(beta_reservoir),'.',''),...
                'jobid', num2str(jid),'runiter',num2str(runiterin),'_wnoise',strrep(num2str(resnoise{1}),'.',''),'Nskip',num2str(Nskipin),'_data_longpred.mat'];
        elseif strcmp(typeflag{1}, 'reservoir')
            filename = [ filelocation, 'fixed_reservoir', '-pool', num2str(pool_size), ...
                'res', num2str(approx_reservoir_size), ...
                'locality',num2str(locality{1}),'trainlen',num2str(train_lengthin),'sigma',...
                strrep(num2str(resparams.sigma_data),'.',''),'radius',strrep(num2str(resparams.radius),'.',''),'betares',strrep(num2str(beta_reservoir),'.',''),...
                'jobid', num2str(jid),'runiter',num2str(runiterin),'_wnoise',strrep(num2str(resnoise{1}),'.',''),'Nskip',num2str(Nskipin),'_data_longpred.mat'];
        end
    end
    
    if strcmp(runflag{1}, 'cluster')
        
        testfilename = ['/lustre/awikner1/LorenzModel3/N960K32I12F15wnoiseNskip',num2str(Nskipin),'_',num2str(jid),'/test_input_sequence.mat'];
        
    elseif strcmp(runflag{1}, 'local')
        
%         testfilename = ['./Data/N960K32I12F15noise',strrep(num2str(noise{1}),'.',''),'_' num2str(jid) '/test_input_sequence.mat'];
        testfilename = ['./Data/N960K32I12F15wnoiseNskip',num2str(Nskipin),'_',num2str(jid),'/test_input_sequence.mat'];
     
    end
    tm = matfile(testfilename);
    
    prediction_lengths = zeros(1,rp.predictions);
    prediction_std = zeros(1,rp.predictions);
    good_predictions = 0;
    for i=1:rp.predictions
        iter = 0;
        for j = 1:rp.predict_length
            if any(isnan(savepred{i}(:,j)))
                break
            end
            iter = iter + 1;
            
        end
        prediction_lengths(i) = iter;
        if iter == rp.predict_length
            good_predictions = good_predictions+1;
            prediction_std(i) = std(savepred{i}(:));
        elseif iter > 100
            usable_data = savepred{i}(:,1:iter-100);
            prediction_std(i) = std(usable_data(:));
        else
            prediction_std(i) = NaN;
        end
    end
    good_pred_frac(jid) = good_predictions/rp.predictions;
    
    data = tm.test_input_sequence;
    data_std = std(data(:));
    if strcmp(typeflag{1}, 'hybrid')
        filename_preds = [ filelocation, 'hybrid', '-pool', num2str(pool_size), ...
                'res', num2str(approx_reservoir_size),  ...
                'locality',num2str(locality{1}),'trainlen',num2str(train_lengthin),'betares',...
                strrep(num2str(beta_reservoir),'.',''),'runiter',num2str(runiterin),'_wnoise',strrep(num2str(resnoise{1}),'.',''),'_predlength.mat'];
    elseif strcmp(typeflag{1}, 'reservoir')
        filename_preds = [ filelocation, 'reservoir', '-pool', num2str(pool_size), ...
                'res', num2str(approx_reservoir_size), ...
                'locality',num2str(locality{1}),'trainlen',num2str(train_lengthin),'betares',...
                strrep(num2str(beta_reservoir),'.',''),'runiter',num2str(runiterin),'_wnoise',strrep(num2str(resnoise{1}),'.',''),'_predlength.mat'];
    end 
    if ifsavepred
        save(filename, 'savepred','resparams','start_iter', 'Woutmat');
%       save(filename_preds, 'prediction_lengths','good_predictions','prediction_std','data_std');
    end

    ModelParams.nstep = resparams.predict_length;
    datavar = datavar{1};
    datamean = datamean{1};
%     test_input_sequence = test_input_sequence{1};
    
    model_average_diff = zeros(1,ModelParams.nstep);
    hybrid_average_diff = zeros(1,resparams.predict_length);
    model_pred_length = zeros(1,resparams.predictions);
    pred_length = zeros(1,resparams.predictions);
    hybrid_err = cell(1,resparams.predictions);
    model_err  = cell(1,resparams.predictions);
    error_cutoff = 0.85;
    iter = 0;
    hybrid_err_sum = 0;
    
%     ModelParams.const = 0;
%     % Precompute various ETDRK4 scalar quantities:
%     k = [0:ModelParams.N/2-1 0 -ModelParams.N/2+1:-1]'*(2*pi/ModelParams.d); % wave numbers
%     if strcmp(ModelParams.errortype,'Diffusion')
%         L = (1+ModelParams.const)*k.^2 - k.^4; % Fourier multipliers
%     else
%         L = k.^2 - k.^4; % Fourier multipliers
%     end
%     
%     ModelParams.E = exp(ModelParams.tau*L); ModelParams.E2 = exp(ModelParams.tau*L/2);
%     M = 16; % no. of points for complex means
%     r = exp(1i*pi*((1:M)-.5)/M); % roots of unity
% 
%     LR = ModelParams.tau*L(:,ones(M,1)) + r(ones(ModelParams.N,1),:);
% 
% 
%     ModelParams.Q = ModelParams.tau*real(mean( (exp(LR/2)-1)./LR ,2));
%     ModelParams.f1 = ModelParams.tau*real(mean( (-4-LR+exp(LR).*(4-3*LR+LR.^2))./LR.^3 ,2));
%     ModelParams.f2 = ModelParams.tau*real(mean( (2+LR+exp(LR).*(-2+LR))./LR.^3 ,2));
%     ModelParams.f3 = ModelParams.tau*real(mean( (-4-3*LR-LR.^2+exp(LR).*(4-LR))./LR.^3 ,2));
%     % Main time-stepping loop:
%     if strcmp(ModelParams.errortype, 'Advection')
%         ModelParams.g = -0.5i*k*(1+ModelParams.const);
%     else
%         ModelParams.g = -0.5i*k;
%     end
%     
    for i=1:resparams.predictions
        xinit = savepred{i}(:,1).*datavar+datamean;
        data1 = rk_solve_m2( xinit, ModelParams, s_mat_k, Z2Xmat);
        data1 = data1(:,2:end);

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
        
        hybrid_err{i} = sqrt(mean((savepred{i}'-...
            tm.test_input_sequence(start_iter(i)+...
            resparams.sync_length+1:start_iter(i)+resparams.sync_length...
            +resparams.predict_length,:)).^2,2)');
        for j=1:length(hybrid_err{i})
            if hybrid_err{i}(j) > error_cutoff
                pred_length(i) = j-1;
                break
            elseif j == length(hybrid_err{i}) && hybrid_err{i}(j) <= error_cutoff
                pred_length(i) = j;
            end
        end
        hybrid_err_sum = hybrid_err_sum + sum(hybrid_err{i})/resparams.predictions;
        hybrid_average_diff = hybrid_average_diff + hybrid_err{i};
        iter = iter+1;
        
    end
    hybrid_average_diff = hybrid_average_diff/iter;
    hybrid_average_diff_tot{jid} = hybrid_average_diff;
    model_average_diff_tot{jid} = model_average_diff;
    pred_length(isnan(pred_length)) = zeros(1,sum(isnan(pred_length)));
    avg_pred_length = mean(pred_length);
    std_pred_length = std(pred_length);
    model_avg_pred_length = mean(model_pred_length);
    model_std_pred_length = std(model_pred_length);
    average_pred_length(jid) = avg_pred_length;
    if strcmp(typeflag{1}, 'hybrid')
        filename_rms = [ filelocation, 'fixed_hybrid', '-pool', num2str(pool_size), ...
                'res', num2str(approx_reservoir_size), ...
                'locality',num2str(locality{1}),'trainlen',num2str(train_lengthin),'sigma',...
                strrep(num2str(resparams.sigma_data),'.',''),'radius',strrep(num2str(resparams.radius),'.',''),'betares',strrep(num2str(beta_reservoir),'.',''),...
                'jobid', num2str(jid),'runiter',num2str(runiterin),'_wnoise',strrep(num2str(resnoise{1}),'.',''),'Nskip',num2str(Nskipin),'_rms_longpred.mat'];
    elseif strcmp(typeflag{1}, 'reservoir')
        filename_rms = [ filelocation, 'fixed_reservoir', '-pool', num2str(pool_size), ...
                'res', num2str(approx_reservoir_size), ...
                'locality',num2str(locality{1}),'trainlen',num2str(train_lengthin),'sigma',...
                strrep(num2str(resparams.sigma_data),'.',''),'radius',strrep(num2str(resparams.radius),'.',''),'betares',strrep(num2str(beta_reservoir),'.',''),...
                'jobid', num2str(jid),'runiter',num2str(runiterin),'_wnoise',strrep(num2str(resnoise{1}),'.',''),'Nskip',num2str(Nskipin),'_rms_longpred.mat'];
    end
    if ifsaverms
        save(filename_rms, 'hybrid_err','hybrid_average_diff_tot','model_err','model_average_diff_tot','iter',...
            'avg_pred_length','std_pred_length','pred_length','model_avg_pred_length','model_std_pred_length','model_pred_length')
    end
end