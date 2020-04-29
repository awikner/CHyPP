%function ks_enkf_test(rho,epsilon, mstep, res_size, sigma, ens_size, gridstep, variation)
clear;
epsilon = 0.1;
mstep = 1;
%res_size = 1000;
sigma = 0.1;
variation = 'grid1';
gridstep = 1;
num_trials = 1;

runflag = 'local';

pred_steps = 1000;
rhos = [3.5,4.0,4.5];

num_rhos = length(rhos);

model_data = ErrorData;
model_data.cov_rhos = rhos;
model_data.vtimes = zeros(length(rhos), num_trials);
model_data.pred_cell = cell(length(rhos),1);
model_data.truth_cell = cell(length(rhos),1);
model_data.error_cell = cell(length(rhos),1);


enses = [70];
tic
for ens_iter = 1:length(enses)
    
ens_size = enses(ens_iter);


    for rho_iter = 1:num_rhos

        rho = rhos(rho_iter);
        model_error_array = zeros(num_trials, pred_steps);
        kalman_error_array = zeros(num_trials, pred_steps);

        for trial_iter = 1:num_trials

            model_error_array = zeros(num_trials, pred_steps);
            kalman_error_array = zeros(num_trials, pred_steps);

            rng('shuffle')
            rng(5)
            ModelParams.N = 128;
            ModelParams.d = 100;
            ModelParams.tau = 0.25;
            ModelParams.nstep = 20000;
            ModelParams.const = 0.1;
% 
%             init = 0.6*(-1+2*rand(ModelParams.N, 1));

%             data = transpose(solve_ks(init, ModelParams));
            
            file = matfile('KS_Data/KS_train_input_sequence.mat');
            data = file.datavar*file.train_input_sequence' + file.datamean;

            measured_vars = gridstep:gridstep:ModelParams.N;

            [m,len] = size(data);
            num_measured = length(measured_vars);
            mu = zeros(num_measured,1);
            R = sigma^2*eye(num_measured);
            R_inverse = (1/sigma^2)*eye(num_measured);

            mtimes = mstep:mstep:len;

            truth = data(:, mtimes);

            Imat = eye(m);

            M_operator = Imat(measured_vars,:);

            z = transpose(mvnrnd(mu,R,length(truth)));
%             z = sigma^2*randn(size(truth,1),size(truth,2));
            measurements = M_operator*truth + z;

            x = 0.1*(-1 + 2*rand(m,1));
            ensemble = bsxfun(@plus, x, 0.1*(randn(m, ens_size))); %initialize ensemble members 

            num_steps = 130000;

            analysis = zeros(m, num_steps);
            compare_truth = zeros(m, num_steps);

            ModelParams.const = epsilon;

            KS = precompute_ks_params(ModelParams);

            illConditionFlag = 0;

            analysis_train = zeros(m, num_steps);

            for i = 1:num_steps

                y_ensemble = M_operator*ensemble;

                y_avg = mean(y_ensemble, 2);

                Y = bsxfun(@minus, y_ensemble, y_avg);

                x_avg = mean(ensemble, 2);              % STEP 2

                X = bsxfun(@minus, ensemble, x_avg); 

                % SKIP LOCALIZATION STEP 3

                C = transpose(Y)*R_inverse; % STEP 4

                temp_mat = (ens_size-1)*eye(ens_size)./rho + C*Y;

                condition_num = cond(temp_mat);       

                if condition_num > 1e02
                   illConditionFlag = 1;
                   break;
                end

                P_tilde = inv(temp_mat); %STEP 5

                W = sqrtm((ens_size-1)*P_tilde); % STEP 6

                w = P_tilde*C*(measurements(:, i) - y_avg);

                w_a = bsxfun(@plus, w, W);

                for j = 1:ens_size
                    ensemble(:,j) = X*w_a(:,j) + x_avg;
                end

                analysis(:,i) = mean(ensemble, 2);

                compare_truth(:,i) = truth(:, i);

                for j = 1:ens_size
                    ensemble(:,j) = forecast_ks(ensemble(:,j), KS, mstep);
                end

                x_analysis = mean(ensemble, 2);

                analysis_train(:, i) = x_analysis;
                
                
            end
            train_input_sequence = measurements(:,1:100000)';
            da_train_input_sequence = analysis(:,1:100000)';
            test_input_sequence = measurements(:,100001:end)';
            da_test_input_sequence = analysis(:,100001:end)';
            
            datamean = mean(train_input_sequence(:));
            da_datamean = mean(da_train_input_sequence(:));
            datavar = std(train_input_sequence(:));
            da_datavar = std(da_train_input_sequence(:));
            
            train_input_sequence = train_input_sequence - datamean;

            train_input_sequence = train_input_sequence./datavar;

            test_input_sequence = test_input_sequence - datamean;

            test_input_sequence = test_input_sequence./datavar;
            
            da_train_input_sequence = da_train_input_sequence - da_datamean;

            da_train_input_sequence = da_train_input_sequence./da_datavar;

            da_test_input_sequence = da_test_input_sequence - da_datamean;

            da_test_input_sequence = da_test_input_sequence./da_datavar;
            
            noise = randn(size(train_input_sequence,1),size(train_input_sequence,2));
            
            save(['KS_Data/KS_train_input_sequence_wDAepsilon',strrep(num2str(epsilon),'.',''),...
                'rho',num2str(rhos(rho_iter)),...
                'noise',num2str(sigma^2),'.mat'],'train_input_sequence', ...
                'da_train_input_sequence','noise','datamean','da_datamean', ...
                'datavar','da_datavar', '-v7.3')
            
            save(['KS_Data/KS_test_input_sequence_wDAepsilon',strrep(num2str(epsilon),'.',''),...
                'rho',num2str(rhos(rho_iter)),...
                'noise',num2str(sigma^2),'.mat'],'test_input_sequence', ...
                'da_test_input_sequence','datamean','da_datamean', ...
                'datavar','da_datavar', '-v7.3')
                
            count = 1;

            modelprediction = zeros(m, pred_steps);
            modelprediction(:,1) = x_analysis;

            for i = 1:pred_steps
                modelprediction(:,i+1) = forecast_ks(modelprediction(:,i), KS, mstep);
            end


            error_model = sqrt(mean((modelprediction(:,2:end) - truth(:, num_steps+2:num_steps+pred_steps+1)).^2,1));
            threshold = 1;
            model_vt = find(error_model > threshold, 1);
            model_data.vtimes(rho_iter, trial_iter) = model_vt;
            model_error_array(trial_iter, :) = error_model;
            disp(trial_iter)
            toc
        end

        model_data.error_cell{rho_iter} = model_error_array;

        if strcmp(runflag, 'cluster')
            basefolder = '/lustre/jpathak/hybridKalmanExpts/';
        elseif strcmp(runflag, 'local')
            basefolder = '';
        end

        filename = makefilename(basefolder, variation, 'NumEns', ens_size, 'ModelError', epsilon, ...
            'noise', sigma);

        saveModel = saveobj(model_data);

        save(filename, 'saveModel', 'sigma', 'epsilon', 'ens_size');

    end

end
%%
subplot(3,1,1)
imagesc(modelprediction(:,2:end) - truth(:, num_steps+2:num_steps+pred_steps+1))
subplot(3,1,2)
imagesc(modelprediction(:, 2:end))
subplot(3,1,3)
imagesc(truth(:, num_steps+2:num_steps+pred_steps+1))