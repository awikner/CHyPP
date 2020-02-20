clear;

ModelParams.N = 960;
ModelParams.K = ModelParams.N/30;
ModelParams.F = 15;
ModelParams.b = 10;
ModelParams.c = 2.5;
ModelParams.I = 12;
ModelParams.tau = 0.005;
ModelParams.noise = 0.15;
seed = 5;
ModelParams.alpha = (3*(ModelParams.I)^2 + 3)/(2*(ModelParams.I^3) + 4*ModelParams.I);
ModelParams.beta = (2*(ModelParams.I)^2 + 1)/((ModelParams.I^4) + 2*(ModelParams.I)^2);
jobid = 1;
rng(seed);
Z = randn(ModelParams.N, 1);
ModelParams.nstep = 2000;
Z2Xmat = sparse(Z2X(ModelParams));


s_mat_k = sparse(getsmat(ModelParams.N, ModelParams.K));

[data, ~] = rk_solve(Z, ModelParams, s_mat_k, Z2Xmat);
Z = data(:,end);
ModelParams.nstep = 400000;
[data,~] = rk_solve(Z, ModelParams, s_mat_k, Z2Xmat);

test_input_sequence = transpose(data(:, 300001:end));
train_input_sequence = transpose(data(:,1:300000));

datamean = mean(train_input_sequence(:));

datavar = std(train_input_sequence(:));

train_input_sequence = train_input_sequence - datamean;

train_input_sequence = train_input_sequence./datavar;

test_input_sequence = test_input_sequence - datamean;

test_input_sequence= test_input_sequence./datavar;
train_input_sequence_nonoise = train_input_sequence;
train_input_sequence = train_input_sequence + ...
    ModelParams.noise*randn(size(train_input_sequence,1),size(train_input_sequence,2));
rng(seed)
start_iter = randi([1,size(test_input_sequence,1)-2001],1,100);

mkdir(['Data/N960K32I12F15noise',strrep(num2str(ModelParams.noise),'.',''),'_',num2str(jobid)])

save(['Data/N960K32I12F15noise',strrep(num2str(ModelParams.noise),'.',''),'_',num2str(jobid),'/train_input_sequence.mat'], 'train_input_sequence', 'train_input_sequence_nonoise','datamean', 'datavar', '-v7.3')
save(['Data/N960K32I12F15noise',strrep(num2str(ModelParams.noise),'.',''),'_',num2str(jobid),'/test_input_sequence.mat'], 'test_input_sequence', 'datamean', 'datavar', '-v7.3')
save(['Data/N960K32I12F15noise',strrep(num2str(ModelParams.noise),'.',''),'_',num2str(jobid),'/pred_start_indices_200000.mat'],'start_iter','-v7.3')