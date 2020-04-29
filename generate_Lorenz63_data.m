function generate_Lorenz63_data(ModelParams,predict_length)

dataseed = 5;
rng(dataseed);
sync_length = 100;
x = rand(3,1);
transient_length = 10000;
rk4L63 = @(x,ModelParams) rk4(@lorenz63,x,ModelParams);
for i = 1:transient_length
    x = rk4L63(x,ModelParams);
end

nsteps = 85000;
train_length = 25000;
data = zeros(3,nsteps);
data(:,1) = x;
for i = 1:nsteps-1
    data(:,i+1) = rk4L63(data(:,i),ModelParams);
end

train_input_sequence = data(:,1:train_length)';
test_input_sequence = data(:,train_length+1:end)';

datamean = mean(train_input_sequence);
datavar = std(train_input_sequence);

train_input_sequence = train_input_sequence - datamean;
train_input_sequence = (train_input_sequence./datavar);

test_input_sequence = test_input_sequence - datamean;
test_input_sequence = (test_input_sequence./datavar);

datamean = datamean';
datavar = datavar';

noise_seed = 20;
rng(noise_seed)
noise = randn(train_length,3);

start_iter = randi([1, nsteps - train_length - sync_length - predict_length - 1], 1,1000);

% Save all data
% filepath = ['/lustre/awikner1/Lorenz63/tau',strrep(num2str(ModelParams.tau),'.','')];
% mkdir(filepath)
% save([filepath,'/L63_train_input_sequence.mat'], 'train_input_sequence', 'noise','datamean', 'datavar', '-v7.3')
% save([filepath,'/L63_test_input_sequence.mat'], 'test_input_sequence', 'datamean', 'datavar', '-v7.3')
% save([filepath,'/L63_pred_start_indices.mat'],'start_iter','-v7.3')
mkdir('Lorenz63_Data')
filepath = ['Lorenz63_Data/tau',strrep(num2str(ModelParams.tau),'.','')];
mkdir(filepath)
save([filepath,'/L63_train_input_sequence.mat'], 'train_input_sequence', 'noise','datamean', 'datavar', '-v7.3')
save([filepath,'/L63_test_input_sequence.mat'], 'test_input_sequence', 'datamean', 'datavar', '-v7.3')
save([filepath,'/L63_pred_start_indices.mat'],'start_iter','-v7.3')


