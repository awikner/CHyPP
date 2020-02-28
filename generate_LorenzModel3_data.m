function generate_LorenzModel3_data(ModelParams,Nskip,predict_length)

seed = 5;
rng(seed,'twister');
Z = randn(ModelParams.N, 1);

ModelParams.nstep = 1000;

data = ModelParams.prediction(Z, ModelParams);
Z = data(:,end);
ModelParams.nstep = 130000;
data = ModelParams.prediction(Z, ModelParams);

test_input_sequence = transpose(data(:, 100001:end));
train_input_sequence = transpose(data(:,1:100000));

test_input_sequence = test_input_sequence(Nskip:Nskip:end,:);
train_input_sequence = train_input_sequence(Nskip:Nskip:end,:);

datamean = mean(train_input_sequence(:));

datavar = std(train_input_sequence(:));

train_input_sequence = train_input_sequence - datamean;

train_input_sequence = train_input_sequence./datavar;

test_input_sequence = test_input_sequence - datamean;

test_input_sequence= test_input_sequence./datavar;

noise = randn(size(train_input_sequence,1),size(train_input_sequence,2));

rng(seed,'twister')
start_iter = randi([1,size(test_input_sequence,1)-predict_length - 1],1,100);

mkdir('LorenzModel3_Data')

save('LorenzModel3_Data/M3_train_input_sequence.mat', 'train_input_sequence', 'noise','datamean', 'datavar', '-v7.3')
save('LorenzModel3_Data/M3_test_input_sequence.mat', 'test_input_sequence', 'datamean', 'datavar', '-v7.3')
save('LorenzModel3_Data/M3_pred_start_indices.mat','start_iter','-v7.3')