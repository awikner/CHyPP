function generate_KS_data(ModelParams,discard_length)

% ModelParams.d = 100;
% ModelParams.tau = 0.25;
% ModelParams.N = 128;
% ModelParams.const = 0;
% ModelParams.modeltype = 'ETDRK';
% 
% % Precompute various ETDRK4 scalar quantities:
% k = [0:ModelParams.N/2-1 0 -ModelParams.N/2+1:-1]'*(2*pi/ModelParams.d); % wave numbers
% L = (1+ModelParams.const)*k.^2 - k.^4; % Fourier multipliers
% ModelParams.E = exp(ModelParams.tau*L); ModelParams.E2 = exp(ModelParams.tau*L/2);
% M = 16; % no. of points for complex means
% r = exp(1i*pi*((1:M)-.5)/M); % roots of unity
% 
% LR = ModelParams.tau*L(:,ones(M,1)) + r(ones(ModelParams.N,1),:);
% 
% 
% ModelParams.Q = ModelParams.tau*real(mean( (exp(LR/2)-1)./LR ,2));
% ModelParams.f1 = ModelParams.tau*real(mean( (-4-LR+exp(LR).*(4-3*LR+LR.^2))./LR.^3 ,2));
% ModelParams.f2 = ModelParams.tau*real(mean( (2+LR+exp(LR).*(-2+LR))./LR.^3 ,2));
% ModelParams.f3 = ModelParams.tau*real(mean( (-4-3*LR-LR.^2+exp(LR).*(4-LR))./LR.^3 ,2));
% % Main time-stepping loop:
% 
% ModelParams.g = -0.5i*k;

dataseed = 5;
rng(dataseed);
nsteps = 100000;
nstepstest = 30000;
x = 0.6*(-1+2*rand(ModelParams.N,1));

transient = 1000;

for i = 1:transient
    x = ModelParams.predict(x,ModelParams);
end

data = x;
train_input_sequence = zeros(nsteps,ModelParams.N);
train_input_sequence(1,:) = data';
for i = 1:nsteps-1
    data = ModelParams.predict(data,ModelParams);
    train_input_sequence(i+1,:) = data';
end

data = ModelParams.predict(data,ModelParams);
test_input_sequence = zeros(nstepstest,ModelParams.N);
test_input_sequence(1,:) = data';
for i = 1:nstepstest-1
    data = ModelParams.predict(data,ModelParams);
    test_input_sequence(i+1,:) = data';
end

datamean = mean(train_input_sequence(:));

datavar = std(train_input_sequence(:));

train_input_sequence = train_input_sequence - datamean;

train_input_sequence = train_input_sequence./datavar;

test_input_sequence = test_input_sequence - datamean;

test_input_sequence= test_input_sequence./datavar;

iterseed = 10;
rng(iterseed);

start_iter = randi([1,size(test_input_sequence,1)-discard_length - 1],1,1000);

noise_seed = 20;
rng(noise_seed);
noise = randn(size(train_input_sequence,1),size(train_input_sequence,2));
mkdir('KS_Data')
addpath KS_Data
save('KS_Data/KS_train_input_sequence.mat', 'train_input_sequence', 'noise','datamean', 'datavar', '-v7.3')
save('KS_Data/KS_test_input_sequence.mat', 'test_input_sequence', 'datamean', 'datavar', '-v7.3')
save('KS_Data/KS_pred_start_indices.mat','start_iter','-v7.3')