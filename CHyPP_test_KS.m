clear;
addpath helpers
% Set parameters of KS equation
ModelParams.d = 100; % Periodicity length L
ModelParams.tau = 0.25; % Time step \Delta t
ModelParams.N = 128; % Number of grid points
ModelParams.const = 0; % Error value \epsilon
ModelParams.modeltype = 'ETDRK'; %Type of integration to use
% Precompute ETDRK parameters
ModelParams = precompute_KS(ModelParams);
% Set model evolution functions
ModelParams.predict = @kursiv_forecast;
ModelParams.prediction = @kursiv_model_predict;
%%
% Generate KS Equation training data and set transient discard length
PredictLength = 1500;
generate_KS_data(ModelParams,PredictLength);

%%
% Set model error to a nonzero value and recompute parameters
ModelParams.const = 0.1;
ModelParams = precompute_KS(ModelParams);

PoolSize = 4;
NumRes = 16;
TrainLength = 50000;
ReservoirSize = 2000;
AvgDegree = 3;
LocalOverlap = 6;
InputWeight = 0.1;
SpectralRadius = 0.6;
RidgeReg = 1e-6;
Predictions = 15;
TrainSteps = 50;
ResOnly = false;
Noise = 0;
RunIter = 1;
OutputData = true;
%%
[~,avg_pred_length_CHyPP,CHyPP_pred_file,CHyPP_rms_file] = CHyPP_multires('PoolSize',PoolSize,...
    'NumRes',NumRes,'TrainLength',TrainLength,...
    'ReservoirSize',ReservoirSize,'AvgDegree',AvgDegree,'LocalOverlap',LocalOverlap,...
    'InputWeight',InputWeight,'SpectralRadius',SpectralRadius,'RidgeReg',RidgeReg,...
    'Predictions',Predictions,'PredictLength',PredictLength,'TrainSteps',TrainSteps,...
    'ResOnly',ResOnly,'Noise',Noise,'RunIter',RunIter,'OutputData',OutputData,...
    'ModelParams',ModelParams);

ResOnly = true;
Noise = 1e-3;
[~,avg_pred_length_CHyPP,CHyPP_pred_file,CHyPP_rms_file] = CHyPP_multires('PoolSize',PoolSize,...
    'NumRes',NumRes,'TrainLength',TrainLength,...
    'ReservoirSize',ReservoirSize,'AvgDegree',AvgDegree,'LocalOverlap',LocalOverlap,...
    'InputWeight',InputWeight,'SpectralRadius',SpectralRadius,'RidgeReg',RidgeReg,...
    'Predictions',Predictions,'PredictLength',PredictLength,'TrainSteps',TrainSteps,...
    'ResOnly',ResOnly,'Noise',Noise,'RunIter',RunIter,'OutputData',OutputData,...
    'ModelParams',ModelParams);
%% Run the CHyPP prediction using the above parameters
[~,avg_pred_length_CHyPP,CHyPP_pred_file,CHyPP_rms_file] = CHyPP('PoolSize',PoolSize,'TrainLength',TrainLength,...
    'ReservoirSize',ReservoirSize,'AvgDegree',AvgDegree,'LocalOverlap',LocalOverlap,...
    'InputWeight',InputWeight,'SpectralRadius',SpectralRadius,'RidgeReg',RidgeReg,...
    'Predictions',Predictions,'PredictLength',PredictLength,'TrainSteps',TrainSteps,...
    'ResOnly',ResOnly,'Noise',Noise,'RunIter',RunIter,'OutputData',OutputData,...
    'ModelParams',ModelParams);

%% Run a reservoirs-only prediction using the same parameters except for noise (s)

ResOnly = true;
Noise = 1e-3;
[~,avg_pred_length_resonly,res_pred_file,res_rms_file] = CHyPP('PoolSize',PoolSize,'TrainLength',TrainLength,...
    'ReservoirSize',ReservoirSize,'AvgDegree',AvgDegree,'LocalOverlap',LocalOverlap,...
    'InputWeight',InputWeight,'SpectralRadius',SpectralRadius,'RidgeReg',RidgeReg,...
    'Predictions',Predictions,'PredictLength',PredictLength,'TrainSteps',TrainSteps,...
    'ResOnly',ResOnly,'Noise',Noise,'RunIter',RunIter,'OutputData',OutputData,...
    'ModelParams',ModelParams);
%% Plot the resulting predictions
load(res_pred_file)
load('KS_data/KS_test_input_sequence.mat')
index = 5;
ModelParams.nstep = PredictLength + 1;
xinit = test_input_sequence(start_iter(index)+resparams.sync_length,:)'.*datavar+datamean;
data = ModelParams.prediction(xinit,ModelParams);
data = (data(:,2:end)-datamean)./datavar;

t = (1:PredictLength)*0.25*0.09;
x = 1:ModelParams.N;
fig1 = figure('Renderer', 'painters', 'Position', [10 10 1400 800]);
subplot(4,2,1)
imagesc(t,x,test_input_sequence(start_iter(index)+resparams.sync_length+1:...
    start_iter(index)+resparams.sync_length+resparams.predict_length,:)')
yticks([1,ModelParams.N/2,ModelParams.N])
yticklabels({'0','L/2','L'})
title('True Dynamics')
caxis([-3,3])
colorbar

subplot(4,2,3)
imagesc(t,x,data)
yticks([1,ModelParams.N/2,ModelParams.N])
yticklabels({'0','L/2','L'})
title('Imperfect Model Prediction')
caxis([-3,3])
colorbar

subplot(4,2,4)
imagesc(t,x,test_input_sequence(start_iter(index)+resparams.sync_length+1:...
    start_iter(index)+resparams.sync_length+resparams.predict_length,:)'-data)
yticks([1,ModelParams.N/2,ModelParams.N])
yticklabels({'0','L/2','L'})
title('Imperfect Model Error')
caxis([-3,3])
colorbar

subplot(4,2,5)
imagesc(t,x,savepred{index})
yticks([1,ModelParams.N/2,ModelParams.N])
yticklabels({'0','L/2','L'})
title('Parallel Reservoir Prediction')
caxis([-3,3])
colorbar

subplot(4,2,6)
imagesc(t,x,test_input_sequence(start_iter(index)+resparams.sync_length+1:...
    start_iter(index)+resparams.sync_length+resparams.predict_length,:)'-savepred{index})
yticks([1,ModelParams.N/2,ModelParams.N])
yticklabels({'0','L/2','L'})
title('Parallel Reservoir Error')
caxis([-3,3])
colorbar

load(CHyPP_pred_file)

subplot(4,2,7)
imagesc(t,x,savepred{index})
yticks([1,ModelParams.N/2,ModelParams.N])
yticklabels({'0','L/2','L'})
xlabel('Lyapunov Time')
title('CHyPP Prediction')
caxis([-3,3])
colorbar

subplot(4,2,8)
imagesc(t,x,test_input_sequence(start_iter(index)+resparams.sync_length+1:...
    start_iter(index)+resparams.sync_length+resparams.predict_length,:)'-savepred{index})
yticks([1,ModelParams.N/2,ModelParams.N])
yticklabels({'0','L/2','L'})
xlabel('Lyapunov Time')
title('CHyPP Error')
caxis([-3,3])
colorbar