clear;
addpath helpers
% Set parameters of the Lorenz Model 3 evolution (from 2005 paper)
ModelParams.N = 960; % Number of grid points
ModelParams.K = ModelParams.N/30; % Large wavelength correlation distance
ModelParams.F = 15; % Forcing
ModelParams.b = 10; %Time scale & amplitude separation
ModelParams.c = 2.5; 
ModelParams.I = 12; % Smoothing length
ModelParams.tau = 0.005; % Time step
ModelParams.alpha = (3*(ModelParams.I)^2 + 3)/(2*(ModelParams.I^3) + 4*ModelParams.I); % Parameters using in smoothing
ModelParams.beta = (2*(ModelParams.I)^2 + 1)/((ModelParams.I^4) + 2*(ModelParams.I)^2);
ModelParams.Z2Xmat = sparse(Z2X(ModelParams)); % Matrix for determining large wavelength dynamics
ModelParams.s_mat_k = sparse(getsmat(ModelParams.N, ModelParams.K)); % Correlation matrix for some N and K
ModelParams.predict = @rk4Model3;
ModelParams.prediction = @rksolveModel3;
%%
% Generate KS Equation training data and set transient discard length
Nskip = 1;
PredictLength = 2500;
generate_LorenzModel3_data(ModelParams,Nskip,PredictLength);

%%
% Set model error to a nonzero value and recompute parameters
ModelParams.N = ModelParams.N/Nskip;
ModelParams.K = ModelParams.N/30;
ModelParams.s_mat_k = sparse(getsmat(ModelParams.N,ModelParams.K));
ModelParams.predict = @rk4Model2;
ModelParams.prediction = @rksolveModel2;

PoolSize = 6;
NumRes = 24;
TrainLength = 80000;
ReservoirSize = 1000;
AvgDegree = 3;
LocalOverlap = 80;
InputWeight = 1;
SpectralRadius = 0.6;
RidgeReg = 1e-4;
Predictions = 15;
TrainSteps = 100;
ResOnly = false;
Noise = 0.085;
RunIter = 1;
OutputData = true;
TrainFile = 'LorenzModel3_Data/M3_train_input_sequence.mat';
TestFile = 'LorenzModel3_Data/M3_test_input_sequence.mat';
StartFile = 'LorenzModel3_Data/M3_pred_start_indices.mat';
OutputLocation = 'LorenzModel3_Data';
ErrorCutoff = 0.85;
%% Run the CHyPP prediction using the above parameters
[avg_pred_length_CHyPP,CHyPP_pred_file,CHyPP_rms_file] = CHyPP('PoolSize',PoolSize,...
    'NumRes',NumRes,'TrainLength',TrainLength,...
    'ReservoirSize',ReservoirSize,'AvgDegree',AvgDegree,'LocalOverlap',LocalOverlap,...
    'InputWeight',InputWeight,'SpectralRadius',SpectralRadius,'RidgeReg',RidgeReg,...
    'Predictions',Predictions,'PredictLength',PredictLength,'TrainSteps',TrainSteps,...
    'ResOnly',ResOnly,'Noise',Noise,'RunIter',RunIter,'OutputData',OutputData,...
    'ModelParams',ModelParams,'TrainFile',TrainFile,'TestFile',TestFile,...
    'StartFile',StartFile,'OutputLocation',OutputLocation,'ErrorCutoff',ErrorCutoff);
% [avg_pred_length_CHyPP,CHyPP_pred_file,CHyPP_rms_file] = CHyPP_serial(...
%     'NumRes',NumRes,'TrainLength',TrainLength,...
%     'ReservoirSize',ReservoirSize,'AvgDegree',AvgDegree,'LocalOverlap',LocalOverlap,...
%     'InputWeight',InputWeight,'SpectralRadius',SpectralRadius,'RidgeReg',RidgeReg,...
%     'Predictions',Predictions,'PredictLength',PredictLength,'TrainSteps',TrainSteps,...
%     'ResOnly',ResOnly,'Noise',Noise,'RunIter',RunIter,'OutputData',OutputData,...
%     'ModelParams',ModelParams,'TrainFile',TrainFile,'TestFile',TestFile,...
%     'StartFile',StartFile,'OutputLocation',OutputLocation,'ErrorCutoff',ErrorCutoff);

%% Run a reservoirs-only prediction using the same number of reservoirs
ResOnly = true;
Noise = 0.1;
[avg_pred_length_res,res_pred_file,res_rms_file] = CHyPP('PoolSize',PoolSize,...
    'NumRes',NumRes,'TrainLength',TrainLength,...
    'ReservoirSize',ReservoirSize,'AvgDegree',AvgDegree,'LocalOverlap',LocalOverlap,...
    'InputWeight',InputWeight,'SpectralRadius',SpectralRadius,'RidgeReg',RidgeReg,...
    'Predictions',Predictions,'PredictLength',PredictLength,'TrainSteps',TrainSteps,...
    'ResOnly',ResOnly,'Noise',Noise,'RunIter',RunIter,'OutputData',OutputData,...
    'ModelParams',ModelParams,'TrainFile',TrainFile,'TestFile',TestFile,...
    'StartFile',StartFile,'OutputLocation',OutputLocation,'ErrorCutoff',ErrorCutoff);
% [avg_pred_length_res,res_pred_file,res_rms_file] = CHyPP(...
%     'NumRes',NumRes,'TrainLength',TrainLength,...
%     'ReservoirSize',ReservoirSize,'AvgDegree',AvgDegree,'LocalOverlap',LocalOverlap,...
%     'InputWeight',InputWeight,'SpectralRadius',SpectralRadius,'RidgeReg',RidgeReg,...
%     'Predictions',Predictions,'PredictLength',PredictLength,'TrainSteps',TrainSteps,...
%     'ResOnly',ResOnly,'Noise',Noise,'RunIter',RunIter,'OutputData',OutputData,...
%     'ModelParams',ModelParams,'TrainFile',TrainFile,'TestFile',TestFile,...
%     'StartFile',StartFile,'OutputLocation',OutputLocation,'ErrorCutoff',ErrorCutoff);

%% Plot the resulting predictions
load(res_pred_file)
load(TestFile)
index = 3;
ModelParams.nstep = PredictLength + 1;
xinit = test_input_sequence(start_iter(index)+resparams.sync_length,:)'.*datavar+datamean;
data = ModelParams.prediction(xinit,ModelParams);
data = (data(:,2:end)-datamean)./datavar;

t = (1:PredictLength)*ModelParams.tau;
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