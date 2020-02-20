clear;
addpath helpers
% Set parameters of KS equation
ModelParams.d = 100; % Periodicity length L
ModelParams.tau = 0.25; % Time step \Delta t
ModelParams.N = 128; % Number of grid points
ModelParams.const = 0; % Error value \epsilon
ModelParams.modeltype = 'ETDRK'; %Type of integration to use

% Precompute various ETDRK4 scalar quantities:
k = [0:ModelParams.N/2-1 0 -ModelParams.N/2+1:-1]'*(2*pi/ModelParams.d); % wave numbers
L = (1+ModelParams.const)*k.^2 - k.^4; % Fourier multipliers
ModelParams.E = exp(ModelParams.tau*L); ModelParams.E2 = exp(ModelParams.tau*L/2);
M = 16; % no. of points for complex means
r = exp(1i*pi*((1:M)-.5)/M); % roots of unity
LR = ModelParams.tau*L(:,ones(M,1)) + r(ones(ModelParams.N,1),:);
ModelParams.Q = ModelParams.tau*real(mean( (exp(LR/2)-1)./LR ,2));
ModelParams.f1 = ModelParams.tau*real(mean( (-4-LR+exp(LR).*(4-3*LR+LR.^2))./LR.^3 ,2));
ModelParams.f2 = ModelParams.tau*real(mean( (2+LR+exp(LR).*(-2+LR))./LR.^3 ,2));
ModelParams.f3 = ModelParams.tau*real(mean( (-4-3*LR-LR.^2+exp(LR).*(4-LR))./LR.^3 ,2));
ModelParams.g = -0.5i*k;
% Set model evolution functions
ModelParams.predict = @kursiv_forecast;
ModelParams.prediction = @kursiv_model_predict;
%%
% Generate KS Equation training data and set transient discard length
discard_length = 1000;
generate_KS_data(ModelParams,discard_length);

%%
% Set model error to a nonzero value and recompute parameters
ModelParams.const = 0.1;
% Precompute various ETDRK4 scalar quantities:
k = [0:ModelParams.N/2-1 0 -ModelParams.N/2+1:-1]'*(2*pi/ModelParams.d); % wave numbers
L = (1+ModelParams.const)*k.^2 - k.^4; % Fourier multipliers
ModelParams.E = exp(ModelParams.tau*L); ModelParams.E2 = exp(ModelParams.tau*L/2);
M = 16; % no. of points for complex means
r = exp(1i*pi*((1:M)-.5)/M); % roots of unity
LR = ModelParams.tau*L(:,ones(M,1)) + r(ones(ModelParams.N,1),:);
ModelParams.Q = ModelParams.tau*real(mean( (exp(LR/2)-1)./LR ,2));
ModelParams.f1 = ModelParams.tau*real(mean( (-4-LR+exp(LR).*(4-3*LR+LR.^2))./LR.^3 ,2));
ModelParams.f2 = ModelParams.tau*real(mean( (2+LR+exp(LR).*(-2+LR))./LR.^3 ,2));
ModelParams.f3 = ModelParams.tau*real(mean( (-4-3*LR-LR.^2+exp(LR).*(4-LR))./LR.^3 ,2));
ModelParams.g = -0.5i*k;

PoolSize = 4;
TrainLength = 50000;
ReservoirSize = 2000;
AvgDegree = 3;
LocalOverlap = 6;
InputWeight = 0.1;
SpectralRadius = 0.6;
RidgeReg = 1e-6;
ResOnly = false;
Noise = 0;
RunIter = 1;
OutputData = true;

[~,avg_pred_length] = CHyPP('PoolSize',PoolSize,'TrainLength',TrainLength,...
    'ReservoirSize',ReservoirSize,'AvgDegree',AvgDegree,'LocalOverlap',LocalOverlap,...
    'InputWeight',InputWeight,'SpectralRadius',SpectralRadius,'RidgeReg',RidgeReg,...
    'ResOnly',ResOnly,'Noise',Noise,'RunIter',RunIter,'OutputData',OutputData,...
    'ModelParams',ModelParams);