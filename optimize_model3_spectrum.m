function loss = optimize_model3_spectrum(sigma_resin, radiusin, resnoisein,num_reservoirs)

% num_reservoirs = 5;
num_predictions = 5;

% Generate KS Equation training data and set transient discard length
Nskip = 1;
PredictLength = 110000;

% Set parameters of the Lorenz Model 3 evolution (from 2005 paper)

ModelParams.F = 15; % Forcing

ModelParams.tau = 0.005; % Time step

% Set model error to a nonzero value and recompute parameters
ModelParams.N = 960/Nskip;
ModelParams.K = ModelParams.N/30;
ModelParams.s_mat_k = sparse(getsmat(ModelParams.N,ModelParams.K));
ModelParams.predict = @rk4Model2;
ModelParams.prediction = @rksolveModel2;

PoolSize = 12;
NumRes = 24;
TrainLength = 140000;
ReservoirSize = 1000;
AvgDegree = 3;
LocalOverlap = 80;
InputWeight = sigma_resin;
SpectralRadius = radiusin;
RidgeReg = 1e-4;
Predictions = num_predictions;
TrainSteps = 200;
ResOnly = false;
Noise = resnoisein;

OutputData = false;
ErrorCutoff = 0.85;

TrainFile = ['/lustre/awikner1/LorenzModel3/N960K32I12F15wnoiseNskip',num2str(Nskip),'_1/train_input_sequence.mat'];
TestFile = ['/lustre/awikner1/LorenzModel3/N960K32I12F15wnoiseNskip',num2str(Nskip),'_1/test_input_sequence.mat'];
StartFile = ['/lustre/awikner1/LorenzModel3/N960K32I12F15wnoiseNskip',num2str(Nskip),'_1/M3_pred_start_indices_Nskip',num2str(Nskip),'.mat'];
OutputLocation = ['/lustre/awikner1/LorenzModel3/N960K32I12F15wnoiseNskip',num2str(Nskip),'_1'];
tm = matfile(['/lustre/awikner1/LorenzModel3/N960K32I12F15wnoiseNskip',num2str(Nskip),'_1/test_input_sequence.mat']);

% TrainFile = ['LorenzModel3_Data/M3_train_input_sequence_Nskip',num2str(Nskip),'.mat'];
% TestFile = ['LorenzModel3_Data/M3_test_input_sequence_Nskip',num2str(Nskip),'.mat'];
% StartFile = ['LorenzModel3_Data/M3_pred_start_indices_Nskip',num2str(Nskip),'.mat'];
% OutputLocation = 'LorenzModel3_Data';
% tm = matfile(['LorenzModel3_Data/M3_test_input_sequence_Nskip',num2str(Nskip),'.mat']);

Fs = 1/0.005;
N_test = size(tm,'test_input_sequence',1);
[test_psd,f_test] = periodogram(tm.test_input_sequence(:,1),hamming(N_test),[],Fs);
transient_length = 10000;
N_pred = PredictLength-transient_length;
loss = zeros(1,num_reservoirs);
for res = 1:num_reservoirs
    RunIter = res;
    [savepred,~,~] = CHyPP('PoolSize',PoolSize,...
        'NumRes',NumRes,'TrainLength',TrainLength,...
        'ReservoirSize',ReservoirSize,'AvgDegree',AvgDegree,'LocalOverlap',LocalOverlap,...
        'InputWeight',InputWeight,'SpectralRadius',SpectralRadius,'RidgeReg',RidgeReg,...
        'Predictions',Predictions,'PredictLength',PredictLength,'TrainSteps',TrainSteps,...
        'ResOnly',ResOnly,'Noise',Noise,'RunIter',RunIter,'OutputData',OutputData,...
        'ModelParams',ModelParams,'TrainFile',TrainFile,'TestFile',TestFile,...
        'StartFile',StartFile,'OutputLocation',OutputLocation,'ErrorCutoff',ErrorCutoff);
    prediction_pdf = cell(num_reservoirs,1);
    for j=1:num_predictions
        [prediction_pdf{j},freq] = periodogram(savepred{j}(...
            randi([1,ModelParams.N],1,1),transient_length+1:end),hamming(N_pred),[],200);
    end
    [~,min_num_points] = min([size(prediction_pdf{1},1),size(test_psd,1)]);
    if min_num_points == 1
        test_psd = interp1(f_test,test_psd,freq);
    end
    filt_dist_test = 61;
    filtered_test_psd = sgolayfilt(test_psd,1,filt_dist_test);
    for j = 1:num_predictions
        if min_num_points == 2
           prediction_pdf{j} = interp1(freq,prediction_pdf{j},f_test);
        end
        filtered_prediction_psd = sgolayfilt(prediction_pdf{j},1,filt_dist_test);
        loss(res) = loss(res) + sqrt(mean((filtered_test_psd - filtered_prediction_psd).^2))...
            /num_reservoirs/num_predictions;
    end
end
loss = mean(loss);
save(['/lustre/awikner1/LorenzModel3/N960K32I12F15wnoiseNskip',num2str(Nskip),...
    '_1/hybrid_optimresults_sigma',strrep(num2str(sigma_resin),'.',''),'rho',...
    strrep(num2str(radiusin),'.',''),'noise',strrep(num2str(resnoisein),'.',''),...
    '.mat'],'sigma_resin','radiusin','resnoisein','loss')