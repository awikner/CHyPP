function CHyPP_test_Lorenz63(tau)
% Set parameters of L63 equation
ModelParams.a = 10;
ModelParams.b = 28;
ModelParams.c = 8/3;
ModelParams.tau = tau;
% Set model evolution functions
ModelParams.predict = @(x,ModelParams) rk4(@lorenz63,x,ModelParams);
ModelParams.prediction = @rk4prediction;
PredictLength = 25/tau;
generate_Lorenz63_data(ModelParams,PredictLength);

num_res = 6;

parfor runiter = 1:num_res
    radii = 0.05:0.05:1;
    input_weights = 0.1:0.1:3;
    leakages = 0.05:0.05:1;
    noises = logspace(-4,-1,4);
    TrainFile = ['Lorenz63_Data/tau',strrep(num2str(tau),'.',''),'/L63_train_input_sequence.mat'];
    TestFile = ['Lorenz63_Data/tau',strrep(num2str(tau),'.',''),'/L63_test_input_sequence.mat'];
    StartFile = ['Lorenz63_Data/tau',strrep(num2str(tau),'.',''),'/L63_pred_start_indices.mat'];
    OutputLocation = ['Lorenz63_Data/tau',strrep(num2str(tau),'.','')];
    %%
    % Generate KS Equation training data and set transient discard length

    NumRes = 1;
    TrainLength = round(200/tau);
    ReservoirSize = 300;
    AvgDegree = 3;
    RidgeReg = 1e-4;
    Predictions = 200;
    TrainSteps = 1;
    RunIter = runiter;
    OutputRMS = true;
    ErrorCutoff = 0.2;
    ResOnly = true;
    for InputWeight = input_weights
        for SpectralRadius = radii
            for Leakage = leakages
                for Noise = noises
                    %% Run a reservoirs-only prediction using the same number of reservoirs

                    % [avg_pred_length_res,res_pred_file,res_rms_file] = CHyPP('PoolSize',PoolSize,...
                    %     'NumRes',NumRes,'TrainLength',TrainLength,...
                    %     'ReservoirSize',ReservoirSize,'AvgDegree',AvgDegree,'LocalOverlap',LocalOverlap,...
                    %     'InputWeight',InputWeight,'SpectralRadius',SpectralRadius,'RidgeReg',RidgeReg,...
                    %     'Predictions',Predictions,'PredictLength',PredictLength,'TrainSteps',TrainSteps,...
                    %     'ResOnly',ResOnly,'Noise',Noise,'RunIter',RunIter,'OutputData',OutputData,...
                    %     'ModelParams',ModelParams,'ErrorCutoff',ErrorCutoff);
                    CHyPP_serial('NumRes',NumRes,'TrainLength',TrainLength,...
                        'ReservoirSize',ReservoirSize,'AvgDegree',AvgDegree,...
                        'InputWeight',InputWeight,'SpectralRadius',SpectralRadius,'RidgeReg',RidgeReg,...
                        'Predictions',Predictions,'PredictLength',PredictLength,'TrainSteps',TrainSteps,...
                        'ResOnly',ResOnly,'Noise',Noise,'RunIter',RunIter,'OutputRMS',OutputRMS,...
                        'ModelParams',ModelParams,'ErrorCutoff',ErrorCutoff,'Leakage',Leakage,...
                        'TrainFile',TrainFile,'TestFile',TestFile,'OutputLocation',OutputLocation,...
                        'StartFile',StartFile);
                end
            end
        end
    end
end