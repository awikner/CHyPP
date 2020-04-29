clear;
load('LorenzModel3_Data/fixed_reservoir-pool24res125locality10trainlen280000sigma1radius06betares00001jobid1runiter1_wnoise025Nskip8_spectrum.mat')
load('LorenzModel3_Data/M3_train_input_sequence.mat')
load('LorenzModel3_Data/M3_test_input_sequence.mat')
Fs = 1/0.005;
transient_length = 50*1/0.005;
N_test = size(test_input_sequence,1);
[test_psd,f_test] = periodogram(test_input_sequence(:,1),hamming(N_test),[],200);
resparams.predictions = 80;
for i=1:resparams.predictions
    if i==1
        reservoir_psd = prediction_pdf{i}/resparams.predictions;
    else
        reservoir_psd = prediction_pdf{i}/resparams.predictions + reservoir_psd;
    end
end
f_pred = freq;
load('Data/N960K32I12F15wnoiseNskip8_1/fixed_hybrid-pool24res125locality10trainlen280000sigma1radius06betares00001jobid1runiter1_wnoise00155Nskip8_spectrum.mat')
resparams.predictions = 80;
for i=1:resparams.predictions
    if any(imag(prediction_pdf{i}) ~=0)
        disp(i)
    end
    if i==1
        hybrid_pred_psd = prediction_pdf{i}/resparams.predictions;
    else
        hybrid_pred_psd = prediction_pdf{i}/resparams.predictions + hybrid_pred_psd;
    end
end

Nskipin = 8;
ModelParams.N = 960/Nskipin;
ModelParams.K = ModelParams.N/30;
ModelParams.F = 15;
ModelParams.b = 10;
ModelParams.c = 2.5;
ModelParams.I = 12;
ModelParams.tau = 0.005;
ModelParams.noise = 0;

ModelParams.alpha = (3*(ModelParams.I)^2 + 3)/(2*(ModelParams.I^3) + 4*ModelParams.I);
ModelParams.beta = (2*(ModelParams.I)^2 + 1)/((ModelParams.I^4) + 2*(ModelParams.I)^2);

Z2Xmat = sparse(Z2X(ModelParams));
s_mat_k = sparse(getsmat(ModelParams.N, ModelParams.K));


ModelParams.nstep = resparams.predict_length;
%     test_input_sequence = test_input_sequence{1};
%%
model_average_diff = zeros(1,ModelParams.nstep);
hybrid_average_diff = zeros(1,resparams.predict_length);
model_pred_length = zeros(1,resparams.predictions);
pred_length = zeros(1,resparams.predictions);
hybrid_err = cell(1,resparams.predictions);
model_err  = cell(1,resparams.predictions);
error_cutoff = 0.85;
iter = 0;
hybrid_err_sum = 0;

 for i=1:1
    xinit = savepred{i}(:,1).*datavar+datamean;
    data1 = rk_solve_m2( xinit, ModelParams, s_mat_k, Z2Xmat);
    data1 = data1(:,2:end);

    model_sequence = transpose(data1);
    model_sequence = model_sequence - datamean;
    model_sequence = model_sequence./datavar;
    data1 = model_sequence';
 end
 %%
 N_pred = length(data1(1,transient_length+1:end));
 [model_pred_psd,f_pred] = periodogram(data1(1,transient_length+1:end),hamming(N_pred),[],Fs);
 %%
figure
plot(f_test,sgolayfilt(test_psd,1,41),'LineWidth',2.5)
hold on
plot(f_pred,sgolayfilt(hybrid_pred_psd,1,101),'LineWidth',2.5);
plot(f_pred,sgolayfilt(reservoir_psd,1,101),'LineWidth',2.5);
plot(f_pred,sgolayfilt(model_pred_psd,1,101),'LineWidth',2.5);
% plot(f_test,test_psd)
% hold on
% plot(f_pred,hybrid_pred_psd);
% plot(f_pred,reservoir_psd);
% plot(f_pred,model_pred_psd);
grid on
xlabel('Frequency (1/Model Time)')
ylabel('Power Spectrum')
legend({'Truth','Hybrid','Reservoir','Model'})
axis([0,2,0,10])
% set(gca, 'YScale', 'log')