function KS =  precompute_ks_params(ModelParams)

k = [0:ModelParams.N/2-1 0 -ModelParams.N/2+1:-1]'*(2*pi/ModelParams.d); % wave numbers
L = (1+ModelParams.const)*k.^2 - k.^4; % Fourier multipliers
KS.E = exp(ModelParams.tau*L); KS.E2 = exp(ModelParams.tau*L/2);
M = 16; % no. of points for complex means
r = exp(1i*pi*((1:M)-.5)/M); % roots of unity

LR = ModelParams.tau*L(:,ones(M,1)) + r(ones(ModelParams.N,1),:);


KS.Q = ModelParams.tau*real(mean( (exp(LR/2)-1)./LR ,2));
KS.f1 = ModelParams.tau*real(mean( (-4-LR+exp(LR).*(4-3*LR+LR.^2))./LR.^3 ,2));
KS.f2 = ModelParams.tau*real(mean( (2+LR+exp(LR).*(-2+LR))./LR.^3 ,2));
KS.f3 = ModelParams.tau*real(mean( (-4-3*LR-LR.^2+exp(LR).*(4-LR))./LR.^3 ,2));
% Main time-stepping loop:

KS.g = -0.5i*k;


