function ModelParams = precompute_KS(ModelParams)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
if strcmp(ModelParams.modeltype, 'ETDRK')
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
else
    warning('modeltype field is not set to ETDRK, no model parameters to precompute.')
end

end

