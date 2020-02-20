function uu = kursiv_model_predict(u, ModelParams)


x = ModelParams.d*(-ModelParams.N/2+1:ModelParams.N/2)'/ModelParams.N;


v = fft(u);
% % Precompute various ETDRK4 scalar quantities:
% k = [0:ModelParams.N/2-1 0 -ModelParams.N/2+1:-1]'*(2*pi/ModelParams.d); % wave numbers
% L = (1+ModelParams.const)*k.^2 - k.^4; % Fourier multipliers
% E = exp(ModelParams.tau*L); E2 = exp(ModelParams.tau*L/2);
% M = 16; % no. of points for complex means
% r = exp(1i*pi*((1:M)-.5)/M); % roots of unity
% 
% LR = ModelParams.tau*L(:,ones(M,1)) + r(ones(ModelParams.N,1),:);
% 
% 
% Q = ModelParams.tau*real(mean( (exp(LR/2)-1)./LR ,2));
% f1 = ModelParams.tau*real(mean( (-4-LR+exp(LR).*(4-3*LR+LR.^2))./LR.^3 ,2));
% f2 = ModelParams.tau*real(mean( (2+LR+exp(LR).*(-2+LR))./LR.^3 ,2));
% f3 = ModelParams.tau*real(mean( (-4-3*LR-LR.^2+exp(LR).*(4-LR))./LR.^3 ,2));
% % Main time-stepping loop:
% 
% g = -0.5i*k;

vv = zeros(ModelParams.N, ModelParams.nstep);

vv(:,1) = v;

for n = 2:ModelParams.nstep
Nv = ModelParams.g.*fft(real(ifft(v)).^2);
a = ModelParams.E2.*v + ModelParams.Q.*Nv;
Na = ModelParams.g.*fft(real(ifft(a)).^2);
b = ModelParams.E2.*v + ModelParams.Q.*Na;
Nb = ModelParams.g.*fft(real(ifft(b)).^2);
c = ModelParams.E2.*a + ModelParams.Q.*(2*Nb-Nv);
Nc = ModelParams.g.*fft(real(ifft(c)).^2);
v = ModelParams.E.*v + Nv.*ModelParams.f1 + 2*(Na+Nb).*ModelParams.f2 + Nc.*ModelParams.f3;
vv(:,n) = v;
end
uu = (real(ifft(vv)));


