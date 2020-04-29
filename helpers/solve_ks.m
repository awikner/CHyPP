function data = solve_ks(init, ModelParams)

x = ModelParams.d*(-ModelParams.N/2+1:ModelParams.N/2)'/ModelParams.N;

u = init;


v = fft(u);
% Precompute various ETDRK4 scalar quantities:
k = [0:ModelParams.N/2-1 0 -ModelParams.N/2+1:-1]'*(2*pi/ModelParams.d); % wave numbers
L = (1+ModelParams.const)*k.^2 - k.^4; % Fourier multipliers
E = exp(ModelParams.tau*L); E2 = exp(ModelParams.tau*L/2);
M = 16; % no. of points for complex means
r = exp(1i*pi*((1:M)-.5)/M); % roots of unity

LR = ModelParams.tau*L(:,ones(M,1)) + r(ones(ModelParams.N,1),:);


Q = ModelParams.tau*real(mean( (exp(LR/2)-1)./LR ,2));
f1 = ModelParams.tau*real(mean( (-4-LR+exp(LR).*(4-3*LR+LR.^2))./LR.^3 ,2));
f2 = ModelParams.tau*real(mean( (2+LR+exp(LR).*(-2+LR))./LR.^3 ,2));
f3 = ModelParams.tau*real(mean( (-4-3*LR-LR.^2+exp(LR).*(4-LR))./LR.^3 ,2));
% Main time-stepping loop:

g = -0.5i*k;

vv = zeros(ModelParams.N, ModelParams.nstep);

vv(:,1) = v;

for n = 2:ModelParams.nstep
Nv = g.*fft(real(ifft(v)).^2);
a = E2.*v + Q.*Nv;
Na = g.*fft(real(ifft(a)).^2);
b = E2.*v + Q.*Na;
Nb = g.*fft(real(ifft(b)).^2);
c = E2.*a + Q.*(2*Nb-Nv);
Nc = g.*fft(real(ifft(c)).^2);
v = E.*v + Nv.*f1 + 2*(Na+Nb).*f2 + Nc.*f3;
vv(:,n) = v;
end


data = transpose(real(ifft(vv)));
