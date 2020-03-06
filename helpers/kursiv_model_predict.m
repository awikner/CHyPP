function uu = kursiv_model_predict(u, ModelParams)
% kursiv_model_predict - gives a forecast of the kuramoto-sivashinky
% equation using the ETDRK method.
%
% Inputs:
%   u - initial condition
%
%   ModelParams - struct specifying evolution model
% Output: u - state at each future time spaced by ModelParams.tau


v = fft(u);

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


