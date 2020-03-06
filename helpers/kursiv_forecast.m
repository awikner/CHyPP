function u = kursiv_forecast(u, ModelParams)
% kursiv_forecast - gives a 1 step forecast of the kuramoto-sivashinky
% equation using the appropriate method.
%
% Inputs:
%   u - initial condition
%
%   ModelParams - struct specifying evolution model
% Output: u - state after time ModelParams.tau

if strcmp(ModelParams.modeltype, 'ETDRK')
    
    v = fft(u);

    Nv = ModelParams.g.*fft(real(ifft(v)).^2);
    a = ModelParams.E2.*v + ModelParams.Q.*Nv;
    Na = ModelParams.g.*fft(real(ifft(a)).^2);
    b = ModelParams.E2.*v + ModelParams.Q.*Na;
    Nb = ModelParams.g.*fft(real(ifft(b)).^2);
    c = ModelParams.E2.*a + ModelParams.Q.*(2*Nb-Nv);
    Nc = ModelParams.g.*fft(real(ifft(c)).^2);
    v = ModelParams.E.*v + Nv.*ModelParams.f1 + 2*(Na+Nb).*ModelParams.f2 + Nc.*ModelParams.f3;

    u = real(ifft(v));
    
elseif strcmp(ModelParams.modeltype, 'FD')
    
    u = rk4(@ks_derivative, u, ModelParams);
    
end
