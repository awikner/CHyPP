function forecast = forecast_ks(init, KS, steps)

u = init;

v = fft(u);


for n = 1:steps
    
    Nv = KS.g.*fft(real(ifft(v)).^2);
    a = KS.E2.*v + KS.Q.*Nv;
    Na = KS.g.*fft(real(ifft(a)).^2);
    b = KS.E2.*v + KS.Q.*Na;
    Nb = KS.g.*fft(real(ifft(b)).^2);
    c = KS.E2.*a + KS.Q.*(2*Nb-Nv);
    Nc = KS.g.*fft(real(ifft(c)).^2);
    v = KS.E.*v + Nv.*KS.f1 + 2*(Na+Nb).*KS.f2 + Nc.*KS.f3;

end


forecast = real(ifft(v));
