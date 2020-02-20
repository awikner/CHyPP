function [ dzdt ] = m4modf(z, ModelParams)

dzdt = zeros(size(z));

x = z(1:ModelParams.K);

y_r = reshape(z(ModelParams.K+1:end), [ModelParams.J, ModelParams.K]);

y = z(ModelParams.K+1:end);

x1 = bsxfun(@times, circshift(x,-2, 1), circshift(x,-1, 1));

x3 = ModelParams.hcb*transpose(sum(y_r, 1));

x2 = bsxfun(@times, circshift(x, -1, 1), circshift(x, 1, 1));

dxdt = -x - x1 + x2 -x3 + ModelParams.Fx + ...
    ModelParams.Frand + ModelParams.Fwave;

dzdt(1:ModelParams.K) = dxdt;

y1 = bsxfun(@times, circshift(y, -2, 1), circshift(y, -1, 1));

y2 = bsxfun(@times, circshift(y,-1,1), circshift(y, 1, 1));

y3 = -ModelParams.cb*y1 + ModelParams.cb*y2 - ModelParams.c*y;

y3_r = reshape(y3, [ModelParams.J, ModelParams.K]);

dydt = transpose(bsxfun(@plus, y3_r', ModelParams.hcb*x)) + ModelParams.Fy;

dzdt(ModelParams.K + 1:end) = reshape(dydt, [ModelParams.K*ModelParams.J,1]);

end