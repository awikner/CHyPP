function [y] = rksolveModel2(init_cond,ModelParams)
% rksolveModel2 - gives a forecast of Lorenz Model 2
% using the RK4 method.
%
% Inputs:
%   init_cond - initial condition
%
%   ModelParams - struct specifying evolution model
% Output: y - state at each future time spaced by ModelParams.tau

%initialize structures
y = zeros(ModelParams.N,ModelParams.nstep);
y(:,1) = init_cond;
for i=1:ModelParams.nstep-1

    %y is handle of solution vector
    y(:,i+1) = rk4Model2(y(:,i), ModelParams);
end

return