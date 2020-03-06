function [y] = rksolveModel3(init_cond,ModelParams)
% rksolveModel3 - gives a forecast of Lorenz Model 3
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
    %update vector
    %y is handle of solution vector
    y(:,i+1) = rk4Model3(y(:,i), ModelParams);
end

return