function [ dxdt ] = m1(x,ModelParams)

%m1 - differential equation for the lorenz 96 model
% Inputs:
%   x - values of X at each grid point
%
%   ModelParams - struct containing model parameters
%Output: dxdt - time derivative at each grid point

p = circshift(x,[-2,0]).*circshift(x, [-1,0]);

q = circshift(x, [-1,0]).*circshift(x, [1,0]);

dxdt = -x - p + q + ModelParams.F;

end