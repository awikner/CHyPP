function [ dxdt ] = m1(x,ModelParams)

%LORENZ MODEL 1: differential equation for the lorenz 96 model
%   Detailed explanation goes here

p = circshift(x,[-2,0]).*circshift(x, [-1,0]);

q = circshift(x, [-1,0]).*circshift(x, [1,0]);

dxdt = -x - p + q + ModelParams.F;

end