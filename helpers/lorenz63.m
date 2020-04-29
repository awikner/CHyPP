function [ dxdt ] = lorenz63(x, ModelParams)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
dxdt    = zeros(3,1);
dxdt(1) = ModelParams.a*(-x(1)+x(2));
dxdt(2) = (ModelParams.b-x(3))*x(1)-x(2);
dxdt(3) = -ModelParams.c*x(3)+x(1)*x(2);

end

