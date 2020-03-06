function dZ = m2(Z, ModelParams)

%m2 - differential equation for the Lorenz model 2
% Inputs:
%   Z - values of Z at each grid point
%
%   ModelParams - struct containing model parameters
%Output: dZ - time derivative at each grid point

XX = XYquadratic(Z,Z, ModelParams.K, ModelParams.s_mat_k);

dZ = XX - Z + ModelParams.F;