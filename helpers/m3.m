function dZ = m3(Z, ModelParams)
%m3 - differential equation for the Lorenz model 3
% Inputs:
%   Z - values of Z at each grid point
%
%   ModelParams - struct containing model parameters
%Output: dZ - time derivative at each grid point

X = ModelParams.Z2Xmat*Z;

Y = Z-X;

XX = XYquadratic(X,X, ModelParams.K, ModelParams.s_mat_k);

YY = -bsxfun(@times,circshift(Y,-2),circshift(Y, -1)) + bsxfun(@times,circshift(Y,-1),circshift(Y, 1));

YX = -bsxfun(@times,circshift(Y,-2),circshift(X, -1)) + bsxfun(@times,circshift(Y,-1),circshift(X, 1));

dZ = XX + (ModelParams.b)^2*YY + ModelParams.c*YX - X -ModelParams.b*Y + ModelParams.F;