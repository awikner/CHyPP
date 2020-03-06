function XY = XYquadratic(X,Y,K,s_mat)
% XYquadratic - calculated the quadractic coupling term in Lorenz Models 2
% and 3 given some coupling distance.
%
% Inputs:
%   X - first coupling input
%   Y - second coupling input
%   K - coupling distance in grid points
%   s_mat - summation matrix used in calculating coupling
%
% Output: XY - quadratic coupling

W = (s_mat*X)./K;
V = (s_mat*Y)./K;

XY = -bsxfun(@times, circshift(W, -2*K), circshift(V, -K)) +  ...
    s_mat*(bsxfun(@times, circshift(W, -K), circshift(Y, K)))./K;


