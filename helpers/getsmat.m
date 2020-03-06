function s_mat = getsmat(N, K)
% getsmat - obtain the summation matrix used in calculating the coupling
% term in Lorenz Models 2 and 3
%
% Inputs:
%   N - total number of grid points
%
%   K - coupling distance
%Outputs: s_mat - summation matrix

mask = zeros(N,1);
% Determine J
if mod(K,2)==0
    J = K/2;
elseif mod(K,2)==1
    K = (K-1)/2;
end

% Determine single row of s_mat using periodic boundary conditions
mask(wrapN(1-(J)+1, N):N) = 1;
mask(1:wrapN(1+J-1,N)) = 1;
if mod(K,2) == 0
    mask(wrapN(1 - J, N)) = 1/2;
    mask(wrapN(1 + J, N)) = 1/2;
elseif mod(K,2) == 1
    mask(wrapN(1 - J, N)) = 1;
    mask(wrapN(1 + J, N)) = 1;
end
% Cyclically shift this row to calculate full s_mat
s_mat = zeros(N,N);
for i = 1:N
    s_mat(i, :) = transpose(circshift(mask, i-1));
end
