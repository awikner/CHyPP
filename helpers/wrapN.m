function wrap = wrapN(x,N)
% wrapN - wraps the input vector periodically for a given periodicity size
% N.
%
% Inputs: x - vector
%         N - size
%
% Output: wrap - wrapped vector
wrap = (1 + mod(x-1, N));
