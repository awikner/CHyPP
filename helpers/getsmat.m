function s_mat = getsmat(N, K)

mask = zeros(N,1);

J = K/2;

mask(wrapN(1-(J)+1, N):N) = 1;
mask(1:wrapN(1+J-1,N)) = 1;
mask(wrapN(1 - J, N)) = 1/2;
mask(wrapN(1 + J, N)) = 1/2;
s_mat = zeros(N,N);
for i = 1:N
    s_mat(i, :) = transpose(circshift(mask, i-1));
end
