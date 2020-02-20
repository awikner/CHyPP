function XY = XYquadratic(X,Y,K,s_mat)


W = (s_mat*X)./K;
V = (s_mat*Y)./K;

XY = -bsxfun(@times, circshift(W, -2*K), circshift(V, -K)) +  ...
    s_mat*(bsxfun(@times, circshift(W, -K), circshift(Y, K)))./K;


