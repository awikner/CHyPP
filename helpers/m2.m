function dZ = m2(Z, ModelParams, s_mat_k, Z2Xmat)


X = Z;

XX = XYquadratic(X,X, ModelParams.K, s_mat_k);

dZ = XX - X + ModelParams.F;