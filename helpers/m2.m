function dZ = m2(Z, ModelParams)


XX = XYquadratic(Z,Z, ModelParams.K, ModelParams.s_mat_k);

dZ = XX - Z + ModelParams.F;