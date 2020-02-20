function [y] = rk_solve_m2(init_cond,ModelParams, s_mat_k, Z2Xmat)

%initialize structures
y = zeros(ModelParams.N,ModelParams.nstep+1);
y(:,1) = init_cond;
for i=1:ModelParams.nstep
    %update vector
    %y is handle of solution vector
    y(:,i+1) = rk4m3(@m2, y(:,i), ModelParams, s_mat_k, Z2Xmat);
end

return