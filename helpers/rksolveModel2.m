function [y] = rksolveModel2(init_cond,ModelParams)

%initialize structures
y = zeros(ModelParams.N,ModelParams.nstep);
y(:,1) = init_cond;
for i=1:ModelParams.nstep-1
    %update vector
    %y is handle of solution vector
    y(:,i+1) = rk4Model2(y(:,i), ModelParams);
end

return