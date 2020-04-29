function [output,fval,exitflag,info] = get_optimal_hyperparams_model3(sigma, rho, noise)

num_reservoirs = 5;
optim_func = @(input) optimize_model3_spectrum(input(1), input(2), input(3), num_reservoirs);
input(1) = sigma;
input(2) = rho;
input(3) = noise;
options = optimset('MaxFunEvals',40);
[output,fval,exitflag,info] = fminsearch(optim_func,input,options);
