function y = rk4m3(model, y, ModelParams, s_mat_k, Z2Xmat)
%RK4 Runge-Kutta 4th order integration
%   RK4(F,Y,T,DT,...) integrates the differential equation y' = f(t,y) from
%   time T to time T+DT, where Y is the value of the solution vector at
%   time T. F is a function handle. For a scalar T and a vector Y, F(T,Y)
%   must return a column vector corresponding to f(t,y). Additional
%   arguments will be passed to the function F(T,Y,...).

k1 = feval(model,        y          , ModelParams, s_mat_k, Z2Xmat);
k2 = feval(model, y + k1*ModelParams.tau/2, ModelParams, s_mat_k, Z2Xmat);
k3 = feval(model, y + k2*ModelParams.tau/2, ModelParams, s_mat_k, Z2Xmat);
k4 = feval(model,   y + k3*ModelParams.tau,   ModelParams, s_mat_k, Z2Xmat);


y = y + ModelParams.tau/6 * (k1 + 2*k2 + 2*k3 + k4);

end
