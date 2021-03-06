function du = ks_derivative(u, ModelParams)
% ks_derivative - approximates the right side of the Kuramoto-Sivashinsky
% equation using the finite difference method.
%
% Inputs:
%   u - spatial values of the system state at grid points
%
%   ModelParams - struct which specified model parameters
% Outputs: du - right side of the KS equation

dx = ModelParams.deltax;

% Evaluate using FD with some order
if ModelParams.order == 4
 
    ux = ((1/12)*circshift(u, -2) - (2/3)*circshift(u, -1) ...
        + (2/3)*circshift(u, 1) - (1/12)*circshift(u, 2))./dx;

    uux = bsxfun(@times, ux, u);

    uxx = ((-1/12)*circshift(u, -2) + (4/3)*circshift(u, -1) ...
        + (4/3)*circshift(u, 1) - (1/12)*circshift(u,2) -(5/2)*u)./(dx^2);

    uxxxx = ((-1/6)*circshift(u, -3) + (2)*circshift(u, -2) - (13/2)*circshift(u, -1) ...
        - (13/2)*circshift(u, 1) + (2)*circshift(u,2) - (1/6)*circshift(u,3) + (28/3)*u)./(dx^4);


elseif ModelParams.order == 2
    
    ux = ((-1/2)*circshift(u, -1) ...
    + (1/2)*circshift(u, 1))./dx;

    uux = bsxfun(@times, ux, u);

    uxx = (circshift(u, -1) ...
        + circshift(u, 1) - 2*u)./(dx^2);

    uxxxx = (circshift(u, -2) - (4)*circshift(u, -1) ...
        - (4)*circshift(u, 1) + circshift(u,2) + (6)*u)./(dx^4);

    
end
    

du = -uux - uxx - uxxxx;