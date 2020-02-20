function du = ks_derivative(u, ModelParams)


dx = ModelParams.deltax;


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