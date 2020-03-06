function Z2Xmat = Z2X(ModelParams)
% Z2X - computes the matrix used for smoothing the atmospheric variable Z
% in Lorenz Model 3 to obtain X.
%
% Input: ModelParams - struct specifying model parameters
% Output: Z2Xmat - matrix for smoothing Z
Z2Xmat = zeros(ModelParams.N, ModelParams.N);


row1 = zeros(1,ModelParams.N);


for i = 1:ModelParams.I+1
    row1(i) = (ModelParams.alpha - ModelParams.beta*abs(i-1));
end

for i = wrapN(1 - ModelParams.I, ModelParams.N): ModelParams.N
    row1(i) = (ModelParams.alpha - ModelParams.beta*abs(ModelParams.N - i+1));
end

row1(ModelParams.I + 1) = row1(ModelParams.I + 1)/2;
row1(wrapN(1 - ModelParams.I, ModelParams.N)) = row1(wrapN(1 - ModelParams.I, ModelParams.N))/2;

for i =1:ModelParams.N
    Z2Xmat(i, :) = circshift(row1, [0,i-1]);
end