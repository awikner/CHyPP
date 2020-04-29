function data = rk4prediction(x,ModelParams)

data = zeros(numel(x),ModelParams.nstep);
data(:,1) = x;
for i = 1:ModelParams.nstep-1
	data(:,i+1) = ModelParams.predict(data(:,i),ModelParams);
end

