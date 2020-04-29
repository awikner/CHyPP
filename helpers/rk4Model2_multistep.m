function y = rk4Model2_multistep(y,ModelParams)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
for i = 1:ModelParams.num_steps
y = rk4Model2(y,ModelParams);
end

end
