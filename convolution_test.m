input_size = 20;
conv_size = 5;
num_filters = 10;
pool_size = 2;

net0 = imageInputLayer([input_size,1,1],'Normalization','none');
net1 = convolution2dLayer([conv_size,1],num_filters,'NumChannels',1);
net1.Weights = 1e-3*rand(conv_size,1,1,num_filters);
net1.Bias = 1e-3*zeros(1,1,num_filters);
% net2 = maxPooling2dLayer([pool_size,1]);
net3 = regressionLayer;
layers = [net0,net1,net3];
% layers = [net0,net1,net3];
network = assembleNetwork(layers);

input = randn(input_size,1,1);

output = predict(network,input)