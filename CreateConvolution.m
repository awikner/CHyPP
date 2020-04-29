function network = CreateConvolution(input_size,train_input_sequence,resparams,conv_type)

outputlayer = regressionLayer;

switch conv_type
    case 'single'
        filter_size = ceil(input_size/3);
        inputlayer = imageInputLayer([input_size,1,1],'Normalization','none');
        pad1 = circularPadLayer(filter_size,'pad1');
        num_filters = 32;
        net1 = convolution2dLayer([filter_size,1],num_filters,'NumChannels',1);
        net1.Weights = rand(filter_size,1,1,num_filters);
        net1.Bias = rand(1,1,num_filters);
        BN1 = batchNormalizationLayer('Scale',ones(1,1,num_filters),'Offset',zeros(1,1,num_filters));
        Nonlin1 = reluLayer('Name','relu1');
        net2 = fullyConnectedLayer(resparams.N);
        net2.Weights = resparams.sigma * randn([resparams.N,(input_size)*num_filters]);
        net2.Bias = randn([resparams.N 1])*0.0001 + 1;
%         BN2 = batchNormalizationLayer('Scale',ones(1,1,resparams.N),'Offset',zeros(1,1,resparams.N));
        layers = [inputlayer;pad1;net1;BN1;Nonlin1;net2;outputlayer];
%         network = assembleNetwork(layers);
    case 'singlewpool'
        filter_size = ceil(input_size/3);
        inputlayer = imageInputLayer([input_size,1,1],'Normalization','none');
        pad1 = circularPadLayer(filter_size,'pad1');
        num_filters = 32;
        net1 = convolution2dLayer([filter_size,1],num_filters,'NumChannels',1);
        net1.Weights = rand(filter_size,1,1,num_filters);
        net1.Bias = rand(1,1,num_filters);
        BN1 = batchNormalizationLayer('Scale',ones(1,1,num_filters),'Offset',zeros(1,1,num_filters));
        Nonlin1 = reluLayer('Name','relu1');
        net2 = fullyConnectedLayer(resparams.N);
        net2.Weights = resparams.sigma * randn([resparams.N,(input_size)*num_filters]);
        net2.Bias = randn([resparams.N 1])*0.0001 + 1;
        
%         BN2 = batchNormalizationLayer('Scale',ones(1,1,resparams.N),'Offset',zeros(1,1,resparams.N));
        layers = [inputlayer;pad1;net1;BN1;Nonlin1;net2;outputlayer];
%         network = assembleNetwork(layers);
    case 'WorldModel'
        filter_size = ceil(input_size/2+1);
        num_filters = 32;
        net1 = convolution2dLayer([filter_size,1],num_filters,'NumChannels',1);
        net1.Weights = rand(filter_size,1,1,num_filters);
        net1.Bias = rand(1,1,num_filters);
        num_filters_2 = 64;
        filter_size_2 = ceil(input_size/4+2);
        net2 = convolution2dLayer([filter_size_2,1],num_filters_2,'NumChannels',num_filters);
        net2.Weights = rand(filter_size_2,1,num_filters,num_filters_2);
        net2.Bias = rand(1,1,num_filters_2);
        num_filters_3 = 128;
        filter_size_3 = ceil(input_size/8+2);
        net3 = convolution2dLayer([filter_size_3,1],num_filters_3,'NumChannels',num_filters_2);
        net3.Weights = rand(filter_size_3,1,num_filters_2,num_filters_3);
        net3.Bias = rand(1,1,num_filters_3);
        net4 = fullyConnectedLayer(resparams.N);
        net4.Weights = resparams.sigma * randn([resparams.N,(filter_size_2 - filter_size_3 + 1)*num_filters_3]);
        net4.Bias = randn([resparams.N 1])*0.0001 + 1;
        layers = [inputlayer;net1;net2;net3;net4;outputlayer];
        network = assembleNetwork(layers);
    otherwise
        error('Convolutional input coupling type not recognized.')
end

x = 1;

switch class(layers)
    case 'nnet.cnn.layer.Layer'
        for l = 1:numel(layers)
            if isa(layers(l),'nnet.cnn.layer.BatchNormalizationLayer')
                mini_net = assembleNetwork([layers(1:l-1);layers(end)]);
                mini_net_result = predict(mini_net,train_input_sequence(:,1));
                output_size = size(mini_net_result,1);
                output_channels = size(mini_net_result,3);
                mini_net_train_result = zeros(output_size*size(train_input_sequence,2),1,output_channels);
                mini_net_train_result(1:output_size,1,:) = mini_net_result;
                for i=2:size(train_input_sequence,2)
                    mini_net_train_result((i-1)*output_size + 1:i*output_size,1,:) = predict(mini_net, train_input_sequence(:,i));
                end
                layers(l) = batchNormalizationLayer('Scale',layers(l).Scale,...
                    'Offset',layers(l).Offset,'TrainedMean',...
                    mean(mini_net_train_result),'TrainedVariance',var(mini_net_train_result));
            end
        end
end

network = assembleNetwork(layers);
                    
                