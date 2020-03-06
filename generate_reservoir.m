function A = generate_reservoir(size, radius, degree, labindex, runiter, num_res, pool_size, res)
% generate_reservoir - creates the adjacency matrix for a recurrent
% artificial neural network that we use as our reservoir.
% Inputs:
%   size - number of nodes in the reservoir
%
%   radius - spectral radius of the reservoir
%
%   degree - average in-degree of each node in the reservoir
%
%   labindex - worker reservoir is being generated for, used to set seed
%              for random reservoir generation
%
%   runiter - iterator used for setting random reservoir seed
%
%   num_res - total number of reservoirs being used, used to set seed for
%             random reservoir generation
%
%   pool_size - total number of workers being used, used to set seed
%               for random reservoir generation
%
%   res - index for reservoir on this particular worker, used to set seed
%         for random reservoir generation
%
% Outputs:
%   A - reservoir adjacency matrix

% set random seed
rng(runiter*num_res+(labindex-1)*pool_size+res,'twister')

% Generate random Erdos-Renyi network with specified in-degree
sparsity = degree/size;

A = sprand(size, size, sparsity);

% Scale network weights so that maximum absolute eigenvalue is radius
e = abs(eigs(A,1));

A = (A./e).*radius;
