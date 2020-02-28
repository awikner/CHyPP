function A = generate_reservoir(size, radius, degree, labindex, runiter, num_res, pool_size, res)

% rng(runiter*pool_size+labindex)
rng(runiter*num_res+(labindex-1)*pool_size+res,'twister')


sparsity = degree/size;

A = sprand(size, size, sparsity);

e = abs(eigs(A,1));

A = (A./e).*radius;
