function A = generate_reservoir(size, radius, degree, labindex, runiter, pool_size)

% rng(runiter*pool_size+labindex)
rng(runiter*pool_size+labindex)


sparsity = degree/size;

A = sprand(size, size, sparsity);

e = max(abs(eigs(A)));

A = (A./e).*radius;
