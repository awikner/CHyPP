clear;
addpath umd_deepthought2.v2.0.0
c = parcluster;

ClusterInfo.setWallTime('168:00:00')
ClusterInfo.setMemUsage('5g')
ClusterInfo.setEmailAddress('awikner1@umd.edu')

sigma = 0.090786;
rho = 0.64443;
noise = 0.090212;
for i=1:length(sigma)
    j = c.batch(@get_optimal_hyperparams_model3,4,{sigma(i),rho(i),noise(i)},'pool', 12);
end