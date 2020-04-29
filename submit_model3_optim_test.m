clear;
addpath umd_deepthought2.v2.0.0
c = parcluster;

ClusterInfo.setWallTime('4:00:00')
ClusterInfo.setMemUsage('5g')
ClusterInfo.setEmailAddress('awikner1@umd.edu')
sigma = 0.090786;
rho = 0.64443;
noise = 0.090212;
pool_size = 12;
num_reservoir = 6;
j = c.batch(@optimize_model3_spectrum,1,{sigma,rho,noise,num_reservoir},'pool', pool_size);