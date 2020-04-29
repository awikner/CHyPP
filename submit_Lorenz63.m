clear;
addpath umd_deepthought2.v2.0.0
c = parcluster;

ClusterInfo.setWallTime('0:10:00')
ClusterInfo.setMemUsage('5g')
ClusterInfo.setEmailAddress('awikner1@umd.edu')
j = c.batch(@CHyPP_test_Lorenz63,0,{0.1},'pool', 1);