clc;clear;

%% input A: the adjacency matrix

option.maxiter = 1000;
option.tolfun = 5e-4;
option.maxiter_pre = 500;
option.verbose = 1;
option.UpdateVi = 0; 
option.lambda = 1;

layers = [256, 128, 36]; %% three layers, the last layer corresponds to the number of communities to detect

p = numel(layers);

%% Deep AE NMF
[U, V, dnorm, dnormarray] = DANMF(A, layers, option);
Vp = V{p}';

%% Vp captures the community memberships
