# NeurIPS2022
training-balanced-nets

Example code that demonstrates how training a RNN on tracking a time-varying input makes it more tightly balanced and tracking on multiple stimulus leads to the emergence of tightly balanced subnetworks that are weakly conncted in between.

## to run code for tracking one input stimulus:
include("ib12trainingBalancedNetsNeurIPS01.jl")
b=1.0
N, Nepochs, B, S, T, seedIC, seedInput, seedNet, lrIn, beta1, beta2, IC, startRandom, plotStuff,tauS, gInit,gbarInit,Iext, Iext1,dt,delay,sigma,nl,wsStd,wsMean,wrStd,wrMean=100,100001,10,1,50,1,1,1,1e-3,0.9,0.999,1,true,true,0.1,0.032,-1.0f0*b,1.0f0*b,0.1f0*b,0.2f0,0,1.0f0,relu,0.01f0,1.0f0,0.2f0,1.0f0; seed=1
losstrace = @time trainCopy.(N, Nepochs, B, S, T,seedIC, seedInput, seed, lrIn, beta1, beta2, IC, startRandom,plotStuff,tauS, gInit,gbarInit, Iext,Iext1,dt,delay,sigma,nl,0wsStd,wsMean,wrStd,wrMean,1f-4,true,"codeNeurIPS")

## to run code for tracking two input stimulus:
include("ib12trainingBalancedNetsNeurIPS01.jl")
b=1.0
N, Nepochs, B, S, T, seedIC, seedInput, seedNet, lrIn, beta1, beta2, IC, startRandom, plotStuff,tauS, gInit,gbarInit,Iext, Iext1,dt,delay,sigma,nl,wsStd,wsMean,wrStd,wrMean=100,100001,10,2,50,1,1,1,1e-3,0.9,0.999,1,true,true,0.1,0.032,-1.0f0*b,1.0f0*b,0.1f0*b,0.2f0,0,1.0f0,relu,0.01f0,1.0f0,0.2f0,1.0f0; seed=1
losstrace = @time trainCopy.(N, Nepochs, B, S, T,seedIC, seedInput, seed, lrIn, beta1, beta2, IC, startRandom,plotStuff,tauS, gInit,gbarInit, Iext,Iext1,dt,delay,sigma,nl,0wsStd,wsMean,wrStd,wrMean,1f-4,true,"codeNeurIPS")


