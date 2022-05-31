# this script trains an RNN to track a time-varying input signal (OU-process) by a linear readout.
# over training, the network becomes more tightly balanced.

include("../src/trainingBalancedNets.jl")
include("../src/autoencodertask.jl")

# Set initial b to a loosely balanced regime.
b=1.0


# set other parameters
N, Nepochs, B, S, T, seedIC, seedInput, seedNet, lrIn, beta1, beta2, IC, startRandom, plotStuff,tauS, gInit,gbarInit,Iext, Iext1,dt,sigma,nl,wsStd,wsMean,wrStd,wrMean,ActivityReg,removeDiag,dir=100,100001,10,1,50,1,1,1,1e-3,0.9,0.999,1,true,true,0.1,0.032,-1.0f0*b,1.0f0*b,0.1f0*b,0.2f0,1.0f0,relu,0.01f0,1.0f0,0.2f0,1.0f0,1f-4,true,"codeNeurIPS"

S=2
# train an plot results for one stimulus
losstrace = @time trainCopy.(N, Nepochs, B, S, T,seedIC, seedInput, seedNet, lrIn, beta1, beta2, IC, startRandom,plotStuff,tauS, gInit,gbarInit, Iext,Iext1,dt,sigma,nl,wsStd,wsMean,wrStd,wrMean,ActivityReg,removeDiag,dir)
