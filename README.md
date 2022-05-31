# A time-resolved theory of information encoding in recurrent neural networks

This repository contains the implementation code for manuscript: <br>
__A time-resolved theory of information encoding in recurrent neural networks__ <br>
## Overview
In this work we show theoretically and empirically that more tightly balanced networks can track time-varying signals more reliable which results in a higher information rate. Moreover, this eample code demonstrates how training a RNN on tracking a time-varying input makes it more tightly balanced and tracking on multiple stimulus leads to the emergence of tightly balanced subnetworks that are weakly conncted in between.

## Installation

#### Prerequisites
- Download [Julia](https://julialang.org/downloads/) 
- 
#### Dependencies
- Julia (>= 1.5, tested on 1.6)
- Flux, PyPlot, Statistics, DelimitedFiles,RandomNumbers, ForwardDiff, BSON
- scipy





## Repository Overview

### src/ contains the source files
generateInputOutputOUAnalytical.jl - this solves the OU-process analytically incrementally and generates input and target output
ib12trainingBalancedNetsNeurIPS03plotJmeans.jl - this trains a RNN on the autoencoder task and visualizes the results during training

### example_code/
Example scripts for training networks on one, two and three stimuli.\
runOneStimulus.jl trains an RNN on tracking one OU-signal showing that the network becomes more tightly balanced over training epochs.
runTwoStimuli.jl trains an RNN on two OU-signal stimulus showing that the network becomes more tightly balanced over training epochs and breaks up into two weakly-connected subnetworks.
runTheeStimuli.jl trains an RNN on two OU-signal stimulus showing that the network becomes more tightly balanced over training epochs and breaks up into three weakly-connected subnetworks.

All simulations were run on a single CPU and took on the order of minutes to a few of hours.

