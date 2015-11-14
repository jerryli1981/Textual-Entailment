#!/bin/bash
# verbose
set -x
###################
# Update items below for each train/test
###################

# training params
epochs=100
step=0.01
numLabels=3
hiddenDim=100
wvecDim=200
miniBatch=128
model=LSTM
optimizer=sgd
debug=False
useLearnedModel=False

outFile="models/${model}_wvecDim_${wvecDim}_step_${step}_optimizer_${optimizer}.bin"



python -u main.py --debug $debug --useLearnedModel $useLearnedModel --step $step --repModel $model \
				  --optimizer $optimizer --hiddenDim $hiddenDim --epochs $epochs --outFile $outFile\
                  				--outputDim $numLabels --minibatch $miniBatch --wvecDim $wvecDim





