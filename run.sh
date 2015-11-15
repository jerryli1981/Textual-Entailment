#!/bin/bash
# verbose
set -x
###################
# Update items below for each train/test
###################

# training params
epochs=1000
step=0.01
numLabels=3
rangeScores=5
hiddenDim=100
wvecDim=200
miniBatch=128
mlpActivation=sigmoid
optimizer=adagrad

outFile="models/${model}_wvecDim_${wvecDim}_step_${step}_optimizer_${optimizer}.bin"


python -u main.py --step $step --mlpActivation $mlpActivation \
				  --optimizer $optimizer --hiddenDim $hiddenDim --epochs $epochs --outFile $outFile\
                  			--rangeScores $rangeScores	--numLabels $numLabels\
                  			--minibatch $miniBatch --wvecDim $wvecDim





