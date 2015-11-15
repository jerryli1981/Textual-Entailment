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
wvecDim=300
miniBatch=128
mlpActivation=sigmoid
optimizer=adagrad

outFile="models/${model}_wvecDim_${wvecDim}_step_${step}_optimizer_${optimizer}.bin"

if [ "$1" == "keras" ]
then
echo "run keras"
python -u main_keras.py --step $step --mlpActivation $mlpActivation \
				  --optimizer $optimizer --hiddenDim $hiddenDim --epochs $epochs --outFile $outFile\
                  			--rangeScores $rangeScores	--numLabels $numLabels\
                  			--minibatch $miniBatch --wvecDim $wvecDim

elif [ "$1" == "lasagne" ]
then
echo "run lasagne"
python -u main_lasagne.py --step $step --mlpActivation $mlpActivation \
				  --optimizer $optimizer --hiddenDim $hiddenDim --epochs $epochs --outFile $outFile\
                  			--rangeScores $rangeScores	--numLabels $numLabels\
                  			--minibatch $miniBatch --wvecDim $wvecDim

fi








