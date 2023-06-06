#/bin/bash

declare -a DATASETS=("RequestForPayment" "BPI_Challenge_2012_A" "BPI_Challenge_2012_Complete" "BPI_Challenge_2013_closed_problems" "BPI_Challenge_2012" "bpi17" "BPI_Challenge_2012_W_Complete" "BPI_Challenge_2012_O" "PrepaidTravelCost" "PermitLog" "bpi_challenge_2013_incidents" "BPI_Challenge_2012_W" "bpi19")
DIR="/home/seidi/datasets/logs"
ENCODINGDIM=(8 16)
declare -a FEATURES=("all" "EF" "EF,TF" "EF,MF")
declare -a MODELS=("RF" "KNN" "SVM" "MLP" "XGB")

for dataset in "${DATASETS[@]}"
do
    for DIM in "${ENCODINGDIM[@]}"
    do
        for F in "${FEATURES[@]}"
        do
            for MODEL in "${MODELS[@]}"
            do
                python3 experiments.py --dataset $dataset --encoding-dim $DIM --features $F --model $MODEL
            done
        done
    done
done