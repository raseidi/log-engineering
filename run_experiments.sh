#/bin/bash

declare -a DATASETS=("PermitLog" "PrepaidTravelCost" "BPI_Challenge_2012_O" "BPI_Challenge_2012" "BPI_Challenge_2012_Complete" "BPI_Challenge_2012_A" "RequestForPayment") 
DIR="/home/seidi/datasets/logs"
ENCODINGDIM=(8)
declare -a FEATURES=("EF" "OH" "EF,TIME" "OH,TIME" "EF,TIME,MFTIME" "OH,TIME,MFTIME")
declare -a MODELS=("KNN" "SVM" "RF")
# declare -a MODELS=("RF")
declare -a TARGETS=("tf_remaining_time") #"last_o_activity")


for dataset in "${DATASETS[@]}"
do
    for DIM in "${ENCODINGDIM[@]}"
    do
        for F in "${FEATURES[@]}"
        do
            for MODEL in "${MODELS[@]}"
            do
                for TARGET in "${TARGETS[@]}"
                do
                    python3 experiments.py --dataset $dataset --encoding-dim $DIM --features $F --model $MODEL --target $TARGET
                done
            done
        done
    done
done