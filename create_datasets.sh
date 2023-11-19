#/bin/bash

declare -a DATASETS=("bpi17" "PermitLog" "PrepaidTravelCost" "BPI_Challenge_2012_O" "BPI_Challenge_2012" "BPI_Challenge_2012_Complete" "BPI_Challenge_2012_A" "RequestForPayment") 

DIR="/home/seidi/datasets/logs"
ENCODINGDIM=(8)

for dataset in "${DATASETS[@]}"
do
    for DIM in "${ENCODINGDIM[@]}"
    do
        python3 create_dataset.py --dataset $dataset --encoding-dim $DIM
    done
done