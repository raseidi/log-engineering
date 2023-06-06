#/bin/bash

declare -a DATASETS=("RequestForPayment" "BPI_Challenge_2012_A" "BPI_Challenge_2012_Complete" "BPI_Challenge_2013_closed_problems" "BPI_Challenge_2012" "bpi17" "BPI_Challenge_2012_W_Complete" "BPI_Challenge_2012_O" "PrepaidTravelCost" "PermitLog" "bpi_challenge_2013_incidents" "BPI_Challenge_2012_W" "bpi19")
DIR="/home/seidi/datasets/logs"
ENCODINGDIM=(8 16)

for dataset in "${DATASETS[@]}"
do
    for DIM in "${ENCODINGDIM[@]}"
    do
        python3 create_dataset.py --dataset $dataset --encoding-dim $DIM
    done
done