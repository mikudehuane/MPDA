#!/bin/bash

init_model_fp="{project_fp}/cloud_models/movielens_lr/checkpoint/epoch/model.ckpt"

random_seed=0
max_match=100
run_name="transfer_movielens_lr_m-${max_match}"
model="lr"

mkdir -p ../log/${run_name}/running-logs/

device="cpu"
for ti in $(seq 0 1 14)
do
  command="../scripts/transfer.py -ti=${ti} -tc=15 --device=${device} -rn=${run_name} -bn=bn -mo=${model} -mm=${max_match} -lr=0.01 -ma=random -rands=${random_seed} -uas=0 -imf=${init_model_fp}"
  echo "python -u ${command} > ../log/${run_name}/running-logs/${ti}.txt 2>&1 &"
  python -u ${command} > ../log/${run_name}/running-logs/${ti}.txt 2>&1 &
done


