mapping_fp="{data_fd}/Amazon/Electronics_5/processed/item2category.csv"
init_model_fp="{project_fp}/cloud_models/amazon_pnn/checkpoint/epoch/model.ckpt"
data_pfd="{data_fd}/Amazon/Electronics_5/processed"
train_data_fd="${data_pfd}/ts=1385078400_train"
eval_data_fd="${data_pfd}/ts=1385078400_test"
examine_user_list_fp="${data_pfd}/ts=1385078400_user-intersect_trainset20.json"

random_seed=0
max_match=200
device="cpu"
model="pnn"
run_name="transfer_amazon_pnn_m-${max_match}"

mkdir -p ../log/${run_name}/running-logs/

for ti in $(seq 0 1 14)
do
  command="../scripts/transfer.py --device=${device} -ti=${ti} -tc=15 -rn=${run_name} -bn=bn -mo=${model} -mm=${max_match} -lr=0.01 -ma=random -rands=${random_seed} -uas=0 -imf=${init_model_fp} -ds=amazon --mapping_fp=${mapping_fp} -tdf=${train_data_fd} -edf=${eval_data_fd} -eulp=${examine_user_list_fp} -aup=${examine_user_list_fp}"
  echo "python -u ${command} > ../log/${run_name}/running-logs/${ti}.txt 2>&1 &"
  python -u ${command} > ../log/${run_name}/running-logs/${ti}.txt 2>&1 &
done
