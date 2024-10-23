Public dataset evaluation for MPDA.

## Dependencies
See [requirements.txt](scripts/requirements.txt) for the dependent pip packages.

## Data Preprocess
First, please download the raw data from https://files.grouplens.org/datasets/movielens/ml-20m.zip 
and http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz.
Unzip and place the downloaded files in '/root/data/'.
Then, run the following commands for MovieLens data preprocessing:
```shell
echo "preprocess"
python scripts/preprocess/movielens/preprocess.py || exit

echo "split"
python scripts/preprocess/movielens/split.py || exit

echo "intersect"
python scripts/preprocess/movielens/get_users_with_train_and_test.py || exit
```
Run the following commands for Amazon data preprocessing:
```shell
echo "preprocess"
python scripts/preprocess/amazon/preprocess.py || exit

echo "split"
python scripts/preprocess/movielens/split.py -ifd={data_fd}/Amazon/Electronics_5/processed -ts=1385078400 || exit

data_pfd="{data_fd}/Amazon/Electronics_5/processed"
train_data_fd="${data_pfd}/ts=1385078400_train"
eval_data_fd="${data_pfd}/ts=1385078400_test"
examine_user_list_fp="${data_pfd}/ts=1385078400_user-intersect.json"

echo "intersect"
python scripts/preprocess/movielens/get_users_with_train_and_test.py -tfd=${train_data_fd} -tefd=${eval_data_fd} -ofp=${examine_user_list_fp} || exit
```

## Initial Model
The initial models (cloud models) are trained via [train_global_model.py](scripts/train_global_model.py).
The resulted models have been placed in [cloud_models](cloud_models) in the dataset-name_model-name format.

## Evaluation
Run [transfer.py](scripts/transfer.py) for evaluating MPDA and the baselines.
Use --help option to show the option list.

We run the scripts on the PAI platform, which itself handles parallel invocation and computation resource allocation.
For other users, we here provide the commands for running in PC environment in [commands](commands).
Each script corresponds to one run with a specific group of hyper-parameters.
E.g., [transfer_amazon-din_m-50.sh](commands/transfer_amazon_din_m-50.sh) corresponds to the run using DIN on Amazon Electronics dataset with 50 matched users.
For compatibility, the scripts specify CPU as the computing device, and you can change the "device" option for running on other devices.
Please spread the tasks to multiple GPUs for accelerating in practice.
The default setting in the scripts with 15 cpu tasks might cost unacceptably long time to complete.

## Visualizing
Please run [visualize](scripts/visualize.py) to produce the files necessary for visualizing, 
and use tensorboard to view the results.
