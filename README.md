Public dataset evaluation for MPDA.

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
The resulted models have been placed in [cloud_models](cloud_models) in the <dataset_name>_<model_name> format.

## Evaluation
Run [train_global_model.py](scripts/train_global_model.py) for evaluating MPDA and the baselines.
Use --help option to show the option list.
The script supports parallel launching via the --task_count and --task_index options for accelerating.

## Visualizing
Please run [visualize](scripts/visualize.py) to produce the files necessary for visualizing, 
and use tensorboard to view the results.
