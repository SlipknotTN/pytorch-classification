# PyTorch Image Classification Train+Test

Tested with python3 and PyTorch 1.4.0.

## Training instructions

Required steps:
- Split dataset training directory in train and validation subsets
- Every train and validation subset must contain a directory for each class

The directory *config* contains training parameters for each model.
You only need to pass a different config to the scripts to change model and hyperparameters.

### Training

#### Training examples

Run script from *pytorch* directory.

SqueezeNet 1.1:

```
python3 ./tools/train.py
--dataset_train_dir ../../dataset/trainval/train
--dataset_val_dir ../../dataset/trainval/val
--config_file ./config/sqnet.cfg
--model_output_path ./export/sqnet.pth
```

Custom model:

TODO

Export directory will contains this models:
- PyTorch format (single pth file with model weights).


### [Optional] Run predictions on test directory

This script runs predictions on dogs vs cats test images and exports results in a CSV format ready for kaggle submission.
Please notice that while the competition is over, you can still evaluate your model through [challenge page](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/leaderboard).

#### Test examples

Run script from *pytorch* directory.

FIXME: Export generic CSV with predictions for each class