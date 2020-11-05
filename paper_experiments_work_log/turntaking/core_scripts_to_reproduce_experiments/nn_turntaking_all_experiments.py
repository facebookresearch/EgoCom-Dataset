# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# This is the core script used for all experiments in the Turn-taking prediction
# of the EgoCom paper. This script requires the preprocessed hd5 named as:
# 'egocom_feature_data_normalized_history_{}_future_{}_binary.hdf5'
# These are already pre-computed in the Dataset release, but can be recomputed
# via `nn_preprocess_data_for_training`

# Example calls of this script are provided in `turntaking_script_examples.bash`

# Imports
# Data processing
import pickle
import pandas as pd
import numpy as np
# Multiprocessing
import itertools
import torch
import torch.nn as nn
# Need these to use EgoCom validation dataset
from skorch.helper import predefined_split
from skorch.dataset import Dataset
from skorch.callbacks import Checkpoint
# Other imports
import datetime
import os

import argparse
# Used to parse command line arguments
parser = argparse.ArgumentParser(description='EgoCom Turn-Taking Prediction')
parser.add_argument('--param-idx', default=0, type=int,
                    help='Specifies which block of hyper parameters'
                         'you will train here, among the num-param-blocks.'
                         'The index is zero-based.'
                         'e.g. if num-param-blocks is 4, then this value can'
                         'be 0, 1, 2, or 3.')
parser.add_argument('--num-param-blocks', default=1, type=int,
                    help='If you want to break up hyper-param optimization'
                         'across multiple GPUs, specify the number of GPUs'
                         'here and be sure to also specify --param_idx'
                         'which specifies which block of the parameters'
                         'you will train here, among the num-param-blocks.')
parser.add_argument('--use-all-perspectives', default=False,
                    action='store_true',
                    help='Only applies for binary prediction tasks.'
                         'If True, combine all three speakers synchronized'
                         'perspectives, such that at each second of data,'
                         'all three speakers features are present. This'
                         'effectively widens the training data to three'
                         'times but reduces the number of data points'
                         'by a third')
parser.add_argument('--predict-only-host', default=False, action='store_true',
                    help='Only applies for binary prediction tasks.'
                         'If True, only predict the hosts labels using either'
                         '(1) only the host data or (2) the combined data'
                         'if use_all_perspectives == True.')
parser.add_argument('--include-prior', default=None, type=str,
                    help='By default (None) this will run train both a model'
                         'with a prior and a model without a prior.'
                         'Set to "true" to include the label of the current'
                         'speaker when predicting who will be speaking in the'
                         'the future. Set to "false" to not include prior label'
                         'information. You can think of this as a prior on'
                         'the person speaking, since the person who will be'
                         'speaking is highly related to the person who is'
                         'currently speaking.')
parser.add_argument('--prediction-task', default='binary', type=str,
                    help='Set to "multi" to predict the label of the person'
                         'who will be speaking the future, a multi-class task.'
                         'Set to "binary" to predict if a given speaker'
                         'will be speaking in the future.')
parser.add_argument('--epochs', default=20, type=int,
                    help='Number of epochs for training.')
parser.add_argument('--use-crossval', default=False, action='store_true',
                    help='Optimize hyper-parameters with cross-validation.'
                         'This script **no longer** supports cross-validation'
                         'because EgoCom has a predefined test set.'
                         'Never set this to True. Included for compatibility.')
parser.add_argument('--seed', default=0, type=int,
                    help='Seed for stochastic code for reproducibility.')

# Extract argument flags
args = parser.parse_args()
param_idx = args.param_idx
num_param_blocks = args.num_param_blocks
use_all_perspectives = args.use_all_perspectives
predict_only_host = args.predict_only_host
if args.include_prior is None:
    include_prior_list = [True, False]
elif args.include_prior.lower() is 'true':
    include_prior_list = [True]
elif args.include_prior.lower() is 'false':
    include_prior_list = [False]
else:
    raise ValueError('--include prior should be None, "true", or "false')
prediction_task = args.prediction_task
epochs = args.epochs
use_crossval = args.use_crossval
seed = args.seed

# Make sure flag values are valid.
assert (use_all_perspectives, predict_only_host) in [
        (True, True), (False, True), (False, False)]

# PyTorch and Skorch imports needed based on prediction task.
if prediction_task == 'multi':
    from skorch import NeuralNetClassifier
elif prediction_task == 'binary':
    from skorch import NeuralNetBinaryClassifier
else:
    assert args.prediction_task in ['binary', 'multi']

flag_settings = {
    'param_idx': param_idx,
    'num_param_blocks': num_param_blocks,
    'use_all_perspectives': use_all_perspectives,
    'predict_only_host': predict_only_host,
    'include_prior': include_prior_list,
    'prediction_task': prediction_task,
    'epochs': epochs,
    'use_crossval': use_crossval,
    'seed': seed,
}
print('Running with settings:', flag_settings)

# Location where dataset and pre-processed data is stored.
egocom_loc = "/datasets/cgn/EGOCOM/egocom_features/no_audio/"
# Seed everything for reproducibility
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
# Make sure GPU can be used with PyTorch
device = torch.device('cuda')
# # Data preparation
video_info = pd.read_csv("/datasets/cgn/EGOCOM/video_info.csv")
kinds = ['text', 'video', 'voxaudio'] # ['audio', 'text', 'video', 'voxaudio']
# cols = pd.read_csv(egocom_loc + "egocom_features_history_4sec.csv.gz").columns
with open(egocom_loc + 'feature_column_names.p', 'rb') as rf:
    cols = pickle.load(rf)
# Generate all combinations of "kinds" of features (i.e. modalities).
experiments = list(
    itertools.chain.from_iterable(itertools.combinations(kinds, r)
                                  for r in range(len(kinds)+1))
)[1:]
experiments = {
    "_".join(e): [c for c in cols if c.split("_")[0] in [z+"feat" for z in e]]
    for e in experiments
}
binary_prior_feature = 'prior_is_speaking'
multiclass_prior_feature = 'prior_multiclass_speaker_label'
label_shifts = [0, 2, 4, 9]  # translates to [1,3,5,10] seconds respectively
histories = [4, 5, 10, 30]


def shift_labels(x, shift = 0):
    """Move labels forward by advance_labels and removes the extra data.

    Parameters
    ----------
    x : pandas DataFrame
      contains features and labels for a ['video_id', 'video_speaker_id'] group
    shift : int
      how far to advance the labels in time (seconds)."""
    if x.name == 'multiclass_speaker_label' or x.name == 'is_speaking':
        return x[shift:].values
    else:
        return x[:-shift].values if shift > 0 else x.values


def remove_prior(dataframe, include_prior):
    """If include_prior is False. Drops the prior features from dataframe."""
    # The prior is included in the data by default.
    if not include_prior:
        print('Removing prior speaking labels from training features.')
        return dataframe.drop(
            [c for c in dataframe.columns if 'prior' in c],
            axis=1,
        )
    return dataframe


def prepare_multiclass_data_from_preprocessed_hdf5(
    experiment_key,
    history,
    future,
    include_prior,
):
    """Produce X_train, X_test, Y_train, Y_test from a preprocessed
    hdf5 file storing the data. Data is already z-score normalized,
    per-column.
    Use this when prediction_task == 'multi' """

    assert prediction_task == 'multi'
    hdf5_fn = 'egocom_feature_data_normalized_history_{}_future_{}_binary' \
              '.hdf5'.format(history, future)
    experiment = experiments[experiment_key]
    new_data = pd.read_hdf(egocom_loc + hdf5_fn, key=hdf5_fn)
    new_data.dropna(inplace=True)  # Remove NaN values if they exist.
    # Include prior features if part of this experiment
    if include_prior:
        experiment += [multiclass_prior_feature]
        x = new_data[multiclass_prior_feature]  # Z-score normalize prior
        new_data[multiclass_prior_feature] = (x - x.mean()) / x.std()
    # Only use 3 speaker conversations because we are going to combine the
    # features from all 3 in multi-class setting and we need all examples
    # to have the same number of features.
    new_data = new_data[new_data['num_speakers'] == 3]

    # Combine all three speakers to single input for each conversation
    gb_cols = ["conversation_id", "clip_id", "multiclass_speaker_label",
               "test", "train", "val"]
    X = new_data.groupby(gb_cols).apply(
        lambda x: x[experiment].values.flatten()).reset_index()
    # Only include examples with all three speakers (same dimension)
    # this occurs at the end of a video (if one speaker's video is ~1s
    # longer than the others) where there are features for that
    # speaker but not the other speakers.
    input_length = max([len(z) for z in X[0]])
    mask = [len(z) == input_length for z in X[0]]
    X = X[mask]

    if use_crossval:  # Cross-validation
        y = X['multiclass_speaker_label'].values
        X = np.stack(X[0])
        return X, y
    else:  # Use EgoCom pre-defined test set (preferred for reproducibility)
        x_train = np.stack(X[X['train']][0])
        x_test = np.stack(X[X['test']][0])
        x_val = np.stack(X[X['val']][0])
        y_train = X[X['train']]['multiclass_speaker_label'].values
        y_test = X[X['test']]['multiclass_speaker_label'].values
        y_val = X[X['val']]['multiclass_speaker_label'].values
        # Convert to cuda (load to GPU)
        x_train = torch.from_numpy(x_train.astype(np.float32))
        x_test = torch.from_numpy(x_test.astype(np.float32))
        x_val = torch.from_numpy(x_val.astype(np.float32))
        y_train = torch.from_numpy(y_train)
        y_test = torch.from_numpy(y_test)
        y_val = torch.from_numpy(y_val)
        return x_train, y_train, x_test, y_test, x_val, y_val


def prepare_binary_data_from_preprocessed_hdf5(
    experiment_key,
    history,
    future,
    include_prior,
):
    """Produce X_train, X_test, Y_train, Y_test from a preprocessed
    hdf5 file storing the data. Data is already z-score normalized,
    per-column.
    Use this when prediction_task == 'binary'."""

    assert prediction_task == 'binary'
    # Make sure use_all_perspectives and predict_only_host bools are valid combo
    if use_all_perspectives and not predict_only_host:
        print("ERROR!")
        print("ERROR: use_all_perspectives = True,"
              "predict_only_host = False is impossible.")
        print("ERROR!")

    hdf5_fn = 'egocom_feature_data_normalized_history_{}_future_{}_binary' \
              '.hdf5'.format(history, future)
    experiment = experiments[experiment_key]
    new_data = pd.read_hdf(egocom_loc + hdf5_fn, key=hdf5_fn)
    new_data.dropna(inplace=True)  # Remove NaN values if they exist.
    # Include prior features if part of this experiment
    if include_prior:
        experiment += [binary_prior_feature]
        x = new_data[binary_prior_feature]  # Z-score normalize prior feature
        new_data[binary_prior_feature] = (x - x.mean()) / x.std()
    if use_all_perspectives:
        # Only use 3 speaker conversations
        new_data = new_data[new_data['num_speakers'] == 3]
    elif predict_only_host:
        # Only use host data and none of the other perspectives.
        new_data = new_data[new_data['speaker_is_host']]

    if use_all_perspectives:
        # Combine all three speakers to single input for each conversation
        X = new_data.groupby(
            ["conversation_id", "clip_id", "multiclass_speaker_label",
             "test", "train", "val"]).apply(
            lambda x: x[experiment].values.flatten()).reset_index()
        if predict_only_host:
            # Only use hosts labels.
            y = new_data.groupby(["conversation_id", "clip_id",
                                  "test", "train", "val"])[
                'multiclass_speaker_label'].apply(
                lambda x: x.iloc[0] == 1)
        else:
            print("ERROR!")
            print("ERROR: use_all_perspectives = True,"
                  "predict_only_host = False is impossible.")
            print("ERROR!")
        # Only include examples with all three speakers (same dimension)
        # this occurs at the end of a video (if one speaker's video is ~1s
        # longer than the others) where there are features for that
        # speaker but not the other speakers.
        input_length = max([len(z) for z in X[0]])
        mask = [len(z) == input_length for z in X[0]]
        X = X[mask]
        y = y[mask]
        if use_crossval:
            X = np.stack(X[0])
            y = y.values
        else:
            x_train = np.stack(X[X['train']][0])
            x_test = np.stack(X[X['test']][0])
            x_val = np.stack(X[X['val']][0])
            y_train = y[y.reset_index()['train'].values].values
            y_test = y[y.reset_index()['test'].values].values
            y_val = y[y.reset_index()['val'].values].values
    else:
        if use_crossval:
            X = new_data[experiment].values
            # Get labels
            y = new_data['is_speaking'].values
        else:
            x_train = new_data[new_data['train']][experiment].values
            x_test = new_data[new_data['test']][experiment].values
            x_val = new_data[new_data['val']][experiment].values
            y_train = new_data[new_data['train']]['is_speaking'].values
            y_test = new_data[new_data['test']]['is_speaking'].values
            y_val = new_data[new_data['val']]['is_speaking'].values
    # Convert to cuda (load to GPU)
    x_train = torch.from_numpy(x_train.astype(np.float32))
    y_train = torch.from_numpy(y_train.astype(np.float32))
    x_test = torch.from_numpy(x_test.astype(np.float32))
    y_test = torch.from_numpy(y_test.astype(np.float32))
    x_val = torch.from_numpy(x_val.astype(np.float32))
    y_val = torch.from_numpy(y_val.astype(np.float32))
    if use_crossval:
        return X, y
    else:
        return x_train, y_train, x_test, y_test, x_val, y_val


# # Training MLP (wrapped in Skorch for sklearn bindings
# Fully connected neural network with two hidden layers
class NNClassifier(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_layer_sizes=(500, 20),
        dropout=0.8,
        output_size=1,
    ):
        super(NNClassifier, self).__init__()
        # Each block contains a relu, dropout, and linear layer.
        blocks = []
        # Append a block for each hidden layer.
        for b in range(1, len(hidden_layer_sizes)):
            blocks.append(
                nn.Sequential(
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_layer_sizes[b - 1]),
                    nn.Dropout(p=dropout),
                    nn.Linear(hidden_layer_sizes[b - 1], hidden_layer_sizes[b]),
                )
            )
        # Build the actual neural network here with input, hidden, and output.
        self.seq = nn.Sequential(
            # Input layer
            nn.Linear(input_size, hidden_layer_sizes[0]),
            # Hidden layers
            nn.Sequential(*blocks),
            # Output layer
            nn.ReLU(),
            nn.BatchNorm1d(hidden_layer_sizes[-1]),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_layer_sizes[-1], output_size),
            # Softmax (multi-class) or Sigmoid (binary) not needed.
            # These final layers are already in the loss criterion.
        )
        # Enable the neural network to work on GPU.
        self.seq.cuda()
    
    def forward(self, x):
        return self.seq(x)

# SGD settings that worked well for multiclass: {'wd': 0.01, 'initial_lr': 0.1,
# 'eta_min': 1e-06, 'num_hidden_layers': 1, 'dropout': 0.2, 'momentum': 0.9}
# Adam settings that worked well for all experiments: {'wd': 0.001,
# 'initial_lr': 0.0005, 'num_hidden_layers': 1, 'dropout': 0.1, 'momentum': 0.9}
# Set-up hyper-parameters
# Modify (if needed) based on type of classification being performed.
wd_list = [0.001]  # [0.01, 0.001, 0.0001]:
initial_lr_list = [1e-3, 3e-3, 5e-3]  # [.052]  #
eta_min_list = [None]  # .038
num_hidden_layers_list = [1]
dropout_list = [0.1, 0.5]  #  0.1,
momentum_list = [0.9]
amsgrad_list = [False, True]
if prediction_task == 'binary':  # Binary classification
    prepare_data_from_preprocessed_hdf5 = \
        prepare_binary_data_from_preprocessed_hdf5
    NeuralNetwork = NeuralNetBinaryClassifier
    criterion = nn.modules.loss.BCEWithLogitsLoss
    output_size = 1  # Only one class because output is a sigmoid p(y=1)
elif prediction_task == 'multi':  # Multi-class classification
    prepare_data_from_preprocessed_hdf5 = \
        prepare_multiclass_data_from_preprocessed_hdf5
    NeuralNetwork = NeuralNetClassifier
    criterion = torch.nn.modules.loss.CrossEntropyLoss
    output_size = 4  # speaker 1, 2, 3, and none ==> 4 classes


def get_parameter_settings():
    """Create all combinations of parameter settings."""
    param_settings = []
    for include_prior in include_prior_list:
        for history in histories:
            for label_shift in label_shifts:
                for experiment_key in experiments.keys():
                    for wd in wd_list:
                        for initial_lr in initial_lr_list:
                            for eta_min in eta_min_list:
                                for num_hidden_layers in num_hidden_layers_list:
                                    for dropout in dropout_list:
                                        for momentum in momentum_list:
                                            for amsgrad in amsgrad_list:
                                                param_settings.append({
                                                    "include_prior":
                                                        include_prior,
                                                    "history": history,
                                                    "future": label_shift + 1,
                                                    "experiment_key":
                                                        experiment_key,
                                                    "wd": wd,
                                                    "initial_lr": initial_lr,
                                                    "eta_min": eta_min,
                                                    "num_hidden_layers":
                                                        num_hidden_layers,
                                                    "dropout": dropout,
                                                    "momentum": momentum,
                                                    "amsgrad": amsgrad,
                                                })
    return param_settings


settings = get_parameter_settings()
# Only test the group of settings as specified by command line arguments
length = len(settings)
block_size = length // num_param_blocks
settings = settings[block_size * param_idx: block_size * (param_idx + 1)]

print('Evaluating settings: {} to {}, of {} settings'.format(
    block_size * param_idx, block_size * (param_idx + 1), length))

start_time = datetime.datetime.now()
previous_data_info = None
for cnt, setting in enumerate(settings):
    # Seed everything
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    # Load data if the experiment data needed changes
    data_info = (setting['experiment_key'], setting['history'],
                 setting['future'], setting['include_prior'])
    if data_info != previous_data_info:
        previous_data_info = data_info
        x_train, y_train, x_test, y_test, x_val, y_val = \
            prepare_data_from_preprocessed_hdf5(*data_info)
        print(x_train.shape, y_train.shape, x_test.shape, y_test.shape,
              x_val.shape, y_val.shape)
    if cnt > 0:
        # Print time elapsed, est time remaining, and progress.
        time_elapsed = datetime.datetime.now() - start_time
        total_time = time_elapsed / cnt * len(settings)
        print("Time elapsed:", str(time_elapsed).split(".")[0])
        print("Time remaining:", str(total_time - time_elapsed).split(".")[0])
        print("Progress: {:.2%}".format(cnt / len(settings)))
    print(setting)
    # Prepare neural network for training
    hidden_layer_sizes = tuple(x_train.shape[1] // (i+1)**2
                               for i in range(setting['num_hidden_layers']))
    # # LR scheduler used with SGD.
    # schedule = LRScheduler(
    #     policy=lr_scheduler.CosineAnnealingLR,
    #     T_max=epochs * (5 / 4),  # lr = 0.04 for max learning with audio_text
    #     eta_min=setting['eta_min'],
    # )

    model_id = start_time.strftime('%Y%m%d%H%M%S')
    base_checkpoint_dir = "/home/cgn/skorch_checkpoints/"
    cp = Checkpoint(
        dirname=base_checkpoint_dir + model_id,
        f_params="nn.pt",  # Name of checkpoint parameters saved.
    )
    # Set-up neural network for training.
    model = NeuralNetwork(  # Skorch NeuralNetClassifer
        module=NNClassifier,
        callbacks=[
            cp,
            # ('lr_scheduler', schedule),  # Use with SGD optimizer
        ],
        lr=setting['initial_lr'],
        module__input_size=x_train.shape[1],
        module__hidden_layer_sizes=hidden_layer_sizes,
        module__dropout=setting['dropout'],
        module__output_size=output_size,
        criterion=criterion,
        optimizer=torch.optim.Adam,  # torch.optim.SGD,
        batch_size=128,
        warm_start=False,
        verbose=2,
        device='cuda',
        train_split=predefined_split(Dataset(x_val, y_val)),  # holdout val set
        optimizer__weight_decay=setting['wd'],
        # optimizer__momentum=setting['momentum'],  # Use with SGD optimizer
        optimizer__amsgrad=setting['amsgrad'],
    )

    if use_crossval:
        # This script is no longer intended to use cross-validation. Please use
        # the provided val and test sets in EgoCom.
        pass
    else:
        model.fit(x_train, y_train, epochs=epochs)
    print(" * Test Acc (last epoch): {:.6f}".format(
        model.score(x_test, y_test)))
    model.load_params(checkpoint=cp)
    print(" ** Test Acc (best val): {:.6f}".format(model.score(x_test, y_test)))

    # Clean-up (remove) checkpoints
    os.remove(base_checkpoint_dir + model_id + "/nn.pt")
    os.remove(base_checkpoint_dir + model_id + "/optimizer.pt")
    os.remove(base_checkpoint_dir + model_id + "/history.json")
    os.rmdir(base_checkpoint_dir + model_id)

