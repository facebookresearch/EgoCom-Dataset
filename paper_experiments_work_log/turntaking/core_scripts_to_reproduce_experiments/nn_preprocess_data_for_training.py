# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# Creates the hd5 preprocessing egocom features for each future setting that are
# used by `nn_turntaking_all_experiments.py`

import pickle
import sys
import pandas as pd
import numpy as np
# Set this file path to the preprocessed features.
egocom_loc = "/datasets/cgn/EGOCOM/egocom_preprocessed_features/"
seed = 0
video_info = pd.read_csv("/datasets/cgn/EGOCOM/video_info.csv")
kinds = ['text', 'video', 'voxaudio']  # ['audio', 'text', 'video', 'voxaudio']
with open(egocom_loc + 'feature_column_names.p', 'rb') as rf:
    cols = pickle.load(rf)
label_shifts = [0, 2, 4, 9]  # translates to [1,3,5,10] seconds respectively
try:  # Allow user to specify history as input otherwise, pre-compute all.
    histories = [int(sys.argv[1])]
except IndexError as e:
    histories = [4, 5, 10, 30]
except Exception as e:
    histories = [4, 5, 10, 30]


def shift_labels(x, shift=0):
    """Move labels forward by shift and removes the extra data.

    Parameters
    ----------
    x : pandas DataFrame
      contains features and labels for a ['video_id', 'video_speaker_id'] group
    shift : int
      how far to advance the labels in time (seconds)."""
    if x.name == 'multiclass_speaker_label' or x.name == 'is_speaking':
        return x[shift:].values
    else:
        return x[:-shift].values if shift != 0 else x.values


def add_priors(sdf):
    """Shifts the data forward one second and uses the current labels as the
    prior. This is necessary because the labels are actually shift ahead of the
    data by 0.5 seconds (the label for time 0 corresponds to the average speaker
    from 0s to 1s). So to get the person speaking relative to the data, we move
    the data forward one second, relative to the labels. We do this for each, x
    which corresponds to a sub dataframe grouped by 'video_id' and
    'video_speaker_id'.

    Parameters
    ----------
    sdf - pd.DataFrame
      subdataframe from a groupby. The original dataframe should be contain
      both labels and features and the labels have not been shifted yet."""
    prior_is_speaking = sdf['is_speaking'][:-1].values
    prior_multiclass_speaker_label = sdf['multiclass_speaker_label'][:-1].values
    table = sdf[1:].reset_index(drop=True)
    table['prior_is_speaking'] = prior_is_speaking
    table['prior_multiclass_speaker_label'] = prior_multiclass_speaker_label
    return table


# Pre-compute normalized data for all history and futures.
# Add labels and priors (current label when predicting future label)
use_crossval = False
for history in histories:
    # Fetch and prepare data
    original_data = pd.read_csv(
        egocom_loc + "egocom_features_history_{}sec.csv.gz".format(history))
    # Pre-compute x_train, y_train, x_test, y_test, x_val, y_val
    for label_shift in label_shifts:
        np.random.seed(seed=seed)
        data = original_data.copy(deep=True)
        # -1 means no one is speaking. Change to 0 for training.
        data["multiclass_speaker_label"].replace(-1, 0, inplace=True)
        # Include video info and pre-sort data by speaker_id within each video.
        data = pd.merge(data, video_info, on=['video_id', 'video_speaker_id'])
        # Add priors to data.
        data = data.groupby(['video_id', 'video_speaker_id']).apply(
            add_priors).reset_index(drop=True)
        # Normalize for every column in the dataframe.
        for feature in ['textfeat', 'videofeat', 'voxaudiofeat']:
            # Fetch relevant columns
            cols = [c for c in data.columns if feature in c]
            # Z-score normalization
            stats = data[cols].describe()
            normed_feats = (data[cols] - stats.T['mean']) / stats.T['std']
            # Clamp between -3 and 3 for each row.
            normed_feats = normed_feats.apply(
                lambda x: [min(max(-3, z), 3) for z in x], axis='rows')
            # Redo z-score normalization now that values are clamped to -3,3 std
            stats = normed_feats.describe()
            normed_feats = (normed_feats - stats.T['mean']) / stats.T['std']
            data[cols] = normed_feats
        # Shift the labels as far in the future as defined by label_shift
        new_data = pd.concat(
            [sdf.apply(shift_labels, shift=label_shift) for i, sdf in
             data.groupby(['video_id', 'video_speaker_id'])], ignore_index=True)
        base = 'egocom_feature_data_normalized_history_{}_future_{}_binary.hdf5'
        fn = base.format(history, label_shift + 1)
        new_data.to_hdf(egocom_loc + fn, key=fn)