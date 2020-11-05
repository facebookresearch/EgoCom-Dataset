# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# This library supports functionality needed for global transcription.
# The global transcription method works by considering transcribed words that
# co-occur from different sources, near in time (less than 0.1 seconds)
# within a conversation, and only keeps the one with the max confidence score,
# thus identifiying the speaker for that word.
# This library supports:
# * Automatic generation of subtitles
# * Finding consecutive values in a list
# * Identifying duplicate words in a pd.DataFrame within a time window threshold
# * Identify duplicates to remove in a pd.DataFrame, unveiling the speaker

from __future__ import (
    print_function, absolute_import, division,
    unicode_literals, with_statement,
)  # Py2 compatibility

import os
import pickle
import pandas as pd
from datetime import datetime, timedelta
from itertools import groupby
import numpy as np


# In[2]:


def async_srt_format_timestamp(seconds):
    seconds = float(seconds)
    stamp = str(timedelta(seconds=seconds))
    first, second, third = stamp.split(":")
    third = "{:.3f}".format(float(third))
    sec, milisec = third.split(".")
    third = ",".join([sec.zfill(2), milisec])
    return ":".join([first.zfill(2), second.zfill(2), third])


def write_subtitles(
        words,
        start_times,
        end_times,
        speakers=None,
        max_words=15,
        wfn=None,
        max_sub_duration=6,
        max_words_per_line=5,
):
    if wfn is not None:
        with open(wfn, 'w') as f:
            f.write('')
        wf = open(wfn, 'a')
    else:
        wf = None

    cnt = 0
    i = 0
    while i < len(words) - 1:
        cnt += 1
        # Advance i as long as not too much time passes
        for j in range(i, i + min(max_words, len(words) - i)):
            if start_times[j] - start_times[i] >= max_sub_duration:
                break
        start_time = start_times[i]
        end_time = end_times[:j][-1]
        w = words[i:j]
        if speakers is None:
            word_str = " ".join(w[:len(w) // 2]) + "\n" + " ".join(
                w[len(w) // 2:])
        else:
            s = speakers[i:j]
            who, word_idx = compute_running_consecutive_idx(s)
            word_str = "\n".join([("Speaker " + str(who[z])
                if who[z] > 1 else "Curtis") + ": " + "\n\t".join(
                [" ".join(w[word_idx[z]:word_idx[z + 1]][
                          i:min(i + max_words_per_line, j)]) for i in
                 range(0, word_idx[z + 1] - word_idx[z], max_words_per_line)])
                                  for z in range(len(who))])
        print(cnt, file=wf)
        print(async_srt_format_timestamp(start_time), "-->",
              async_srt_format_timestamp(end_time), file=wf)
        print(word_str, file=wf)
        print(file=wf)

        # Increment start (i) to end (j) of current subtitle
        i = j


# In[3]:


def compute_running_consecutive_idx(lst):
    '''Returns two lists, the first is a list of the consecutive values,
    and the second list is their associated starting indices. Feel free to
    zip the two lists together as a single list of tuples if you desire. '''
    consec = [(k, sum(1 for i in g)) for k, g in groupby(lst)]
    val, idx = list(zip(*consec))
    idx = np.cumsum([0] + list(idx))
    return list(val), list(idx)


def compute_duplicates_mask(df, threshold=0.1):
    '''This function returns a list of True/False boolean values, true
    whenever a pair of identical words occurs within the threshold (in seconds)
    of eachother, by different speakers.
    
    This is a helper function for find_which_duplicates_to_remove()
    You may find it helpful to read that docstring as well.'''

    combined_filter = None
    for offset in [0, -1]:
        # Create a filter to determine if two adjacent words started within
        # threshold seconds.
        intertimes = np.ediff1d(df['startTime'])
        close_start = (
                abs(np.insert(intertimes, offset, 1)) <= threshold + 1e-6)
        # Create a filter to determine if two adjacent words ended within 0.1
        # seconds.
        intertimes = np.ediff1d(df['endTime'])
        close_end = (abs(np.insert(intertimes, offset, 1)) <= threshold + 1e-6)
        # Combine filters
        near_filter = close_start | close_end
        # Create a filter that checks if the speaker is different
        intertimes = np.ediff1d(df['speaker'])
        diff_speaker = (np.insert(intertimes, offset, 1) != 0)
        # Create a filter that checks if the word is the same
        intertimes = np.ediff1d(df['word'].apply(lambda x: hash(x)))
        same_word = (np.insert(intertimes, offset, 1) == 0)
        # Combine filters
        same_word_diff_speaker = same_word & diff_speaker
        both = near_filter & same_word_diff_speaker
        combined_filter = \
            both if combined_filter is None else combined_filter | both

    return combined_filter


def find_which_duplicates_to_remove(df, threshold_seconds=0.1):
    '''This identifies when the same word is picked up and transcribed by
    multiple speaker's microphones, even though it was only spoken once,
    by a single speaker. The additional duplicate words are removed from the
    pandas.DataFrame containing the transcriptions.
    
    The duplicate with the highest confidence is the word we keep.
    
    It will not remove duplicates spoken by the same speaker - for example
    "Okay, okay let's go" or "trains go choo choo" -- those examples will be
    kept in the transcriptions because the duplicated words "okay" and "choo"
    belong to the same speaker.
    
    An issue can occur if two speakers both legitimately say, for example,
    "Okay" within the time threshold (default is 0.1 seconds) of eachother.
    In this rare case, a legitimate spoken word may be removed.
    
    Returns a list of True/False boolean values with true whenever a word
    should be removed. '''

    if "duplicates" not in df.columns:
        df["duplicates"] = compute_duplicates_mask(df,
                                                   threshold=threshold_seconds)

    remove_mask = []
    prev_word = ""
    c_lst = []
    for c, d, w in [tuple(z) for z in
                    df[["confidence", "duplicates", "word"]].itertuples(
                            index=False)]:
        same_word_as_prev = prev_word == w
        if d:
            if same_word_as_prev:  # Same word group
                c_lst.append(c)
            else:
                remove_mask = remove_mask + [z != max(c_lst) for z in c_lst]
                c_lst = [c]
        else:
            remove_mask = remove_mask + [z != max(c_lst) for z in c_lst] + [
                False]
            c_lst = []

        prev_word = w
    # Added the last group which gets missed since we don't add until we
    # detect a new group, and there's no new group after the last group.
    remove_mask = remove_mask + [z != max(c_lst) for z in c_lst]

    return remove_mask
