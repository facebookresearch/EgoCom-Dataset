# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# Computes accuracy of transcription models using 1 - word error rate (wer)
# wer uses the Wagner-Fischer Algorithm to compute the Levenstein distance at
# both the sentence and word level.
# To use this package, please install jiwer: "pip install jiwer"
# To use this package, please install num2words: "pip install num2words"

from __future__ import (
    print_function, absolute_import, division,
    unicode_literals, with_statement,  # Python 2 compatibility
)
import numpy as np
import re
import pandas as pd
from jiwer import wer
# This package converts things like "42" to "forty-two"
from num2words import num2words
# For parallel processing
import multiprocessing as mp

max_threads = mp.cpu_count()

SUPPRESS_WARNINGS = False

filler_words = ["huh", "uh", "uhh", "uhhh", "erm", "er", "um", "umm", "mm",
                "hmm", "hm", "ah", "ahh", "ugh", "eyyy", "eyy", "ey", "ay",
                "ayy", "ayyy"]  # "oh", "ohh "
# Words introduced by rev.com erroneously without [] around them.
filler_words += ["inaudible", "crosstalk"]
contractions = {
    "d": "would",  # could also be had, should, etc.
    "ve": "have",
    "s": "is",
    "m": "am",
    "ll": "will",
    "t": "not",
    "re": "are",
}


def process_transcript_data(
        df,
        verbose=True,
        replace_numbers_with_words=True,
        remove_spaces=True,
        remove_filler_words=True,
        remove_capitalization=True,
):
    """Pre-processing for transcripts for normalization and tokenization.

    Parameters
    ----------
    df: pd.DataFrame
      cols ["conversation_id", "startTime", "speaker_id", "endTime", "word"]"""

    if verbose:
        print("Original length |", len(df))

    # If a row contains a word with spaces, split into two rows in the
    # dataframe.
    lod = df.to_dict('records')  # List of dictionaries
    df = pd.DataFrame([dict(d, **{"word": token}) for d in lod for token in
                       d["word"].split(" ")])
    if verbose:
        print("After splitting words with spaces into seperate rows |", len(df))

    # Replace empty strings with spaces. This is caused by using word.split("
    # ") when word == " "
    df["word"] = df["word"].apply(lambda x: " " if x == "" else x)
    if verbose:
        print("After replacing empty strings with spaces |", len(df))

    # Remove rows with duplicate spaces (word == " ") in dataframe.
    df = df[(df['word'].shift(1) != df['word']) | (df['word'] != " ")]
    if verbose:
        print("After removing duplicate rows containing only spaces |", len(df))

    # Seperate punctuation, numbers, and non-numeric words. Seperates
    # contractions, too. e.g. "1900s." becomes ["1900", "s", "."] e.g.
    # contractions like "they've" ---> ["they", "'", "ve"]
    lod = df.to_dict('records')  # List of dictionaries
    df = pd.DataFrame([dict(d, **{"word": token}) for d in lod for token in (
        [" "] if d["word"] == " " else re.findall(r"[^\W\d_]+|\d+|[^\w\s]",
                                                  d["word"]))])
    if verbose:
        print("After 1900s. --> [1900, s, .] and they've --> [they, ', ve] |",
              len(df))

    if replace_numbers_with_words:
        # Change all numbers to word-based versions of the number, e.g. ()
        df['word'] = df['word'].apply(
            lambda x: num2words(int(x)) if x.isdigit() else x)
        if verbose:
            print("After 1100 --> one thousand, one hundred |", len(df))

        # Seperate numbers with dashes into parts, e.g. "twenty-two" becomes
        # ["twenty", "-", "two"] Also seperate numbers with spaces like "one
        # hundred" --> ["one", "hundred"]
        lod = df.to_dict('records')  # List of dictionaries
        df = pd.DataFrame([dict(d, **{"word": token}) for d in lod for token in
                           ([" "] if d["word"] == " " else re.findall(
                               r"\w+|[^\w\s]", d["word"]))])
        if verbose:
            print("After twenty-two --> twenty two |", len(df))

    if remove_spaces:
        df = df[~df['word'].str.isspace()]
        if verbose:
            print('After removing spaces |', len(df))

    if remove_capitalization:
        df['word'] = df['word'].str.lower()
        if verbose:
            print('After removing capitalization |', len(df))

    if remove_filler_words:
        df = df[df["word"].apply(lambda x: not x in filler_words)]
        if verbose:
            print('After removing filler words |', len(df))

    return df


def df2transcripts(
        df,
        remove_actions=True,  # (laughs)
        remove_punctuation=True,
        expand_contractions=True,
):
    """Convert large dataframe with all video transcripts into a
    dict of dataframes for each video using the video names as dict keys

    Returns a dict of dataframes for each video
    using the video names as dict keys"""

    # Create dict of dataframes for each video.
    dfs = {i: v for i, v in df.groupby('conversation_id')}

    transcripts = {}
    for key in sorted(dfs.keys()):
        words = dfs[key]['word']
        if expand_contractions:
            words = words.apply(
                lambda x: contractions[x] if x in contractions else x)
        transcript = " ".join(words)
        if remove_actions:
            transcript = re.sub("[\(\[].*?[\)\]]", "", transcript)
        if remove_punctuation:
            transcript = re.sub(r'[^\w\s]', '', transcript)
        # Reduce spaces to single space
        transcript = re.sub(r'\s+', ' ', transcript)

        transcripts[key] = transcript

    return transcripts


def create_processed_transcripts(
        df,
):
    """Converts large dataframe with all video transcripts into a
    dict of dataframes for each video using the video names as dict keys.

    Unlike df2transcripts(), this function automatically performs all the
    preprocessing in process_transcript_data() for you and makes no assumptions
    on the input df except that it contains columns:
    ["conversation_id", "startTime", "speaker_id", "endTime", "word"]

    Returns a dict of dataframes for each video
    using the video names as dict keys"""

    processed_df = process_transcript_data(df)
    return df2transcripts(processed_df)


# Analyze word-error-rate (wer) results in parallel on all cores.
def _run_thread_job(params):
    try:
        key = params['key']
        ts_t = params['truth']  # Transcription string of ground truth
        ts_h = params[
            'hypothesis']  # An automatically estimated transcription string
        error = wer(ts_t, ts_h)
        print(key, "|", round(1 - error, 3))
        return error

    except Exception as e:
        if not SUPPRESS_WARNINGS:
            warnings.warn('ERROR in thread' + str(
                mp.current_process()) + "with exception:\n" + str(e))
            return None


def _parallel_param_opt(lst, threads=max_threads):
    pool = mp.Pool(threads)
    results = pool.map(_run_thread_job, lst)
    pool.close()
    pool.join()
    return results


def compute_wer_for_all_videos(ts_dict_hypothesis, ts_dict_truth):
    """Takes two transcript dictionaries (first is estimated, second is
    ground truth). Each are formatted as {'key1':transcript_str1,
    'key2':transcript_str2, ...}

    Returns a dictionary mapping the keys of the hypothesis dictionary to the
    word-error-rates for the associated transcript."""

    print("\nTranscription accuracy for each video")
    print("-------------------------------------")
    keys = ts_dict_hypothesis.keys()
    jobs = []
    for key in sorted(keys):
        jobs.append({
            "key": key,
            "hypothesis": ts_dict_hypothesis[key],
            "truth": ts_dict_truth[key],
        })
    results = _parallel_param_opt(jobs)
    print("Average Accuracy (1 - word-error-rate):",
          np.round(1 - np.mean(results), 3))
    return dict(zip(keys, results))


def compute_duration_total_weighted_error(error_dict, transcript_len_dict):
    '''Every error in error_dict maps to a video of some length.
    The error captures the transcription wer error over that video.
    So when we average over all the videos, the error needs to be weighted by
    the length of the number of transcripts in each video.'''
    df_len_err = pd.DataFrame(pd.Series(transcript_len_dict, name='len')).join(
        pd.Series(error_dict, name='err')
    )
    df_len_err["len"] /= sum(df_len_err["len"])
    return np.dot(df_len_err["len"], df_len_err['err'])


def error_as_percent_acc(error):
    '''Input error is a float. Prints an accuracy % as a str.'''
    return str(100 * np.round(1 - error, 4))[:5] + "%"
