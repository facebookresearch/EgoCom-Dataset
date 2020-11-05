# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# audio.py supports the following:
# playing audio files using sounddevice library
# plotting audio with axis capturing time information
# normalization, smart-clipping audio within a range, reducing audio peaks
# extracting audio tracks (as numpy arrays) from MP4 files.
# quantization (max_pooling, average_pooling, median_pooling)
# Denoising and identifying noise and removing clicks
# computing signal2noise ratio statically and dynamically
# simple cosine and butterworth bandpass filtering

from __future__ import (
    print_function, absolute_import, division,
    unicode_literals, with_statement,  # Python 2 compatibility
)

import os
import subprocess
from matplotlib import pyplot as plt
import numpy as np
from scipy.io import wavfile
import sounddevice as sd
import datetime
import time
from scipy import fftpack
from scipy.signal import butter, sosfilt


def read_wav(filename):
    samplerate, wav = wavfile.read(filename)
    return samplerate, norm_center_clip(wav)


def compute_wav_seconds(wav, samplerate):
    return np.arange(0, len(wav) / float(samplerate), 1 / float(samplerate))


def plot_wav(
        wav,
        samplerate=44100,
        wav_seconds=None,
        figsize=(25, 5),
        downsample=1,
        **kwargs,
):
    if wav_seconds is None:
        if downsample == 1:
            wav_seconds = compute_wav_seconds(wav, samplerate)
        else:
            assert (downsample >= 1)
            len_seconds = len(wav) / float(samplerate)
            wav_seconds = np.linspace(0, len_seconds, np.ceil(
                len_seconds * samplerate / float(downsample)))

    fig = plt.figure(figsize=figsize)
    lines = plt.plot(wav_seconds, wav[::downsample], **kwargs)
    _ = plt.xlabel('Time (s)', fontsize=20)
    return plt.gca(), lines


def play(wav, samplerate, make_fig=False, playbar_width=166,
         num_seconds_to_play=5):
    wav = wav[:num_seconds_to_play * samplerate]
    wav_seconds = compute_wav_seconds(wav, samplerate)
    if make_fig:
        plot_wav(wav, samplerate, wav_seconds, figsize=(25, 5))
    sd.play(wav, samplerate)
    seconds = len(wav) / float(samplerate)
    window = 0.1  # how often to print out in seconds
    for s in np.arange(0, seconds, window):
        num_stars = int(playbar_width * s / float(seconds)) - 1
        status = "[" + "*" * num_stars + " " * (
                    playbar_width - num_stars - 4) + "]"
        time_elapsed = str(datetime.timedelta(seconds=s))
        time_elapsed = time_elapsed[
                       :-5] if "." in time_elapsed else time_elapsed + ".0"
        print("\r▻", time_elapsed + ": " + status, end="", flush=True)
        time.sleep(window)
    print()


def norm_center_clip(wav, bottom=-1., top=1.):
    """Makes the wav have max = top, min = bottom, and mean = top - bottom."""
    # Normalize between bottom and top values
    wav = normalize_between(wav.astype(float), bottom=bottom, top=top)
    # Center vertically around halfway between bottom and top.
    wav = wav - np.mean(wav)
    # Clip between bottom and top values
    wav = smart_clip(wav, bottom=bottom, top=top)
    return wav


def normalize_between(wav, bottom=-1., top=1.):
    # Normalize amplitude between -1 and 1
    return (wav.astype(float) - np.min(wav)) / (
                (np.max(wav) - np.min(wav)) / (top - bottom)) + bottom


def smart_clip(wav, bottom=-1., top=1.):
    """clips wav within range while maintaining shape (no flat-tops)"""
    min_val = float(np.min(wav))
    max_val = float(np.max(wav))
    clipped_wav = np.array(wav).astype(float)
    clipped_wav[clipped_wav < bottom] = clipped_wav[clipped_wav < bottom] / (
                min_val / bottom)
    clipped_wav[clipped_wav > top] = clipped_wav[clipped_wav > top] / (
                max_val / top)
    return clipped_wav


def normalize_and_reduce_peaks(wav, bottom=-1, top=1, look_above_value=0.9,
                               frac_above=.0005):
    """Looks at the fraction of wav points ABOVE "look_above_value",
    and if its greater than "frac_above",
    smart_clips those points down to maximum of "look_above_value".
    Then looks at the fraction of wav points BELOW (-1)*"look_above_value",
    and if its greater than "frac_above",
    smart_clips those points down to minimum of (-1)*"look_above_value".
    Repeats this procedure until the wav is no longer changing.

    This method will force the input wav to be normalized using
    norm_center_clip(wav, bottom = -1, top = 1),
    thus the wav has max = 1, min = -1, and centered around zero.

    Guarantees to return a wav (1d numpy array) with the form
    (min=-1,max=1,mean=0)"""

    still_changing = True
    while still_changing:

        wav = norm_center_clip(wav)

        still_changing = False
        frac_of_wav = np.sum(wav > look_above_value) / float(len(wav))
        if frac_of_wav < frac_above:  # < frac wav above look_above_value
            wav = smart_clip_top(wav, top=look_above_value)
            still_changing = True

        frac_of_wav = np.sum(wav < -1 * look_above_value) / float(len(wav))
        if frac_of_wav < frac_above:  # < frac_wav below look_above_value
            wav = smart_clip_bottom(wav, bottom=-1 * look_above_value)
            still_changing = True

    wav = norm_center_clip(wav)

    return wav


def smart_clip_bottom(wav, bottom=-1):
    """clips wav to range while maintaining shape (no flat-bottomed curves)"""
    min_val = float(np.min(wav))
    clipped_wav = np.array(wav).astype(float)
    clipped_wav[clipped_wav < bottom] = clipped_wav[clipped_wav < bottom] / (
                min_val / bottom)
    return clipped_wav


def smart_clip_top(wav, top=1):
    """clips wav within range while maintaining shape (no flat-tops)"""
    max_val = float(np.max(wav))
    clipped_wav = np.array(wav).astype(float)
    clipped_wav[clipped_wav > top] = clipped_wav[clipped_wav > top] / (
                max_val / top)
    return clipped_wav


# In[4]:


def get_samplerate_wav_from_list_of_mp4_fns(fn_list, n_sec=None,
                                            normalize_wav=True):
    """fn_list is a list of .mp4 videos with stereo audio (two-channel).
    Returns a list of samplerates and list of numpy arrays containing wav audio.

    Parameters
    ----------
    n_seconds : int
        Number of seconds of audio to fetch for each mp4, starting from 0."""

    samplerate_list = []
    wav_list = []

    for fn in fn_list:
        # Seperate audio from video and create .wav audio file
        s = " -t 00:" + str(n_sec // 60) + ":" + str(
            n_sec % 60) if n_sec is not None else ""
        cmd = "ffmpeg -i '{f}'{s} '{f}'.wav".format(f=fn, s=s)
        subprocess.getoutput(cmd)

    for fn in [x + ".wav" for x in fn_list]:
        # Get samplerate and audio numpy arrays
        samplerate, wav = wavfile.read(fn)
        if normalize_wav:
            wav = norm_center_clip(wav)
        wav_list.append(wav)
        samplerate_list.append(samplerate)

        # Delete .wav file
        os.remove(fn)

    return samplerate_list, wav_list


def avg_pool_1d(arr, pool_size=5, filler=True, weights=None):
    new_len = len(arr) if filler else int(np.ceil(len(arr) / float(pool_size)))
    result = np.ones(new_len)
    for i, idx in enumerate(range(0, len(arr), pool_size)):
        chunk = arr[idx:idx + pool_size]
        j = i * pool_size if filler else i
        if weights is not None and len(weights) == len(chunk):
            result[j:j + len(chunk)] = np.dot(chunk,
                                              weights)  # Weighted average
        else:
            result[j:j + len(chunk)] = np.mean(chunk)
    return result


def max_pool_1d(arr, pool_size=5, filler=True):
    new_len = len(arr) if filler else int(np.ceil(len(arr) / float(pool_size)))
    result = np.ones(new_len)
    for i, idx in enumerate(range(0, len(arr), pool_size)):
        chunk = arr[idx:idx + pool_size]
        j = i * pool_size if filler else i
        result[j:j + len(chunk)] = np.max(chunk)
    return result


def median_pool_1d(arr, pool_size=5, filler=True):
    new_len = len(arr) if filler else int(np.ceil(len(arr) / float(pool_size)))
    result = np.ones(new_len)
    for i, idx in enumerate(range(0, len(arr), pool_size)):
        chunk = arr[idx:idx + pool_size]
        j = i * pool_size if filler else i
        result[j:j + len(chunk)] = np.median(chunk)
    return result


def upsample_1d(arr, length, pool_size):
    return np.array([y for z in arr for y in [z] * pool_size])[:length]


def align_two_wav_arrays(
        wav1,
        wav2,
        samplerate1,
        samplerate2=None,
        window_start_seconds=10,
        window_end_seconds=30,
        verbose=False,
):
    window_end_seconds = min(
        len(wav1) / float(samplerate1),
        len(wav2) / float(samplerate2),
        window_end_seconds,
    )

    # Take right channel if wav is more than one channel
    if len(wav1.shape) > 1 and wav1.shape[1] > 1:
        wav1 = wav1[:, 1]
    if len(wav2.shape) > 1 and wav2.shape[1] > 1:
        wav2 = wav2[:, 1]

    if samplerate2 is None:
        samplerate2 = samplerate1
        print(window_end_seconds)

    start1 = int(samplerate1 * window_start_seconds)
    end1 = int(samplerate1 * window_end_seconds)
    start2 = int(samplerate2 * window_start_seconds)
    end2 = int(samplerate2 * window_end_seconds)
    A = fftpack.fft(wav1[start1:end1])
    B = fftpack.fft(wav2[start2:end2])
    Ar = -A.conjugate()
    Br = -B.conjugate()
    time_shift_ab = np.argmax(np.abs(fftpack.ifft(Ar * B)))
    time_shift_ba = np.argmax(np.abs(fftpack.ifft(A * Br)))

    if verbose:
        print(time_shift_ab, time_shift_ba)
        print((end1 - start1) / samplerate1, 'second window',
              time_shift_ab / float(end1 - start1))
        if time_shift_ab < time_shift_ba:
            print('Advancing second wav file by', time_shift_ab)
        else:
            print('Advancing first wav file by', time_shift_ba)

    #     if time_shift_ab < time_shift_ba:
    #         return 'advance_second_wav_param_by', time_shift_ab
    #     return 'advance_first_wav_param_by', time_shift_ba

    if time_shift_ab < time_shift_ba:
        return wav1, wav2[time_shift_ab:], min(time_shift_ab, time_shift_ba)
    return wav1[time_shift_ba:], wav2, min(time_shift_ab, time_shift_ba)


def cosine_filter(wav):
    wav_filter = np.cos(np.linspace(-0.5 * np.pi, 0.5 * np.pi, num=len(wav)))
    return (wav_filter * wav.T).T


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y


def power(x):
    """Returns average squared value of numpy array x."""
    if np.isnan(x).any():
        return np.nan
    if len(x) == 0:
        return np.nan  # nothing to average
    return np.mean(x.astype('float') ** 2)


def snr_static(signal, noise):
    #     return 10 * np.log10(power(signal) - power(noise))
    return 10 * np.log10((power(signal) - power(noise)) / power(noise))


def static_signal2noise_ratio(wav, samplerate=None, threshold=0.12):
    """Computes the relative power of signal above some threshold to signal
    below. Assumes that noise is static (doesn't change through the course of
    the signal).

    Increase the threshold if you believe the amount of noise is very high. A
    good default for typical speech with light background noise is 0.07. """
    if type(wav) == str:
        samplerate, wav = read_wav(wav)
    assert (type(wav) is np.ndarray)

    snr = np.nan
    while np.isnan(snr).any() and threshold <= 0.99:
        likely_noise = wav[
            find_noise(wav, samplerate, intensity=96, threshold=threshold)]
        likely_signal = wav[
            ~find_noise(wav, samplerate, intensity=99.99, threshold=threshold)]
        snr = snr_static(likely_signal, likely_noise)
        threshold += 0.05
    return snr


def find_noise(wav, samplerate, intensity=99.9, make_fig=False, threshold=0.12):
    """Returns a boolean mask of the locations of pure static noise in the
    wav file. intensity should be between 80 and 99.99999. The higher,
    the more noise will be removed, but speech may be removed, too. The
    lower, the less noise will be removed but the more accurate the removal
    will be. The threshold is multiplied by the intensity (percentile) of the
    max and everything below that is considered noise. Increase the threshold
    to increase noise found, but you risk including speech signal. """

    # First do a quick pass to remove large deviating clicks.
    abs_of_median_pooled_wav = median_pool_1d(np.abs(wav),
                                              pool_size=samplerate // 200)

    # Pool every 0.05 seconds.
    wav_pooled = max_pool_1d(abs_of_median_pooled_wav,
                             pool_size=samplerate // 20)
    # Anything less than threshold percent of the intensity percentile is
    # detected as noise
    cutoff = np.percentile(wav_pooled, intensity) * threshold
    noise_mask = wav_pooled < cutoff
    if make_fig:
        wav_seconds = np.arange(0, len(wav) / float(samplerate),
                                1 / float(samplerate))
        plt.figure(figsize=(20, 5))
        plt.plot(wav_seconds, wav, alpha=0.5)
        plt.plot(wav_seconds, wav_pooled, alpha=0.5)
        plt.scatter(wav_seconds[noise_mask], wav[noise_mask], color='crimson')
        plt.show()
    return noise_mask


def find_clicks(wav, samplerate, bin_size_in_seconds=0.3):
    click_mask = np.zeros(len(wav), dtype=bool)
    step = int(samplerate * bin_size_in_seconds)
    fraction_zero = (step - np.count_nonzero(wav[0:step])) / float(step)
    prev = fraction_zero > 0.5
    for s in np.arange(step, len(wav), step):
        fraction_zero = (step - np.count_nonzero(wav[s:s + step])) / float(step)
        cur = fraction_zero > 0.5
        if prev and cur:
            click_mask[s - step:s] = True
        prev = cur
    return click_mask


def make_odd(x):
    return x if x % 2 == 1 else x + 1


def denoise_wav(wav, samplerate=None, intensity=99.9, threshold=0.12,
                bin_size_in_seconds=0.3):
    """OUTPUT: return (samplerate, wav)
    Returns the wav file with noise and clicks removed.
    """
    if type(wav) == str:
        samplerate, wav = read_wav(wav)
    assert (type(wav) is np.ndarray)

    wav = np.array(wav)
    noise_mask = find_noise(wav, samplerate, intensity=intensity,
                            threshold=threshold)
    wav[noise_mask] = 0
    click_mask = find_clicks(wav, samplerate,
                             bin_size_in_seconds=bin_size_in_seconds)

    wav[noise_mask | click_mask] = 0

    return wav
