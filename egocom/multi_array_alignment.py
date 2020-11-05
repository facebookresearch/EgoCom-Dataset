# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# # This library is used to align multiple arrays. 
# Here, we align stereo audio wavs in the form of numpy arrays.
# Audio is presumed to be humans talking in conversation,
# with multiple conversation participants.
# We align audio from microphones near each of the conversation participants.
# Sources are mixed, but each person is loudest in their own microphone.
# This library works generally for any alignment problem and does not require
# audio data, although this is the benchmark dataset that this library was
# tested on. In particular, this library was used to automatically align the
# EgoCom dataset. It does not require any loud constant sound for alignment.
# It works by locally normalizing each audio file so that all speakers are the
# same volume, then finds the shifts that maximize the correlation relative to
# one of the arrays.


from __future__ import print_function, absolute_import, division, unicode_literals, with_statement # Python 2 compatibility

import numpy as np
from skimage.feature import register_translation
from itertools import combinations
from scipy.io import wavfile
from egocom import audio


# In[2]:


def gaussian_kernel(kernel_length=100, nsigma=3):
    '''Returns a 2D Gaussian kernel array.
    
    Parameters
    ----------
    kernel_length : int
        The length of the returned array.
        
    nsigma : int
        The # of standard deviations around the mean to compute the Gaussian shape.'''
    
    from scipy.stats import norm
    
    interval = (2*nsigma+1.)/(kernel_length)
    x = np.linspace(-nsigma-interval/2., nsigma+interval/2., kernel_length+1)
    kern1d = np.diff(norm.cdf(x))
    kernel_raw = np.sqrt(kern1d)
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

def norm_signal(arr, samplerate = 44100, window_size = 0.1, also_return_divisor = False):
    '''Returns a locally-normalized array by dividing each point by a the 
    sum of the points around it, with greater emphasis on the points 
    nearest (using a Guassian convolution)
    
    Parameters
    ----------
    arr : np.array
    samplerate : int
    window_size : float (in seconds)
    
    Returns
    -------
    A Guassian convolution locally normalized version of the input arr'''
    
    kern = gaussian_kernel(kernel_length=int(samplerate * window_size), nsigma=3)
    local_power = np.convolve(arr, kern, 'same')
    resp = arr / local_power
    return resp, local_power if also_return_divisor else resp


# In[3]:


def verify_alignments_for_three_wavs(
    shift_wav1_to_wav2,
    shift_wav2_to_wav3,
    shift_wav1_to_wav3,
    nearness_in_seconds = 0.1,
    samplerate = 44100,
):
    '''Verifies that alignment results agree for three wavs
    e.g. shift from wav1 to wav2 + shift from wav2 to wav3
    should be near shift from wav1 to wav 3'''
    
    threshold = samplerate * nearness_in_seconds
    assert(abs(shift_wav1_to_wav2 + shift_wav2_to_wav3 - shift_wav1_to_wav3) < 
        threshold)


# In[4]:


def align_wavs(wav_list, samplerate = 44100, samples_at_end_to_ignore = 10):
    '''Automatically aligns a list of stereo (2-channel) wav np.arrays'''
    
    num_wavs = len(wav_list)
    # Avoid artifacts that may exist in the last samples_at_end_to_ignore samles
    length = min(len(w) for w in wav_list) - samples_at_end_to_ignore
    # Make all wav files the same length
    wavs = [abs(z)[:length] for z in wav_list]
    # Normalize locally
    wavs = [np.apply_along_axis(lambda x: norm_signal(x), axis = 0, arr = z) for z in wavs]
    # Normalize globally
    wavs = [audio.norm_center_clip(z) for z in wavs]
    
    shifts_relative_to_first_wav = [0]
    for w1, w2 in combinations(wavs, 2):
        # Compute the shifts for all combinations of left/right audio streams from both wav files
        combs = [(0, 0), (0, 1), (1, 0), (1, 1)]
        shifts = [-1 * register_translation(w1[:, a], w2[:, b])[0][0] for a,b in combs]
        shift = int(np.median(shifts))
        if len(shifts_relative_to_first_wav) < num_wavs:
            shifts_relative_to_first_wav.append(shift)
        elif num_wavs == 3:
            verify_alignments_for_three_wavs(
                shift_wav1_to_wav2 = shifts_relative_to_first_wav[-2],
                shift_wav2_to_wav3 = shift,
                shift_wav1_to_wav3 = shifts_relative_to_first_wav[-1],
            )
    alignment = np.array(shifts_relative_to_first_wav) - min(shifts_relative_to_first_wav)
    
    return alignment


# In[5]:


def create_combined_wav_audio_sample(
    wav_list, 
    samplerate = 44100,
    alignment = None,
    wfn = "output.wav", # WriteFileName
    nbits = 16, 
    force_mono = False,
):
    '''Combines the wav files after aligning 
    so you can listen and see if they are aligned.'''
    
    if alignment is None:
        alignment = [0] * len(wav_list)
    # Align wav files
    aligned_wavs = [wav[alignment[i]:] for i, wav in enumerate(wav_list)]
    # Make all wav files normalized and the same length.
    duration = min([len(w) for w in aligned_wavs])
    y = sum([audio.norm_center_clip(z[:duration]) for z in aligned_wavs])
    write_wav(y, samplerate, wfn, nbits, force_mono)


# In[6]:


def write_wav(
    wav, 
    samplerate = 44100,
    wfn = "output.wav", # WriteFileName
    nbits = 16, 
    force_mono = False,
):
    '''Writes a wav file to directory wfn'''
    # Normalize and reduce to mono if needed -- required by Google Speech-to-Text
    y = audio.norm_center_clip(wav.sum(axis=1) if force_mono else wav)
    # Set bitsize of audio. 
    y_int = ((2**(nbits - 1) - 1) * y).astype(eval("np.int" + str(nbits)))
    # Write file to the WriteFileName specified by wfn
    wavfile.write(wfn, samplerate, y_int)

