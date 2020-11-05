# EgoCom: A Multi-person Multi-modal Egocentric Communications Dataset
This is the release of the EgoCom Dataset associated with the [T-PAMI paper](https://ieeexplore.ieee.org/document/9200754) entitled "EgoCom: A Multi-person Multi-modal Egocentric Communications Dataset".

![](assets/f1.png)

![](assets/f23.png)

![](assets/gogloo_glasses.png)

# Citing EgoCom
If you use this package or the EgoCom Dataset in your work, please cite:

    @ARTICLE{9200754,
      author={Curtis G. {Northcutt} and Shengxin {Zha} and Steven {Lovegrove} and Richard {Newcombe}},
      journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
      title={EgoCom: A Multi-person Multi-modal Egocentric Communications Dataset}, 
      year={2020},
      volume={},
      number={},
      pages={1-12},
      doi={10.1109/TPAMI.2020.3025105}}
    
# Download the EgoCom Dataset
[[TODO]]

**`egocom`** is a Python package for handling egocentric video across multiple participants supporting libraries for audio, transcription, alignment, source separation, NLP, language modeling, video processing, and more. This package was used to create the EgoCom dataset here: https://our.intern.facebook.com/intern/wiki/LiveMaps/EgoCom/

The package is structured into two parts: (1) core libraries in egocom/egocom/ and (2) code using each of those libraries is located in egocom/examples/. The package structure and contents are as follows:


## **egocom libraries (in egocom/egocom/):**

* `multi_array_alignment.py`
    - Library for automatic multi-array alignment.  
    - For our purpose, we are aligning stereo audio wavs in the form of numpy arrays. The content of the audio is conversation, with multiple conversation participants. We are aligning audio from microphones near each of the conversation participants. Thus the sources are mixed, but each person is loudest in their own microphone.
    - This library works generally for any alignment problem and does not require audio data, although this is the benchmark dataset that this library was tested on. In particular, this library was used to automatically align the EgoCom dataset. It does not require any loud constant sound for alignment. It works by locally normalizing each audio file so that all speakers are the same volume, then finds the shifts that maximize the correlation relative to one of the arrays.
* `audio.py`
    * Library supports numerous general audio processing methods including:
        * playing audio files using sounddevice library
        * plotting audio with axis capturing time information
        * normalization, smart-clipping audio within a range, reducing audio peaks
        * extracting audio tracks (as numpy arrays) from MP4 files.
        * quantization (max_pooling, average_pooling, median_pooling)
        * Denoising and identifying noise and removing clicks
        * computing signal2noise ratio statically and dynamically
        * simple cosine and butterworth bandpass filtering
* `transcription.py`
    * library for producing automatic global transcription plus speaker identification, where the input is multiple observed audio signals (stereo in our case, but mono works as well) coming from each of multiple speakers.
    * The automatic multi-speaker transcription algorithm is very simple! It looks at same transcribed words that occur from different sources, near in time (less than 0.1 seconds) within a conversation, and only keeps the one with the max confidence score, thus identifying the speaker for that word.
    * This library supports
        - Automatic generation of subtitles
        - Finding consecutive values in a list
        - Identifying duplicate words in a pandas DataFrame within a time window threshold
        - Identify duplicates to remove in pandas DataFrame, unveiling the speaker
* `word_error_rate_analysis.py`
    * This library is used for computing the accuracy of transcription models using 1 - word error rate (wer).
        * wer computation uses the Wagner-Fischer Algorithm to compute the Levenstein distance at both the sentence and word level.
        
## egocom examples using libraries (in egocom/examples):

* **EgoCom production code and examples (examples/EgoCom_dataset)**
    * `alignment_example.py`
        * An example demonstrating how to automatically align audio for multi-perspective egocentric convefsation data using the alignment package to align hundreds of wav files from the EgoCom dataset.
    * `spectrogram_from_audio.py`
        * This is a simple example of creating a spectrogram for an audio file.
    * `computing_wer_accuracy/compute_wer.py`
        * This file uses the word_error_rate_analysis package to compute the accuracy of our global transcriptions methods on EgoCom.
    * `data_parsing/`
        * `mundane_egocom_video_tasks/`
            * scripts for aligning, trimming, and parsing the EgoCom dataset.
        * `auto_align_egocom_dataset.py`
            * Example script showing how we automatically aligned the entire EgoCom dataset - no human aid necessary.
        * `generate_raw_audio_dataset.py`
            * This is how the /EgoCom/raw_audio dataset is generated.
            * Extracts raw audio of EgoCom dataset from original source
        * `rev_create_videos_with_combined_audio.py`
            * This script creates the videos to be transcribed by humans via the vendor rev.com.
            * All this script is doing is combining the video from person 1's perspective with the summed audio from all perspectives.
    * `source_seperation/source_seperation.py` (mispelled but that's the current name)
        * Work in progress script for performing source separation for each conversation in the EgoCom dataset. We try two approaches (1) ICA, and (2) only keeping the signal with max frequency for each time window. Neither reveal the sources. A new approach is in progress. Code is marked DO-NOT-USE.
    * `transcription/`
        * Scripts for obtaining Google/Rev ground truth/and automatic transcriptions for EgoCom
        * `auto_global_transcription_methods.py`
            * This is the code used to execute our global transcription methods. It relies heavily on the egocom/transcription.py library.
        * `create_ground_truth_transcriptions_from_rev.py`
            * This script is used to extract the HUMAN GROUND TRUTH TRANSCRIPTIONS from [rev.com](http://rev.com/) servers via HTTP FETCH requests and create the EgoCom/ground_truth_transcriptions.csv
        * `google_speech2text_transcription.py`
            * This file uses Google Speech to Text to transcribe all of the EgoCom audio (as well as ICA source estimates).
            * This script uses smart GET/FETCH HTTP protocols to asynchronously transcribe many audio files in parallel. It repeatedly queries Google's servers, always pushing the max limit of parallel requests it can take, and waits automatically when needed.
    
# Team
Curtis G. Northcutt, Shengxin Cindy Zha, Steven Lovegrove, and Richard Newcombe

# Contact
Curtis G. Northcutt, curtis@chipbrain.com
Steven Lovegrove, stevenlovegrove@fb.com

# License
Copyright (c) 2018-2021 Facebook Inc. Released under a modified MIT License. See [LICENSE](LICENSE) for details.
