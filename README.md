# EgoCom:A Multi-person Multi-modal Egocentric Communications Dataset

Egocentric Communications (EgoCom) is a first-of-its-kind natural conversations dataset containing multi-modal human communication data captured simultaneously from the participants' egocentric perspectives. The EgoCom dataset includes 38.5 hours of conversations comprised of synchronized embodied stereo audio and egocentric video along with 240,000 ground-truth, time-stamped word-level transcriptions and speaker labels from 34 diverse speakers.

This is the release of the EgoCom Dataset associated with the [T-PAMI paper](https://ieeexplore.ieee.org/document/9200754) entitled "EgoCom: A Multi-person Multi-modal Egocentric Communications Dataset".

![](assets/f1.png)

![](assets/f23.png)

![](assets/gogloo_glasses.png)

## Citing EgoCom
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
    
## Download the EgoCom Dataset
[[TODO]]


## The EgoCom Dataset provides Video, Audio, and Text Modalities

*   **Video**
    * 1080p @ 30 FPS, as well as 720p, 480p, 240p for faster downloading / processing)
*   **Audio**
    * binaural/stereo 2-channel mp4a/aac, 44,100 samples/second,
    64-bit)=
*   **Text**
    * Transcripts (by human experts) including speaker identification, start time-stamp, end
    time-stamp)
    * Available here: [/egocom_dataset/ground_truth_transcriptions.csv](https://github.com/facebookresearch/EgoCom-Dataset/blob/main/egocom_dataset/ground_truth_transcriptions.csv)
*   **Metadata for each video**
    * Available here: [/egocom_dataset/video_info.csv](https://github.com/facebookresearch/EgoCom-Dataset/blob/main/egocom_dataset/video_info.csv)
    * Included features:
        - video_id             (*int64*)
        - conversation_id      (*str*)
        - video_speaker_id     (*int64*)
        - num_speakers         (*int64*)
        - speaker_name         (*str*)
        - speaker_gender       (*str*)
        - duration_seconds     (*int64*)
        - word_count           (*int64*)
        - speaker_is_host      (*bool*)
        - tokenized_words      (*str*)
        - native_speaker       (*bool*)
        - video_name           (*str*)
        - background_fan       (*bool*)
        - background_music     (*bool*)
        - cid                  (*str*)
        - train                (*bool*)
        - val                  (*bool*)
        - test                 (*bool*)
    
## Dataset Specifications and Details

EgoCom contains 175 mp4 files which collectively comprise 39 conversations.
For half of the dataset (87 videos),
conversations are broken up into 5 minute clips. For the other half (88
videos) contains full conversations between 15 and 30 minutes. Every
conversation has three participants and either two or three recording
devices are worn in each conversation.

The following specifications are true for all video/audio/transcripts in
the EgoCom dataset.

-   Videos contain 3 speakers wearing glasses (2 in view, 1 is the
    camera perspective).
    -   Conversations are comprised of three speakers.
    -   Although all participants are wearing glasses, sometimes 1
        person is wearing regular, non-recording prescription glasses
-   Video/audio captured with Gogloo glasses.
    -   **Video Format**: RAW 1080P H.264 mp4, framerate of 30 FPS
        (camera located between eyes)
    -   **Audio Format**: RAW 2-channel (left and right), 64-bit
        mp4a/aac (extracted to wav), 44100 samples/second (recorded near
        ears)
    -   **Ground Truth Transcripts**: Human transcribed words. Although
        we use the term *ground truth*, some errors may exist due to
        human error. Stored as a .csv (267k rows) file with the
        following columns:
        -   key - identifies the video in which the word was spoken
        -   startTime - the time when the word started to be spoken.
        -   speaker - \[0,1,2\] identifies the speaker
        -   endTime - the time when the word finished being spoken.
        -   word - the word spoken
-   We provide the dataset in the following mp4 sizes
    -   1080p (1920x1080) uncompressed
    -   720p (1280x720)
    -   480p (640x480)
    -   240p (352x240)
    -   Example of how we perform the compression (720p in this example):

`ffmpeg -i input.mp4 -s 1280x720 -aspect 1280:720 -vcodec libx264 -crf 20  -threads 12 compressed.mp4`

-   Every conversation includes a host who directs topics.
-   All conversations are spoken in English (US).

## Topics of Conversation

Every conversation includes a host, Curtis Northcutt, who directs
topics. The dataset covers the following topics.

*   Teaching, learning, and understanding
*   Playing and teaching how to play card games like poker and blackjack
*   Playing Q/A AI-complete games (similar to games *Heads Up* and *Hedbanz*)
*   Pontification about topics and thought experiments.
*   Discussing things people like (favorite cities, food, seasons, sports, movies, etc)
*   Describing objects in the room spatially and qualitatively
*   Novel discovery question/answering about how things work
*   Interacting, discussing, and looking at mirrors

## Research Areas Enabled

The EGOCOM dataset opens up a diverse array of new research
opportunities. The combination of egocentric
video/transcripts/audio from multiple simultaneous and aligned
perspectives in natural conversation enables opportunities for the following research areas:


* **Artificial Intelligence** 
    -   General AI solutions for the "Heads Up" question-based word-guessing game, building knowledge graphs of objects and properties, game bots
    -   Throughout the dataset we ask questions like: "What's that called?", "What shape is the \<object\>?", "What color is the \<object\>?", etc.
* **Conversational predictive tasks**
    -   Automatic Question-Answering, predict when/who will speak next,
        lip-reading predict speech from video (without audio), etc.
* **Natural Language Processing and Understanding (NLP/NLU), and
    Automatic Speech Recognition (ASR)**
    -   2+ speaker 2+ sources complete conversation transcription with
        speaker identification, contextual transcription
* **Source separation**
    -   Multi-model source separation (combining audio and video
        inputs), audio-only source separation, cocktail party-problem
        solutions
* **Spatial estimation and beam-forming audio analysis**
    -   Speaker localization, head/body pose estimation, etc.
* **Conversation analysis**
    -   Semantic Analysis (Linguistics), Communication (modeling), etc.
* **Human learning, teaching, and pedagogical efficacy**
    -   Automatic identification of teaching styles, meta-understanding
        (understanding when a learner understands), casual inference




## The Egocentric Communications (EgoCom) Vision

To achieve artificial intelligence, we must first **accurately** capture
the **sensory input data** from which intelligence evolved.

-   **accurately** = egocentrically
-   **sensory input data** = for now, audio (near ears) and video (near
    eyes)

Intelligence evolved through our own egocentric sensory perspective, yet
(artificial) intelligence solutions often assume a nicely pre-processed
omniscient perspective. Egocentric data is data gathered from the human
perspective — as was the case in the evolution of human intelligence.


## Code Details

**`egocom`** is a Python package for handling egocentric video across multiple participants supporting libraries for audio, transcription, alignment, source separation, NLP, language modeling, video processing, and more. This package was used to create the EgoCom dataset here: https://our.intern.facebook.com/intern/wiki/LiveMaps/EgoCom/

The package is structured into two parts: (1) core libraries in `EgoCom-Dataset/egocom/` and (2) code using each of those libraries is located in egocom/examples/. The package structure and contents are as follows:




### **egocom libraries (in egocom/egocom/):**

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
        
### egocom examples using libraries (in egocom/examples):

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
    
## Team
Curtis G. Northcutt, Shengxin Cindy Zha, Steven Lovegrove, and Richard Newcombe

## Contact
Curtis G. Northcutt, curtis@chipbrain.com
Steven Lovegrove, stevenlovegrove@fb.com

## License
Copyright (c) 2018-2021 Facebook Inc. Released under a modified MIT License. See [LICENSE](LICENSE) for details.
