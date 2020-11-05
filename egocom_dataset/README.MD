# EgoCom Dataset

This folder contains metadata and transcriptions for the EgoCom Dataset.

File descriptions:
* `download_egocom.sh`
  - Shell script to download the EgoCom Dataset.
  - usage: $0 <dataset name> <directory> where <dataset name> is one of egocom1080p_uncompressed, egocom720p, egocom480p, egocom240p, egocom_pretrained_features
  - Example: `download_egocom.sh egocom720p .`
* `video_info.csv`
  - Provides essential metadata for each video in EgoCom (video name, conversation part, number of speakers, duration, transcripts, etc)
* `ground_truth_transcriptions.csv`
  - Provides the text transcriptions for each video in EgoCom.
* `speaker_labels/` 
    1. `raw_audio_speaker_labels.json` -- speaker labels obtained using maximum magnitude among perspectives in a conversation for every 1 second
    2. `rev_ground_truth_speaker_labels.json` -- speaker labels obtained by human annotation.
