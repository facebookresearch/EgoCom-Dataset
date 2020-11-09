# Copyright (c) Facebook, Inc. and its affiliates.

#!/usr/bin/env bash

# Example execution:
# download_egocom.sh egocom240p .

# Abort on error
set -e

datasets="egocom1080p_uncompressed, egocom720p, egocom480p, egocom240p, egocom_pretrained_features, egocom_audio_only"

if [[ "$1" == "" || "$2" == "" ]]; then
  echo "usage: $0 <dataset name> <directory>"
  echo "  where <dataset name> is one of $datasets"
  exit 1
fi

if [ "$1" == "egocom1080p_uncompressed" ]; then
	parts1=f
	parts2=v
elif [ "$1" == "egocom720p" ]; then
	parts1=b
	parts2=w
elif [ "$1" == "egocom480p" ]; then
	parts1=a
  	parts2=q
elif [ "$1" == "egocom240p" ]; then
	parts1=a
  	parts2=f
elif [ "$1" == "egocom_pretrained_features" ]; then
	parts1=b
  	parts2=a
elif [ "$1" == "egocom_audio_only" ]; then
	parts1=a
  	parts2=j
else
	echo "Invalid dataset '$1', must be one of $datasets!"
	exit 1
fi

echo "Downloading and decompressing EgoCom to $2. The script can resume"
echo "partial downloads -- if your download gets interrupted, simply run it again."

for p1 in $(eval echo "{a..$parts1}"); do
	if [ "$p1" == "$parts1" ]; then
  	p2max=$parts2
  else
  	p2max=z
  fi
  for p2 in $(eval echo "{a..$p2max}"); do
  	# Ensure files are continued in case the script gets interrupted halfway through
  	wget --continue https://github.com/facebookresearch/EgoCom-Dataset/releases/download/v1.0/$1.tar.gz.${p1}${p2}
	done
done

# Create the destination directory if it doesn't exist yet
mkdir -p $2

cat $1.tar.gz.?? | unpigz -p 32  | tar -xvC $2
