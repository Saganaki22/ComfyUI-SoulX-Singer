#!/bin/bash

script_dir=$(dirname "$(realpath "$0")")
root_dir=$(dirname "$script_dir")

cd $root_dir || exit
export PYTHONPATH=$root_dir:$PYTHONPATH

device=cuda


####### Run Prompt Annotation #######
audio_path=example/audio/zh_prompt.mp3
save_dir=example/transcriptions/zh_prompt
language=Mandarin
vocal_sep=False
max_merge_duration=30000

python -m preprocess.pipeline \
    --audio_path $audio_path \
    --save_dir $save_dir \
    --language $language \
    --device $device \
    --vocal_sep $vocal_sep \
    --max_merge_duration $max_merge_duration


####### Run Target Annotation #######
audio_path=example/audio/music.mp3
save_dir=example/transcriptions/music
language=Mandarin
vocal_sep=True
max_merge_duration=60000

python -m preprocess.pipeline \
    --audio_path $audio_path \
    --save_dir $save_dir \
    --language $language \
    --device $device \
    --vocal_sep $vocal_sep \
    --max_merge_duration $max_merge_duration