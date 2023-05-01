#!/bin/bash
DATE=`date +%m-%d-%H:%M`

python3 video_chat.py --config_path ./config/debug.yaml 2>&1 | tee ./log/${DATE}.log
