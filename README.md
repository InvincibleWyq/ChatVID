# ChatVID
Chat about anything on any video!

## 

## Gradio Example
Video Quick View:

[![The Temple Of Heaven](https://img.youtube.com/vi/9xLoyYY_5rc/0.jpg)](https://www.youtube.com/watch?v=9xLoyYY_5rc "The Temple Of Heaven")
https://imgur.com/a/Vv9dZQD
<img width="1637" alt="image" src="https://github.com/Go2Heart/ChatVID/assets/71871209/f696591e-0fb7-40c4-bc92-d221c3aa6ca5">

<img width="1624" alt="image" src="https://github.com/Go2Heart/ChatVID/assets/71871209/92659f68-0a32-4c3e-979d-047b3a94de36">


## Install Instructions
```
pip install -r requirements.txt

mim install mmengine mmcv mmaction2
# Install ffmpeg for Whisper.
# Note that if Whisper encounters permission errors, you may need to specify the DATA_GYM_CACHE_DIR to your writable cache directory.


```
<!-- # change the scenic/dataset_lib/video_ops.py and scenic/train_lib_deprecated/train_utils.py -->

## Setting Up Checkpoints
### Grit Checkpoints
Put [Grit](https://datarelease.blob.core.windows.net/grit/models/grit_b_densecap_objectdet.pth) into `pretrained_models` folder in the ChatVID's root folder.

<!-- 
### BLIP2 Checkpoints
```
It will be acquired automatically using Hugging Face's transformers
```
### Whisper Checkpoints
```

```
 -->



### Vicuna Weights

ChatVID uses frozen Vicuna 7B and 13B models. Please first follow the [instructions](https://github.com/lm-sys/FastChat) to prepare Vicuna v1.1 weights. 
Then modify the `vicuna.model_path` in the [Infer Config](https://github.com/Go2Heart/ChatVID/blob/master/config/infer.yaml) to the folder that contains Vicuna weights.

### Vid2Seq Checkpoints

1. Prepare CLIP first for feature extraction in Vid2Seq.
Get the CLIP [Checkpoints](). Specify the `vid2seq.clip_path` in the [Infer Config](https://github.com/Go2Heart/ChatVID/blob/master/config/infer.yaml) to the checkpoint path. 
`vid2seq.output_path` is used to store the generated TFRecords and can be specified to any writable directory. 
`vid2seq.work_dir` is the Flax's working directory and can be specified to any writable directory.



2. Prepare Vid2Seq ActivityNet Checkpoints
Get the Vid2Seq ActivityNet [Checkpoints](https://storage.googleapis.com/scenic-bucket/vid2seq/anet-captions). And then rename it as `checkpoint_200000`. After that, change the `vid2seq.checkpoint_path` in the  [Infer Config](https://github.com/Go2Heart/ChatVID/blob/master/config/infer.yaml) to the folder directory where contains the checkpoint.



## Gradio WebUI Usage
```
python demo.py --config config/debug.yaml
```

