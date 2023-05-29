
<h1 align="center">
 <img src="https://github.com/Go2Heart/ChatVID/assets/71871209/006c4820-824b-4268-b035-102949af1739" height=320/><br>
 ChatVID
</h1>
<h4 align="center">💬 Chat about anything on any video! 🎥</h4>

Our ChatVID combines the understanding ability of Large Language Models([Vicuna](https://vicuna.lmsys.org) for current version) and the sensing ablity of Vision Models([BLIP2](https://blog.salesforceairesearch.com/blip-2/), [GRiT](https://github.com/JialianW/GRiT), [Vid2Seq](https://antoyang.github.io/vid2seq.html)) and Audio Models([Whisper](https://openai.com/research/whisper)). Altogether, they cooperate a powerful capability to bring users the experience to talk about anything in a video.



## Gradio Example ✨
<h1 align="center">
<img src="https://github.com/Go2Heart/ChatVID/assets/71871209/346e5d92-9b64-48b6-8001-87e80386329a" alt="The Temple Of Heaven" class="center">
<img src="https://github.com/Go2Heart/ChatVID/assets/71871209/aa96f310-83e7-4f7f-9458-9adb1019338f" alt="Cook" class="center">
</h1>
<img width="1624" alt="image" src="https://github.com/Go2Heart/ChatVID/assets/71871209/f696591e-0fb7-40c4-bc92-d221c3aa6ca5">
<img width="1624" alt="image" src="https://github.com/Go2Heart/ChatVID/assets/71871209/92659f68-0a32-4c3e-979d-047b3a94de36">

## Install Instructions 💻

```
pip install -r requirements.txt

mim install mmengine mmcv mmaction2
```

You will also need to install [ffmpeg](https://ffmpeg.org) for Whisper. Note that if Whisper encounters permission errors, you may need to specify the DATA_GYM_CACHE_DIR to your writable cache directory.
<!-- # change the scenic/dataset_lib/video_ops.py and scenic/train_lib_deprecated/train_utils.py -->

## Setting Up Checkpoints 📦

### Grit Checkpoints 🚀

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

### Vicuna Weights 🐆

ChatVID uses frozen Vicuna 7B and 13B models. Please first follow the [instructions](https://github.com/lm-sys/FastChat) to prepare Vicuna v1.1 weights. 
Then modify the `vicuna.model_path` in the [Infer Config](https://github.com/Go2Heart/ChatVID/blob/master/config/infer.yaml) to the folder that contains Vicuna weights.

### Vid2Seq Checkpoints 🎥📊

1. Prepare CLIP first for feature extraction in Vid2Seq.
Get the CLIP [Checkpoints](). Specify the `vid2seq.clip_path` in the [Infer Config](https://github.com/Go2Heart/ChatVID/blob/master/config/infer.yaml) to the checkpoint path. 
`vid2seq.output_path` is used to store the generated TFRecords and can be specified to any writable directory. 
`vid2seq.work_dir` is the Flax's working directory and can be specified to any writable directory.

2. Prepare Vid2Seq ActivityNet Checkpoints
Get the Vid2Seq ActivityNet [Checkpoints](https://storage.googleapis.com/scenic-bucket/vid2seq/anet-captions). And then rename it as `checkpoint_200000`. After that, change the `vid2seq.checkpoint_path` in the [Infer Config](https://github.com/Go2Heart/ChatVID/blob/master/config/infer.yaml) to the folder directory where contains the checkpoint.

## Gradio WebUI Usage 🌐

```
python demo.py --config config/infer.yaml
```
