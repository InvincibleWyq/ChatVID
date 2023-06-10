
<h1 align="center">
 <img src="https://github.com/InvincibleWyq/ChatVID/assets/37479394/1a7f47ca-ffbd-4720-b43a-4304fcaa8657" height=320/><br>
 ChatVID
</h1>
<h4 align="center">💬 Chat about anything on any video! 🎥</h4>

<p align="center">
  <b>Authors:</b><br>
  <b><a href="https://github.com/Go2Heart"> Yibin Yan#</a></b>, BUPT<br>
  <b><a href="https://github.com/InvincibleWyq"> Yiqin Wang#</a></b>, Tsinghua University<br>
  <b><a herf="https://andytang15.github.io"> Yansong Tang#</a></b>, Tsinghua-Berkeley Shenzhen Institute<br>
  (# = equal contribution, names listed alphabetically)<br>
  This work is done during Yibin and Yiqin's internship with Prof. Tang.
</p>

Our ChatVID combines the understanding ability of Large Language Models and the sensing ablity of Vision Models and Audio Models. Altogether, they form a powerful capability to bring users the experience to talk about anything in a video.


## Highlights 🔥
- 🔍 Leverage the power of Large Language Models, Vision Models, and Audio Models to enable conversations about videos.
- 🤖 Utilize [Vicuna](https://vicuna.lmsys.org) as the Large Language Model for understanding user queries and responses.
- 📷 Incorporate state-of-the-art Vision Models like [BLIP2](https://blog.salesforceairesearch.com/blip-2/), [GRiT](https://github.com/JialianW/GRiT), and [Vid2Seq](https://antoyang.github.io/vid2seq.html) for visual understanding and analysis.
- 🎤 Employ [Whisper](https://openai.com/research/whisper) as an Audio Model to process audio content within videos.
- 💬 Enable users to have conversations and discussions about any aspect of a video.
- 🚀 Enhance the overall video-watching experience by providing an interactive and engaging platform.
- 🚗 ChatVID with Vicuna-7B (8bit) is able to run with a Nvidia GPU with 24G RAM.

## Gradio Example ✨
<h1 align="center">
<img src="https://github.com/InvincibleWyq/ChatVID/assets/37479394/509aa0ce-233a-4418-b245-ebc52e7e9ad9" alt="The Temple Of Heaven" class="center">
<img src="https://github.com/InvincibleWyq/ChatVID/assets/37479394/66d4aec8-a322-4cf2-ac74-9aad9fd89d16" alt="Cook" class="center">
</h1>
<img width="1624" alt="image" src="https://github.com/InvincibleWyq/ChatVID/assets/37479394/b682bd98-5a6d-400d-9679-c80a258a76b5">
<img width="1624" alt="image" src="https://github.com/InvincibleWyq/ChatVID/assets/37479394/da9f79b7-8845-4e91-829e-c3c001d67294">

## Install Instructions 💻

```
pip install -r requirements.txt

mim install mmengine mmcv mmaction2
```

You will also need to install [ffmpeg](https://ffmpeg.org) for Whisper. Note that if Whisper encounters permission errors, you may need to specify environment variable `DATA_GYM_CACHE_DIR='/YourRootDir/ChatVID/.cache'`, a writable cache directory.
<!-- # change the scenic/dataset_lib/video_ops.py and scenic/train_lib_deprecated/train_utils.py -->

## Setting Up Checkpoints 📦

### Grit Checkpoints 🚀

Put [Grit](https://datarelease.blob.core.windows.net/grit/models/grit_b_densecap_objectdet.pth) into `pretrained_models` folder.

<!-- 
### BLIP2 Checkpoints
```
It will be acquired automatically using Hugging Face's transformers
```
### Whisper Checkpoints
```

```
 -->

### Vicuna Weights 🦙

ChatVID uses frozen Vicuna 7B and 13B models. Please first follow the [instructions](https://github.com/lm-sys/FastChat) to prepare Vicuna v1.1 weights. 
Then modify the `vicuna.model_path` in the [Infer Config](https://github.com/InvincibleWyq/ChatVID/blob/main/config/infer.yaml) to the folder that contains Vicuna weights.

### Vid2Seq Checkpoints 🎥📊

1. Prepare CLIP ViT-L/14 Checkpoint for feature extraction in Vid2Seq.
Get [CLIP ViT-L/14 Checkpoint](https://github.com/openai/CLIP/blob/main/clip/clip.py#L38). Specify the `vid2seq.clip_path` in the [Infer Config](https://github.com/InvincibleWyq/ChatVID/blob/main/config/infer.yaml) to the checkpoint path. 
`vid2seq.output_path` is used to store the generated TFRecords and can be specified to any writable directory. 
`vid2seq.work_dir` is the Flax's working directory and can be specified to any writable directory.

2. Prepare Vid2Seq ActivityNet Checkpoint
Get the [Vid2Seq ActivityNet Checkpoint](https://storage.googleapis.com/scenic-bucket/vid2seq/anet-captions). And then rename it as `checkpoint_200001`. After that, change the `vid2seq.checkpoint_path` in the [Infer Config](https://github.com/InvincibleWyq/ChatVID/blob/main/config/infer.yaml) to the folder directory where contains the checkpoint.

### File Structure

```txt
ChatVID/
|__config/
    |__...
|__model/
    |__...
|__scenic/
    |__...
|__pretrained_models/
    |__grit_b_densecap_objectdet.pth
|__vicuna-7b/
    |__pytorch_model-00001-of-00002.bin
    |__pytorch_model-00002-of-00002.bin
    |__...
|__vid2seq_ckpt/
    |__checkpoint_200001
|__clip_ckpt/
    |__ViT-L-14.pt
|__demo.py
|__README.md
|__requirements.txt
|__LICENSE

```

## Gradio WebUI Usage 🌐

```bash
python demo.py --config config/infer.yaml
```

## Demo 🤗

Coming soon...

## Acknowledgment

This work is based on [Vicuna](https://github.com/lm-sys/FastChat), [BLIP-2](https://huggingface.co/spaces/Salesforce/BLIP2), [GRiT](https://github.com/JialianW/GRiT), [Vid2Seq](https://github.com/google-research/scenic/tree/main/scenic/projects/vid2seq), [Whisper](https://github.com/openai/whisper).
