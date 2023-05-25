# ChatVID
Chat about anything on any video!

## Gradio Example
<img width="1759" alt="image" src="https://user-images.githubusercontent.com/71871209/235849280-1e7b3ba4-80c4-44c3-940c-cf09775d984f.png">

## Install Instructions
```
pip install -r requirements.txt

mim install mmengine mmcv mmaction2
# Install ffmpeg for Whisper.
# Note that if Whisper encounters permission errors, you may need to specify the DATA_GYM_CACHE_DIR to your writable cache directory.

cd path_to_scenic/ 
pip install . # have to use local module, change here TODO

# change the scenic/dataset_lib/video_ops.py and scenic/train_lib_deprecated/train_utils.py
```


## Gradio WebUI Usage
### Step I
```
python demo.py --config config/debug.yaml
```

