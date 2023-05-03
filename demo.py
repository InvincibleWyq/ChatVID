import argparse
from model import Captioner, VicunaHandler
from config.config_utils import get_config
import gradio as gr
import time



def set_example_video(example: list) -> dict:
    return gr.Video.update(value=example[0])

def upload_file(files):
    file_paths = [file.name for file in files]
    return file_paths

def upload_video(video):
    print(video)
    return video

def respond(input, chat_history):
    bot_response = handler.gr_chat(input)
    chat_history.append((input, bot_response))
    time.sleep(0.1)
    return "", chat_history
    

if __name__ == '__main__':
    _argparser = argparse.ArgumentParser()
    _argparser.add_argument("--config_path", required=True, type=str)
    _args = _argparser.parse_args()
    config = get_config(_args.config_path)

    captioner = Captioner(config)
    # prompted_captions = captioner.caption_frames(config['video_path'], config['video_name'], config['fps'])
    
    # with open('./test.json', 'w') as f:
    #     json.dump(prompted_captions, f)
    global handler
    handler = VicunaHandler(config['vicuna'])
    
        
    with gr.Blocks() as demo:
        gr.Markdown("## <h1><center>ChatVID</center></h1>")
        gr.Markdown(
        """
        Chat about any video with ChatVID! ChatVID is a video chatbot that can chat about any video.
        """
    )
        with gr.Row():
            with gr.Column():
                video_path = gr.Video(label="Video")
                
                # file_output = gr.File()
                with gr.Column():
                    # upload = gr.UploadButton("TEST")
                    upload_button = gr.Button("Upload Video")
                    chat_button = gr.Button("Let's Chat!", interactive=False)
                    fps = gr.Slider(minimum=120, maximum=720, step=1, label="FPS")
                    
            with gr.Column():
                chatbot = gr.Chatbot()
                prompted_captions = gr.State("")
                summarised_caption = gr.State("")
                video_name = gr.State("")
                with gr.Row(visible=False) as input:
                    with gr.Column(scale=0.7):
                        txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter").style(container=False)
                    with gr.Column(scale=0.15, min_width=0):
                        run_button = gr.Button("RUN!")
                    with gr.Column(scale=0.15, min_width=0):
                        clear_button = gr.Button("CLEAR")
        # with gr.Row():
            # example_videos = gr.Dataset(components=[video_path], samples=[['examples/temple_of_heaven_720p.mp4']])
            
        # example_videos.click(fn=set_example_video, inputs=example_videos, outputs=example_videos.components)
        # upload_button.click(lambda: gr.update(interactive=True), None, chat_button)
        # upload_button.click(lambda: [], None, chatbot)
        # upload.upload(upload_file, upload, file_output)
        
        # upload_button.click(
        #     upload_video, [video_path], None
        # )
        
        upload_button.click(
            captioner.caption_frames, [video_path, video_name, fps], [prompted_captions]
        ).then(
            lambda: gr.update(interactive=True), None, chat_button
        ).then(
            lambda: [], None, chatbot
        )
        
        chat_button.click(
            handler.summarise_caption, [prompted_captions], [summarised_caption]
        ).then(
            handler.gr_chatbot_init, [summarised_caption], None
        ).then(
            lambda: gr.update(visible=True), None, input
        )
        
        txt.submit(respond, inputs=[txt, chatbot], outputs=[txt,chatbot])
        clear_button.click(lambda: None, None, chatbot, queue=False)
        
        
        
        
        

    demo.launch(share=True)
                
            
    
    # handler.chat()