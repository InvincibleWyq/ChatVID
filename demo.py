import argparse
import time

import gradio as gr

from config.config_utils import get_config
from model import Captioner, VicunaHandler


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


def clear_chat(chat_history):
    handler.chatbot.clear_conv_()

    return "", []


if __name__ == '__main__':
    _argparser = argparse.ArgumentParser()
    _argparser.add_argument("--config_path", required=True, type=str)
    _args = _argparser.parse_args()
    config = get_config(_args.config_path)

    captioner = Captioner(config)  # global

    global handler
    handler = VicunaHandler(config['vicuna'])

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("## <h1><center>ChatVID</center></h1>")
        gr.Markdown("""
        ChatVID is a video chatbot that can chat about any video.
        """)
        with gr.Row():
            with gr.Column():
                video_path = gr.Video(label="Video")

                with gr.Column():
                    upload_button = gr.Button(
                        "Upload & Watch. (Click once and wait 3min )")
                    chat_button = gr.Button("Let's Chat!", interactive=False)
                    num_frames = gr.Slider(
                        minimum=5,
                        value=12,
                        maximum=12,
                        step=1,
                        label="Number of frames (no more than 12)")

            with gr.Column():
                chatbot = gr.Chatbot()
                captions = gr.State("")
                with gr.Row(visible=False) as input:
                    with gr.Column(scale=0.7):
                        txt = gr.Textbox(
                            show_label=False,
                            placeholder="Enter text and press enter").style(
                                container=False)
                    with gr.Column(scale=0.15, min_width=0):
                        run_button = gr.Button("RUN!")
                    with gr.Column(scale=0.15, min_width=0):
                        clear_button = gr.Button("CLEAR")

        upload_button.click(
            lambda: gr.update(interactive=False), None, chat_button).then(
                lambda: gr.update(visible=False), None,
                input).then(lambda: [], None, chatbot).then(
                    captioner.caption_video, [video_path, num_frames],
                    [captions]).then(lambda: gr.update(interactive=True), None,
                                     chat_button)

        chat_button.click(handler.gr_chatbot_init, [captions],
                          None).then(lambda: gr.update(visible=True), None,
                                     input)

        txt.submit(respond, inputs=[txt, chatbot], outputs=[txt, chatbot])
        run_button.click(
            respond, inputs=[txt, chatbot], outputs=[txt, chatbot])
        clear_button.click(
            clear_chat, inputs=[chatbot], outputs=[txt, chatbot])

    demo.launch(share=True)
