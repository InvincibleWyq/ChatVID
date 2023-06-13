import time

import gradio as gr

from config.config_utils import get_config
from model import Captioner, VicunaHandler


def mirror(x):
    return x


def clear_chat(conv_template):
    return "", [], conv_template


def clear_four():
    return [], [], [], []


def respond(input, chat_history, conv):
    bot_response, new_conv = handler.gr_chat(input, conv)
    chat_history.append((input, bot_response))
    time.sleep(0.1)
    return "", chat_history, new_conv


# global variables
config = get_config('config/infer.yaml')
captioner = Captioner(config)
handler = VicunaHandler(config['vicuna'])

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "## <h1><center><img src='https://github.com/InvincibleWyq/ChatVID/assets/37479394/1a7f47ca-ffbd-4720-b43a-4304fcaa8657' height=40/> ChatVID</center></h1>"
    )
    gr.Markdown("""🔥 [ChatVID](https://github.com/InvincibleWyq/ChatVID) is a
    video chatbot. Please give us a ⭐ Star!""")
    gr.Markdown("""🎥 You may use the example video by clicking it.""")
    gr.Markdown("""🚀 For any questions or suggestions, feel free to drop Yiqin
    an email at <a href="mailto:wyq1217@outlook.com">wyq1217@outlook.com</a>
    or open an issue.""")

    with gr.Row():
        with gr.Column():
            video_path = gr.Video(label="Video")

            with gr.Column():
                upload_button = gr.Button("""Upload & Process.
                    (Click and wait 3min until dialog box appears)""")

                num_frames = gr.Slider(
                    minimum=5,
                    value=12,
                    maximum=12,
                    step=1,
                    label="Number of frames")

                gr.Markdown("## Video Examples")
                gr.Examples(
                    examples=[
                        "examples/cook_720p.mp4",
                        "examples/temple_of_heaven_720p.mp4"
                    ],
                    inputs=video_path,
                    outputs=video_path,
                    fn=mirror,
                    cache_examples=True,
                )

        with gr.Column():
            caption_box = gr.Textbox("")
            chatbot = gr.Chatbot()
            conv_template = gr.State("")  # determined by the video
            conv = gr.State("")  # updated thourghout the conversation
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

    # conv_template and conv are `Conversation` objects
    upload_button.click(lambda: gr.update(visible=False), None, input).then(
        clear_four, None, [chatbot, conv, conv_template, caption_box]).then(
            captioner.caption_video, [video_path, num_frames],
            [conv_template]).then(mirror, [conv_template], [caption_box]).then(
                handler.gr_chatbot_init, [conv_template],
                [conv_template, conv]).then(lambda: gr.update(visible=True),
                                            None, input)

    txt.submit(
        respond, inputs=[txt, chatbot, conv], outputs=[txt, chatbot, conv])
    run_button.click(
        respond, inputs=[txt, chatbot, conv], outputs=[txt, chatbot, conv])
    clear_button.click(
        clear_chat, inputs=[conv_template], outputs=[txt, chatbot, conv])

demo.launch(share=True)
