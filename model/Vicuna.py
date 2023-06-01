from model.fastchat.conversation import (Conversation, SeparatorStyle,
                                         compute_skip_echo_len,
                                         get_default_conv_template)
from model.fastchat.serve.inference import (ChatIO, chat_loop, generate_stream,
                                            load_model)


class SimpleChatIO(ChatIO):

    def prompt_for_input(self, role) -> str:
        return input(f"{role}: ")

    def prompt_for_output(self, role: str):
        print(f"{role}: ", end="", flush=True)

    def stream_output(self, output_stream, skip_echo_len: int):
        pre = 0
        for outputs in output_stream:
            outputs = outputs[skip_echo_len:].strip()
            outputs = outputs.split(" ")
            now = len(outputs) - 1
            if now > pre:
                print(" ".join(outputs[pre:now]), end=" ", flush=True)
                pre = now
        print(" ".join(outputs[pre:]), flush=True)
        return " ".join(outputs)


class VicunaChatBot:

    def __init__(
        self,
        model_path: str,
        device: str,
        num_gpus: str,
        max_gpu_memory: str,
        load_8bit: bool,
        conv_template,
        ChatIO: ChatIO,
        debug: bool,
    ):
        self.model_path = model_path
        self.device = device
        self.chatio = ChatIO
        self.debug = debug

        self.model, self.tokenizer = load_model(self.model_path, device,
                                                num_gpus, max_gpu_memory,
                                                load_8bit, debug)

        if conv_template:
            self.conv = conv_template.copy()
        else:
            self.conv = get_default_conv_template(model_path).copy()

        self.conv_template = self.conv.copy()

    def chat(self, inp: str, temperature: float, max_new_tokens: int):
        """ Vicuna as a chatbot. """
        self.conv.append_message(self.conv.roles[0], inp)
        self.conv.append_message(self.conv.roles[1], None)

        generate_stream_func = generate_stream
        prompt = self.conv.get_prompt()

        skip_echo_len = compute_skip_echo_len(self.model_path, self.conv,
                                              prompt)
        stop_str = (
            self.conv.sep if self.conv.sep_style
            in [SeparatorStyle.SINGLE, SeparatorStyle.BAIZE] else None)
        params = {
            "model": self.model_path,
            "prompt": prompt,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "stop": stop_str,
        }
        print(prompt)
        self.chatio.prompt_for_output(self.conv.roles[1])
        output_stream = generate_stream_func(self.model, self.tokenizer,
                                             params, self.device)
        outputs = self.chatio.stream_output(output_stream, skip_echo_len)
        # NOTE: strip is important to align with the training data.
        self.conv.messages[-1][-1] = outputs.strip()
        return outputs

    def summarise(self, caption: dict, temperature: float,
                  max_new_tokens: int):
        """ Vicuna as a summariser. """
        questions = caption
        captions = {}
        for id, question in questions.items():
            # Reset the conversation for each iteration
            self.conv = get_default_conv_template(self.model_path).copy()
            self.conv.append_message(self.conv.roles[0], question)
            self.conv.append_message(self.conv.roles[1], None)

            generate_stream_func = generate_stream
            prompt = self.conv.get_prompt()

            skip_echo_len = compute_skip_echo_len(self.model_path, self.conv,
                                                  prompt)
            stop_str = (
                self.conv.sep if self.conv.sep_style
                in [SeparatorStyle.SINGLE, SeparatorStyle.BAIZE] else None)

            params = {
                "model": self.model_path,
                "prompt": prompt,
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "stop": stop_str,
            }

            self.chatio.prompt_for_output(self.conv.roles[1])
            output_stream = generate_stream_func(self.model, self.tokenizer,
                                                 params, self.device)
            outputs = self.chatio.stream_output(output_stream, skip_echo_len)
            captions[id] = outputs

            if self.debug:
                print("\n", {"prompt": prompt, "outputs": outputs}, "\n")

        print(captions)
        return captions

    def clear_conv_(self):
        """ Clear the conversation. """
        self.conv = self.conv_template.copy()

    def change_conv_template_(self, conv_template):
        self.conv_template = conv_template.copy()
        self.conv = conv_template.copy()

    def change_conv_(self, conv_template):
        """ Change the conversation. """
        self.conv = conv_template.copy()


class VicunaHandler:
    """ VicunaHandler is a class that handles the communication between the
    frontend and the backend. """

    def __init__(self, config):
        self.config = config
        self.chat_io = SimpleChatIO()
        self.chatbot = VicunaChatBot(
            self.config['model_path'],
            self.config['device'],
            self.config['num_gpus'],
            self.config['max_gpu_memory'],
            self.config['load_8bit'],
            None,
            self.chat_io,
            self.config['debug'],
        )

    def chat(self):
        """ Chat with the Vicuna. """
        template = self._construct_conversation("")
        chat_loop(
            self.config['model_path'],
            self.config['device'],
            self.config['num_gpus'],
            self.config['max_gpu_memory'],
            self.config['load_8bit'],
            template,
            self.config['temperature'],
            self.config['max_new_tokens'],
            self.chat_io,
            self.config['debug'],
        )

    def gr_chatbot_init(self, caption: str):
        """ Initialise the chatbot for gradio. """

        template = self._construct_conversation(caption)
        self.chatbot.change_conv_template_(template)
        print("Chatbot initialised.")

    def gr_chat(self, inp):
        """ Chat using gradio as the frontend. """
        return self.chatbot.chat(inp, self.config['temperature'],
                                 self.config['max_new_tokens'])

    def _construct_conversation(self, prompt):
        """ Construct a conversation template.
        Args:
            prompt: the prompt for the conversation.
        """

        user_message = "The following text described what you have " +\
            "seen, found, heard and notice from a consecutive video." +\
            " Some of the texts may not be accurate. " +\
            "Try to conclude what happens in the video, " +\
            "then answer my question based on your conclusion.\n" +\
            "<video begin>\n" + prompt + "<video end>\n" +\
            "Example: Is this a Video?"

        user_message = user_message.strip()

        print(user_message)

        return Conversation(
            system=
            "A chat between a curious user and an artificial intelligence assistant answering quetions on videos."
            "The assistant answers the questions based on the given video captions and speech in time order.",
            roles=("USER", "ASSISTANT"),
            messages=(("USER", user_message), ("ASSISTANT", "yes")),
            offset=0,
            sep_style=SeparatorStyle.TWO,
            sep=" ",
            sep2="</s>",
        )
