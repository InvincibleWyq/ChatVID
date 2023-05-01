from model.fastchat.serve.inference import ChatIO, question_loop, answer_loop, chat_loop
from model.fastchat.conversation import Conversation, SeparatorStyle
import json, os


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
    
class VicunaHandler:
    """ VicunaHandler is a class that handles the communication between the frontend and the backend.
    """
    def __init__(self, config):
        self.config = config
        self.chat_io = SimpleChatIO()
        
    def summarise_caption(self, caption):
        """ Summarise the caption to paragraph.
        """
        question_loop(
            self.config['model_path'], 
            self.config['device'], 
            self.config['num_gpus'],
            self.config['max_gpu_memory'],
            self.config['load_8bit'],
            self.config['conv_template'],
            self.config['temperature'],
            self.config['max_new_tokens'],
            self.chat_io,
            self.config['debug'],
            prompt_caption=caption,
            output_path=self.config['output_path'],
        )
        
    def chat(self):
        """ Chat with the Vicuna.
        """
        prompt = self._get_prompt()
        template = self._construct_conversation(prompt)
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
    
    def _construct_conversation(self, prompt):
        
        return Conversation(
            system="A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions.",
            roles=("USER", "ASSISTANT"),
            messages=(("USER", prompt),("ASSISTANT","yes")),
            offset=0,
            sep_style=SeparatorStyle.TWO,
            sep=" ",
            sep2="</s>",
        )
    
    def _get_prompt(self):
        caption = dict()
        with open(self.config['output_path'], 'r') as f:
            caption = json.load(f)
        captions = ""
        for it, v in enumerate(caption.values()):
            captions += "Caption" + str(it)+ ": " + v + "\n"
        prompt = "Answer the questions below based on the given video captions in time order. Imagine people's action based on simple words in caption(Caption0: ...).\n----\n" + captions + "\n----\nAnswer with 'yes' or 'no' only for each question.\n----\n Example: Is there a person?"
        # random print 20 questions and answers
        # random_ids = random.sample(list(data_info[video_name]['questions'].keys()), 10)
        # for ix ,id in enumerate(random_ids):
        #     print(str(ix)+'. '+data_info[video_name]['questions'][id])
        #     prompt += 'Question' + str(ix) + ': ' + data_info[video_name]['questions'][id] + '\n'

        # answer = {}
        # for ix, id in enumerate(random_ids):
        #     print(str(ix)+'. '+data_info[video_name]['answers'][id])
        #     answer[str(ix)] = data_info[video_name]['answers'][id]
        return prompt