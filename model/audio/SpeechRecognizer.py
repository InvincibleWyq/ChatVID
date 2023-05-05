import torch
import whisper


class SpeechRecognizer:

    def __init__(self, device, base_model='whisper'):
        self.device = device
        if self.device == 'cpu':
            self.data_type = torch.float32
        else:
            self.data_type = torch.float16
        self.base_model = base_model
        if self.base_model == 'whisper':
            self.model = whisper.load_model('base').to(self.device)
        else:
            raise NotImplementedError(
                f"Model {self.base_model} not implemented")

    def recognize_speech(self, video_src=None):
        generated_text = self.model.transcribe(video_src)['text']
        if generated_text == '':
            ret_text = "You hear nothing.\n"
        else:
            ret_text = "You hear \"" + generated_text + "\"\n"
        # print('\033[1;35m' + '*' * 100 + '\033[0m')
        # print('\nStep3, Whisper generated text:')
        # print(ret_text)
        # print('\033[1;35m' + '*' * 100 + '\033[0m')
        return ret_text