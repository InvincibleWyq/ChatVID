import whisper


class SpeechRecognizer:

    def __init__(self, device='cuda'):
        self.model = whisper.load_model('base').to(device)

    def __call__(self, video_path):
        generated_text = self.model.transcribe(video_path)['text']
        return generated_text
