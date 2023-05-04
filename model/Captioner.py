from model.vision import DenseCaptioner, ImageCaptioner, Prompter
from model.audio import SpeechRecognizer
from utils.util import display_images_and_text, read_image_width_height
import av, os


class Captioner:
    """ Captioner class for video captioning
    """

    def __init__(self, config):
        """ Initialize the captioner
        
        Args:
            config: configuration file
        
        """
        self.config = config
        self.image_captioner = ImageCaptioner(config['device'])
        self.dense_captioner = DenseCaptioner(config['device'])
        self.speech_recognizer = SpeechRecognizer(config['device'])
        self.prompter = Prompter()

        self.src_dir = ''

        # self.frames_folder = config['frames_folder']

    def caption_frames(self, video_src, video_name, num_frames=20):
        """ Caption all frames in the folder
        """
        print("Captioning frames...")
        if video_src[-4:] == '.mp4':
            print("video_src is a video file")
            video_name = video_src.split('/')[-1]
            video_src = video_src[:-len(video_name)]
        frame_list = self._get_frames(video_src, video_name, num_frames)
        # frame_folder = self.src_dir + 'frames'

        # get frame paths in the folder and sort them
        # frame_paths = [os.path.join(frame_folder, f) for f in os.listdir(frame_folder)]
        # sort in numerical order
        # frame_paths.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))

        # caption each frame
        captions = "In this video: \n"
        for it, frame in enumerate(frame_list):
            captions += self.image_captioner.caption_image(image=frame)
            captions += self.dense_captioner.image_dense_caption(
                image_src=None, image=frame)

        # recognize speech
        captions += self.speech_recognizer.recognize_speech(video_src +
                                                            video_name)

        print("Captions generated")
        print(captions)
        return captions

    def _get_frames(self, video_src, video_name, num_frames=20, save=False):
        """ Get frames from a video
        
        Args:
            video_src: path to the video
            num_frames: number of frames to be sampled
        
        """
        if not os.path.exists(video_src):
            raise FileNotFoundError(f"{video_src} not found")
        self.src_dir = video_src
        container = av.open(video_src + video_name)
        frames = []
        total_frames = container.streams.video[0].frames
        interval = total_frames // num_frames
        for frame in container.decode(video=0):
            if frame.index % interval == 0:
                frames.append(frame.to_image())  # av.VideoFrame to PIL.Image
        # save frames to folder
        if save:
            if not os.path.exists(self.src_dir + '/frames'):
                os.mkdir(self.src_dir + '/frames')
            for i, frame in enumerate(frames):
                frame.save(self.src_dir + '/frames' + f'/{i}.jpg')
        return frames

    def _caption_frame(self, image):
        """ Caption a frame from PIL.Image
        """
        width, height = image.size
        image_caption = self.image_captioner.caption_image(image=image)
        dense_caption = self.dense_captioner.image_dense_caption(
            image_src=None, image=image)
        prompt = self.prompter.generate_prompt(image_caption, dense_caption,
                                               width, height)
        return prompt

    def _caption_frame_from_path(self, image_src):
        """ Caption a frame from image path
        
        Args:
            image_src: path to the image
        
        """
        width, height = read_image_width_height(image_src)
        image_caption = self.image_captioner.caption_image(
            image_src=image_src, image=None)
        dense_caption = self.dense_captioner.image_dense_caption(image_src)
        prompt = self.prompter.generate_prompt(image_caption, dense_caption,
                                               width, height)
        return prompt
