from model.vision import DenseCaptioner, ImageCaptioner, Prompter
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
        self.prompter = Prompter()
        
        self.src_dir = ''
        
        # self.frames_folder = config['frames_folder']
        
    def caption_frames(self, video_src, video_name, fps=30):
        """ Caption all frames in the folder
        """
        self._get_frames(video_src, video_name, fps)
        frame_folder = self.src_dir + 'frames'
        
        # get frame paths in the folder and sort them
        frame_paths = [os.path.join(frame_folder, f) for f in os.listdir(frame_folder)]
        frame_paths.sort()
        
        # caption each frame
        captions = {}
        for frame_path in frame_paths:
            caption = self._caption_frame(frame_path)
            captions[frame_path] = caption
        
        return captions  
    
    def _get_frames(self, video_src, video_name, fps=30):
        """ Get frames from a video
        
        Args:
            video_src: path to the video
            fps: frames per second
        
        """
        if not os.path.exists(video_src):
            raise FileNotFoundError(f"{video_src} not found")
        self.src_dir = video_src
        container = av.open(video_src+video_name)
        frames = []
        for frame in container.decode(video=0):
            if frame.index % fps == 0:
                frames.append(frame.to_image())
        # save frames to folder
        if not os.path.exists(self.src_dir + '/frames'):
            os.mkdir(self.src_dir + '/frames')
        for i, frame in enumerate(frames):
            frame.save(self.src_dir + '/frames' +f'/{i}.jpg')
        return frames
  
        
        
        
    def _caption_frame(self, image_src):
        """ Caption a frame
        
        Args:
            image_src: path to the image
        
        """
        width, height = read_image_width_height(image_src)
        image_caption = self.image_captioner.caption_image(image_src=image_src)
        dense_caption = self.dense_captioner.image_dense_caption(image_src)
        prompt = self.prompter.generate_prompt(image_caption, dense_caption, width, height)
        return prompt
    
    
    