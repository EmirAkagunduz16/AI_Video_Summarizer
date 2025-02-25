import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from transcription.transcription import VideoTranscriber
import dotenv

dotenv.load_dotenv()
transcription_api_key = os.getenv("TRANS_API_KEY")
summarization_api_key = os.getenv("SUM_API_KEY")

class Process:
    def __init__(self):
      self.transcriber = VideoTranscriber(transcription_api_key)

    def download_video(self, mp4_url, is_downloaded):
      return self.transcriber.download_video(mp4_url, is_downloaded)

    def extract_audio(self, video_path):
      return self.transcriber.extract_audio(video_path)
    
    def transcribe_audio(self, audio_path):
      return self.transcriber.transcribe_audio(audio_path)
    
    def save_transcription(self, transcription_text, video_path):
      return self.transcriber.save_transcription(transcription_text, video_path)

    def process_all(self, mp4_url: str, is_downloaded: bool):
      self.video_path = self.download_video(mp4_url, is_downloaded)  
      self.audio_path = self.extract_audio(self.video_path)
      self.transcription_text, self.transcription_segments = self.transcribe_audio(self.audio_path)
      self.text_file_path = self.save_transcription(self.transcription_text, self.video_path)