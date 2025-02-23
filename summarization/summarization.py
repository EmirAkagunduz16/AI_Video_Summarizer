import dotenv
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transcription.transcription import VideoTranscriber
from groq import Groq


dotenv.load_dotenv()
transcription_api_key = os.getenv("TRANS_API_KEY")
summarization_api_key = os.getenv("SUM_API_KEY")


class Summarizer:
  def __init__(self, api_key: str):
    self.sum_api_key = api_key
    
  @staticmethod
  def read_text_file(text_file):
    with open(text_file, 'r', encoding="utf-8") as file:
      text = file.read()
    return text
    
  def summarize_text(self, text):
    client = Groq(api_key=self.sum_api_key)
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": text}],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=False,  
        stop=None,
    )

    self.summarized_text = response.choices[0].message.content if response.choices else ""

  def print_summarized_text(self):
      print(self.summarized_text)



class Process:
    def __init__(self):
      self.transcriber = VideoTranscriber(transcription_api_key)
      self.summarizer = Summarizer(summarization_api_key)
      self.mp4_url = input("Url: ")
      is_download = input('Did you give the your url before? Y/n: ').lower()

      if is_download == "y":
        self.is_downloaded = True
      elif is_download == "n":
        self.is_downloaded = False
      else:
        print('Wrong key!')
        sys.exit(0)

      self.video_path = self.download_video()  
      self.audio_path = self.extract_audio()
      self.transcription_text, self.transcription_segments = self.transcribe_audio()
      self.text_file_path = self.save_transcription()
      texts = self.summarizer.read_text_file(self.text_file_path)
      self.summarizer.summarize_text(texts)   
      self.sum_text = self.summarizer.summarized_text     

    def download_video(self):
      return self.transcriber.download_video(self.mp4_url, self.is_downloaded)

    def extract_audio(self):
      return self.transcriber.extract_audio(self.video_path)
    
    def transcribe_audio(self):
      return self.transcriber.transcribe_audio(self.audio_path)
    
    def save_transcription(self):
      return self.transcriber.save_transcription(self.transcription_text, self.video_path)

      
      
if __name__ == "__main__":
  process = Process()