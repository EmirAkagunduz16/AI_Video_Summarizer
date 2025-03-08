import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import re
from groq import Groq
from transcription.audio_extractor import extract_audio
import os
import moviepy as mp
import yt_dlp


class VideoTranscriber:
  def __init__(self, api_key: str, download_dir: str = r"C:\Users\Victus\Desktop\Desktop\AI\ML_Projects\YoutubeVideoSummarizer\downloads"):
    self.client = Groq(api_key=api_key)
    self.download_dir = download_dir
    os.makedirs(download_dir, exist_ok=True)


  def download_video(self, url: str, download: bool=False) -> str:
    file_name = re.sub(r'[<>:"/\\|?*]', '', url.split("/")[-1]).strip()  # Clean extra spaces
    file_path = os.path.join(self.download_dir, file_name + ".mp4")

    # Check if file exists when download is True
    if download:
      if not os.path.exists(file_path):
        raise FileNotFoundError(f"Video file not found at: {file_path}")
      return file_path
 
    else:
      ydl_opts = {
          "format": "bestvideo+bestaudio/best",  
          "merge_output_format": "mp4",         
          "outtmpl": file_path                    
      }
      with yt_dlp.YoutubeDL(ydl_opts) as ydl:
          ydl.download([url])

    print(f"Video downloaded: {file_path}")
    return file_path


  def extract_audio(self, video_path: str) -> str:
    audio_path = os.path.splitext(video_path)[0] + ".mp3"
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, codec="mp3")
    video.close()
    print(f"Audio extracted: {audio_path}")
    return audio_path

  @staticmethod
  def clean_transcription_segments(segments, min_words=3, max_silence=1.0):
      """
      Clean and preprocess transcription segments:
      - Remove duplicate or near-duplicate segments.
      - Filter out short segments with too few words.
      - Merge adjacent segments that are part of the same sentence.

      Args:
          segments (list): List of transcription segments from Whisper.
          min_words (int): Minimum number of words for a segment to be kept.
          max_silence (float): Maximum silence (in seconds) between segments to merge them.

      Returns:
          list: Cleaned and merged segments.
      """
      cleaned_segments = []
      previous_segment = None

      for segment in segments:
          text = segment['text'].strip()
          start = segment['start']
          end = segment['end']

          # Skip empty or very short segments
          if len(text.split()) < min_words:
              continue

          # Skip duplicate or near-duplicate segments
          if previous_segment and text == previous_segment['text']:
              continue

          # Merge with previous segment if the gap is small
          if previous_segment and (start - previous_segment['end']) <= max_silence:
              previous_segment['text'] += " " + text
              previous_segment['end'] = end
          else:
              # Add the segment to the cleaned list
              cleaned_segments.append({
                  'start': start,
                  'end': end,
                  'text': text
              })
              previous_segment = cleaned_segments[-1]

      return cleaned_segments
    

  def transcribe_audio(self, audio_path: str) -> str:
      prompt = "Please transcribe the audio accurately, keeping all important details."

      with open(audio_path, "rb") as file:
          transcription = self.client.audio.transcriptions.create(
              file=(audio_path, file.read()),
              model="whisper-large-v3",
              prompt=prompt,
              response_format="verbose_json",
          )

      text = transcription.text
      segments = transcription.segments

      # Clean the transcription segments
      cleaned_segments = self.clean_transcription_segments(segments)

      print("Transcription completed.")
      return text, cleaned_segments


  def save_transcription(self, text: str, file_path: str) -> str:
    output_text_path = os.path.splitext(file_path)[0] + "_transcription.txt"
    with open(output_text_path, "w", encoding="utf-8") as text_file:
        text_file.write(text)
    print(f"Transcription saved to: {output_text_path}")
    return output_text_path
  
  
# def download_video(self, url: str) -> str:
#       # URL'den geçerli bir dosya adı oluştur
#       file_name = url.split("/")[-1]
#       file_name = re.sub(r'[<>:"/\\|?*]', '', file_name)  # Geçersiz karakterleri temizle
#       file_name = file_name.split("&")[0]  # Fazladan parametreleri kes

#       file_path = os.path.join(self.download_dir, file_name + ".mp4")

#       response = requests.get(url, stream=True)
#       if response.status_code == 200:
#           with open(file_path, "wb") as file:
#               for chunk in response.iter_content(chunk_size=1024):
#                   file.write(chunk)
#           print(f"Video downloaded: {file_path}")
#           return file_path
#       else:
#           raise Exception(f"Failed to download video: {url}")