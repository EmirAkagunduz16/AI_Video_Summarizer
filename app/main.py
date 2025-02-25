import os 
import sys
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from summarization.summarization import Process
from highlight_and_timestamps.highlight_of_videos import Highlighting, Segment
from summarization.summarization import Summarizer
from sentence_transformers import SentenceTransformer
from moviepy import VideoFileClip
import dotenv

dotenv.load_dotenv()
summarization_api_key = os.getenv("SUM_API_KEY")

app = FastAPI()

class Url(BaseModel):
  url: str

class Download(BaseModel):
  is_downloaded: bool = False

      
@app.get("/")
async def root():
    return RedirectResponse(url="/docs")


@app.post("/summarize")
def summarize(url: Url, download: Download):
  process = Process()      
  summarizer = Summarizer(api_key=summarization_api_key)
  
  process.process_all(url.url, download.is_downloaded)
  texts = summarizer.read_text_file(process.text_file_path)
  summarizer.summarize_text(texts) 
  
  return {"summarized_text": summarizer.summarized_text}    
  

@app.post("/highlight")
def highlight(url: Url, download: Download):
  ## creating object
  process = Process()
  process.process_all(url.url, download.is_downloaded)
  
  highlighting = Highlighting()
  
  video = VideoFileClip(process.video_path)
  video_duration = video.duration
  video.close() 
  
  #### Highlight hyperparameters ####
  similarity_threshold = 0.5
  MAX_SEGMENTS_PER_HIGHLIGHT = 2  
  MIN_TIME_GAP = 5
  min_clip_duration = 2  
  max_clip_duration = 40
  
  # Pre-defined terms
  highlight_segments = []  
  used_intervals = []
  split_segments = []
  
  
  highlighting.generate_highlight(process.transcription_text)
  response_message = highlighting.highlight_parsing()
  
  model = SentenceTransformer("all-mpnet-base-v2")

  
  for segment in process.transcription_segments:
    split_segments.extend(Segment.split_segment(segment))

  valid_segments = [s for s in split_segments if s['end'] > s['start'] and s['text'].strip()]

  # Split long segments and sort
  valid_long_segments = Segment.split_long_segments(valid_segments)
  sorted_splitted_long_segments = sorted(valid_long_segments, key=lambda x: x['start'])
  
  for highlight_idx, highlight in enumerate(response_message, 1):
    print(f"\n{'='*50}")
    print(f"PROCESSING HIGHLIGHT {highlight_idx}/{len(response_message)}")
    print(f"Highlight Text: {highlight[:150]}...")
    
  selected = highlighting.select_segments_for_highlight(highlight, sorted_splitted_long_segments, used_intervals, model, similarity_threshold, MAX_SEGMENTS_PER_HIGHLIGHT, MIN_TIME_GAP)
  
  for seg in selected:
    start = max(0, seg['start'] - 1)
    end = min(seg['end'] + 1, video_duration)
    if end - start > max_clip_duration:
      end = start + max_clip_duration
    highlight_segments.append({
      'text': highlight,
      'start': start,
      'end': end
    })

  filtered_highlight_segments = [
    segment for segment in highlight_segments
    if segment['end'] - segment['start'] >= min_clip_duration
    and segment['end'] - segment['start'] <= max_clip_duration
    and segment['start'] < segment['end']
  ]
  filtered_highlight_segments = sorted(
    filtered_highlight_segments,
    key=lambda x: x['start']
  )
  
  result = highlighting.create_highlight_video(selected, process.video_path)

  return {"highlight_video": result if result else "No highlights created."}