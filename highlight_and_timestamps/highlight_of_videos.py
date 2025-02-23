import sys
import os
import numpy as np
from moviepy import VideoFileClip, concatenate_videoclips
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from dotenv import load_dotenv
import re
from urllib.parse import urlparse, unquote
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from summarization.summarization import Process

load_dotenv()
highlight_api_key = os.getenv('HIGHLIGHT_API_KEY')

print('###################################')
processor = Process()
print('###################################')
# print(processor.transcription_text)


## TODO: improve this system and solve duplicated problem
client = Groq(api_key=highlight_api_key)
response = client.chat.completions.create(
    model="qwen-2.5-32b",
    messages=[
        {
            "role": "system",
            "content": "Analyze this transcript and extract the most important moments. Focus on actions, events, or statements that are significant to the context of the content. Be concise and limit the summary to 5-7 key moments."
        },
        {
            "role": "user",
            "content": processor.transcription_text
        }
    ],
    temperature=0.6,
    max_completion_tokens=4096,
    top_p=0.95,
    stream=False,
    stop=None,
)

# Improved highlight parsing with regex
raw_highlights = response.choices[0].message.content
response_message = []
for line in raw_highlights.split('\n'):
    line = re.sub(r'^(\d+[\.\)]?|[-*])\s*', '', line.strip())  # Remove numbering/bullets
    if line:
        response_message.append(line)

print("Parsed Highlights:", response_message)


model = SentenceTransformer("all-MiniLM-L6-v2")

# # Print transcription segments for debugging
# print("Transcription Segments:")
# for segment in processor.transcription_segments:
#     print(f"Start: {segment['start']}, End: {segment['end']}, Text: {segment['text']}")

def split_long_segments(segments, max_duration=60):
    new_segments = []
    for seg in segments:
        duration = seg['end'] - seg['start']
        if duration > max_duration:
            mid_point = seg['start'] + (duration/2)
            new_segments.append({
                'start': max(0, mid_point - 15),
                'end': min(mid_point + 15, seg['end']),
                'text': seg['text']
            })
        else:
            new_segments.append(seg)
    return new_segments

def split_segment(segment, max_duration=20):
    start = segment['start']
    end = segment['end']
    text = segment['text']
    duration = end - start
    
    if duration <= max_duration:
        return [segment]
    
    # Metni zaman bazlı kelime hızıyla böl
    words = text.split()
    words_per_sec = len(words) / duration  # Saniyede düşen kelime sayısı
    chunks = []
    chunk_start = start
    
    while chunk_start < end:
        chunk_end = min(chunk_start + max_duration, end)
        chunk_duration = chunk_end - chunk_start
        num_words = int(words_per_sec * chunk_duration)
        
        # Orijinal metinden zaman dilimine karşılık gelen kısmı al
        start_word_idx = int((chunk_start - start) * words_per_sec)
        end_word_idx = start_word_idx + num_words
        chunk_text = " ".join(words[start_word_idx:end_word_idx])
        
        chunks.append({
            'start': chunk_start,
            'end': chunk_end,
            'text': chunk_text.strip()
        })
        chunk_start = chunk_end
    return chunks

# Split all transcription segments
split_segments = []
for segment in processor.transcription_segments:
    split_segments.extend(split_segment(segment))

valid_segments = [s for s in split_segments if s['end'] > s['start'] and s['text'].strip()]

# Split long segments and sort
valid_long_segments = split_long_segments(valid_segments)
sorted_splitted_long_segments = sorted(valid_long_segments, key=lambda x: x['start'])

print('###DEBUG:SortedS Splitted Segments ###')
for segment in sorted_splitted_long_segments:
    print(f"Start: {segment['start']}, End: {segment['end']}, Text: {segment['text']}")


# Process highlights
highlight_segments = []
similarity_threshold = 0.5
MAX_SEGMENTS_PER_HIGHLIGHT = 2  
MIN_TIME_GAP = 5
min_clip_duration = 2  
max_clip_duration = 40

video = VideoFileClip(processor.video_path)
video_duration = video.duration
video.close()  

used_segments = set()
used_intervals = []

def debug_segment_info(seg, similarity, conflict, used_intervals):
    print(f"\nSegment:")
    print(f"Text: {seg['text'][:100]}...")
    print(f"Time: {seg['start']:.1f}s - {seg['end']:.1f}s")
    print(f"Similarity: {similarity:.2f}")
    print(f"Conflict: {'Yes' if conflict else 'No'}")
    if conflict:
        print(f"Conflicting Intervals: {used_intervals}")
    print(f"Status: {'Selected' if similarity >= similarity_threshold and not conflict else 'Rejected'}")


print(f"\n{'#'*40}")
print(f"DEBUG MODE ACTIVE")
print(f"SIMILARITY THRESHOLD: {similarity_threshold}")
print(f"MAX CLIP DURATION: {max_clip_duration}s")
print(f"{'#'*40}\n")

def select_segments_for_highlight(highlight, sorted_splitted_long_segments, used_intervals, model, similarity_threshold, MAX_SEGMENTS_PER_HIGHLIGHT, MIN_TIME_GAP):
    best_matches = []
    highlight_embed = model.encode(highlight)
    
    for seg in sorted_splitted_long_segments:
        conflict = any(
            (seg['start'] >= (u_start - MIN_TIME_GAP)) and 
            (seg['start'] <= (u_end + MIN_TIME_GAP))
            for u_start, u_end in used_intervals
        )
        
        seg_embed = model.encode(seg['text'])
        similarity = cosine_similarity([highlight_embed], [seg_embed])[0][0]
        
        debug_segment_info(seg, similarity, conflict, used_intervals)

        if similarity >= similarity_threshold and not conflict:
            best_matches.append((similarity, seg))
    
    best_matches.sort(reverse=True, key=lambda x: x[0])
    
    selected = []
    for sim, seg in best_matches:
        if len(selected) >= MAX_SEGMENTS_PER_HIGHLIGHT:
            break
        selected.append(seg)
        used_intervals.append((seg['start'], seg['end']))
    
    return selected

for highlight_idx, highlight in enumerate(response_message, 1):
    print(f"\n{'='*50}")
    print(f"PROCESSING HIGHLIGHT {highlight_idx}/{len(response_message)}")
    print(f"Highlight Text: {highlight[:150]}...")
    
    selected = select_segments_for_highlight(highlight, sorted_splitted_long_segments, used_intervals, model, similarity_threshold, MAX_SEGMENTS_PER_HIGHLIGHT, MIN_TIME_GAP)
    
    print(f"\nSelected Segments ({len(selected)}):")
    for seg in selected:
        start = max(0, seg['start'] - 1)
        end = min(seg['end'] + 1, video_duration)
        print(f"• {start:.1f}s-{end:.1f}s ({end-start:.1f}s) | {seg['text'][:50]}...")
    
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



# Filter out segments that are too short or invalid

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
# for segment in filtered_highlight_segments:
#     print(f"Previewing Clip: {segment['text']} (Start: {segment['start']}, End: {segment['end']})")
#     video = VideoFileClip(processor.video_path)
#     clip = video.subclipped(segment['start'], segment['end'])
#     clip.preview()


# Proceed with video clipping
# Create final video
if filtered_highlight_segments:
    video = VideoFileClip(processor.video_path)
    clips = []
    for seg in sorted(highlight_segments, key=lambda x: x['start']):
        try:
            clip = video.subclipped(seg['start'], seg['end'])
            clips.append(clip)
        except Exception as e:
            print(f"Error processing clip {seg}: {str(e)}")
    
    if clips:
        final_clip = concatenate_videoclips(clips)
        # Generate safe filename
        if video.filename.startswith('http'):
            parsed = urlparse(processor.video_path)
            base_name = os.path.splitext(unquote(parsed.path))[0]
        else:
            base_name = os.path.splitext(processor.video_path)[0]
        output_path = f"{base_name}_highlight.mp4"
        
        final_clip.write_videofile(
            output_path,
            codec="libx264",
            fps=24,
            preset='fast',
            threads=4
        )
        print(f"Highlight video saved to: {output_path}")
        final_clip.preview()
    else:
        print("No valid clips to process.")
else:
    print("No highlights generated.")

# https://youtu.be/3ec3B0cjOMs?si=6uccJDZ_HIxLNK7o (mini english podcast)
# https://youtu.be/9xYdYbz9THg?si=GWuzjfsWOOdG_mnE (english podcast)
# https://youtu.be/n4LCbWCshdw?si=NmRfL9l1vH5C1R03 (baris ozcan)
# https://youtu.be/Gc0yVaM-ADY?si=X0oxq2IpmnYjPE4o (gs)
# https://youtu.be/aKSHgMqCwbQ?si=wAL3xcFxorOkgCoQ (spain)