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


#### Highlight hyperparameters ####
similarity_threshold = 0.35  # Further lowered for better matching
MAX_SEGMENTS_PER_HIGHLIGHT = 3  # Increased to allow more segments per highlight
MIN_TIME_GAP = 2.0  # Reduced to allow closer segments
min_clip_duration = 3.0
max_clip_duration = 30.0  # Reduced to avoid too long clips

def debug_segment_info(seg, similarity, conflict, used_intervals):
    print(f"\nSegment Analysis:")
    print(f"Text: {seg['text'][:100]}...")
    print(f"Time: {seg['start']:.1f}s - {seg['end']:.1f}s (Duration: {seg['end']-seg['start']:.1f}s)")
    print(f"Similarity Score: {similarity:.3f} (Threshold: {similarity_threshold})")
    print(f"Overlap Check: {'Failed' if conflict else 'Passed'}")
    if conflict:
        print(f"Overlapping with: {used_intervals}")
    print(f"Final Decision: {'✓ Selected' if similarity >= similarity_threshold and not conflict else '✗ Rejected'}")
    print("-" * 50)


class Highlighting:
   
    def generate_highlight(self, transcription_text):
        client = Groq(api_key=highlight_api_key)
        self.response = client.chat.completions.create(
            model="qwen-2.5-32b",
            messages=[
                {
                    "role": "system",
                    "content": "Analyze this transcript and extract the most important moments. Focus on actions, events, or statements that are significant to the context of the content. Be concise and limit the summary to 5-7 key moments."
                },
                {
                    "role": "user",
                    "content": transcription_text
                }
            ],
            temperature=0.6,
            max_completion_tokens=4096,
            top_p=0.95,
            stream=False,
            stop=None,
        )
    
    def highlight_parsing(self):
        raw_highlights = self.response.choices[0].message.content
        response_message = []
        for line in raw_highlights.split('\n'):
            line = re.sub(r'^(\d+[\.\)]?|[-*])\s*', '', line.strip())  # Remove numbering/bullets
            if line:
                response_message.append(line)
        return response_message
    
    def select_segments_for_highlight(self, highlight, sorted_splitted_long_segments, used_intervals, model, similarity_threshold, MAX_SEGMENTS_PER_HIGHLIGHT, MIN_TIME_GAP):
        selected = []
        highlight_embedding = model.encode([highlight])[0]
        
        # First pass: Find segments with high similarity
        for segment in sorted_splitted_long_segments:
            if len(selected) >= MAX_SEGMENTS_PER_HIGHLIGHT:
                break
                
            # Skip if segment text is too short
            if len(segment['text'].split()) < 3:
                continue
                
            # Calculate similarity
            segment_embedding = model.encode([segment['text']])[0]
            similarity = np.dot(highlight_embedding, segment_embedding)
            
            # Check for conflicts with previously used intervals
            conflict = any(
                max(segment['start'], used_start) <= min(segment['end'], used_end)
                for used_start, used_end in used_intervals
            )
            
            # Adjust similarity threshold for key moments
            adjusted_threshold = similarity_threshold
            if any(keyword in highlight.lower() for keyword in ['goal', 'score', 'win', 'victory', 'champion']):
                adjusted_threshold = similarity_threshold * 0.8  # Lower threshold for key moments
            
            if similarity >= adjusted_threshold and not conflict:
                # Extend segment time slightly to capture context
                start = max(0, segment['start'] - 1)
                end = segment['end'] + 1
                
                # Add to selected segments
                selected.append({
                    'start': start,
                    'end': end,
                    'text': segment['text'],
                    'similarity': similarity
                })
                
                # Add to used intervals
                used_intervals.append((start, end))
                
                debug_segment_info(segment, similarity, conflict, used_intervals)
                
        # Second pass: If no segments were selected, try with a lower threshold
        if not selected and any(keyword in highlight.lower() for keyword in ['goal', 'score', 'win', 'victory', 'champion']):
            for segment in sorted_splitted_long_segments:
                if len(selected) >= MAX_SEGMENTS_PER_HIGHLIGHT:
                    break
                    
                segment_embedding = model.encode([segment['text']])[0]
                similarity = np.dot(highlight_embedding, segment_embedding)
                
                conflict = any(
                    max(segment['start'], used_start) <= min(segment['end'], used_end)
                    for used_start, used_end in used_intervals
                )
                
                if similarity >= similarity_threshold * 0.7 and not conflict:  # Even lower threshold for second pass
                    start = max(0, segment['start'] - 1)
                    end = segment['end'] + 1
                    selected.append({
                        'start': start,
                        'end': end,
                        'text': segment['text'],
                        'similarity': similarity
                    })
                    used_intervals.append((start, end))
                    debug_segment_info(segment, similarity, conflict, used_intervals)
        
        return selected

    def create_highlight_video(self, selected_segments, video_path):
        try:
            video = VideoFileClip(video_path)
            clips = []
            
            # Sort segments by start time
            sorted_segments = sorted(selected_segments, key=lambda x: x['start'])
            
            for seg in sorted_segments:
                try:
                    # Ensure start and end times are valid numbers
                    start_time = float(seg['start'])
                    end_time = float(seg['end'])
                    
                    # Add some validation
                    if start_time >= end_time:
                        print(f"Invalid time range for segment: start={start_time}, end={end_time}")
                        continue
                        
                    if start_time < 0:
                        start_time = 0
                    
                    # Create subclip using the correct method name
                    clip = video.subclipped(start_time, end_time)
                    clips.append(clip)
                    print(f"Successfully added clip from {start_time:.1f}s to {end_time:.1f}s")
                except Exception as e:
                    print(f"Error processing clip {seg}: {str(e)}")

            if clips:
                print(f"\nCreating final video from {len(clips)} clips...")
                final_clip = concatenate_videoclips(clips)
                if video.filename.startswith('http'):
                    parsed = urlparse(video_path)
                    base_name = os.path.splitext(unquote(parsed.path))[0]
                else:
                    base_name = os.path.splitext(video_path)[0]
                output_path = f"{base_name}_highlight.mp4"
                
                try:
                    final_clip.write_videofile(
                        output_path,
                        codec="libx264",
                        fps=24,
                        preset='fast',
                        threads=4
                    )
                    print(f"\nHighlight video successfully saved to: {output_path}")
                finally:
                    # Properly close all clips
                    final_clip.close()
                    for clip in clips:
                        clip.close()
                    video.close()
                return output_path
            else:
                print("No valid clips were created. Please check the segment times and video duration.")
                return None
        except Exception as e:
            print(f"Error creating highlight video: {str(e)}")
            return None
        finally:
            # Ensure video is closed even if an error occurs
            if 'video' in locals():
                video.close()
    

class Segment:
  
  def split_long_segments(segments, max_duration=30):  # Reduced max duration
    new_segments = []
    for seg in segments:
        duration = seg['end'] - seg['start']
        if duration > max_duration:
            # Create smaller segments with small overlap
            num_splits = int(np.ceil(duration / (max_duration - 5)))  # 5 second overlap
            split_duration = duration / num_splits
            
            for i in range(num_splits):
                split_start = seg['start'] + (i * (split_duration - 5))  # Overlap previous segment
                split_end = min(split_start + split_duration, seg['end'])
                
                # Only add if duration is meaningful
                if split_end - split_start >= min_clip_duration:
                    new_segments.append({
                        'start': max(0, split_start),
                        'end': split_end,
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
    
    words = text.split()
    words_per_sec = len(words) / duration  
    chunks = []
    chunk_start = start
    
    while chunk_start < end:
        chunk_end = min(chunk_start + max_duration, end)
        chunk_duration = chunk_end - chunk_start
        num_words = int(words_per_sec * chunk_duration)
        
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




def main():
    similarity_threshold = 0.35
    MAX_SEGMENTS_PER_HIGHLIGHT = 3  # Increased from default
    MIN_TIME_GAP = 2.0
    min_clip_duration = 3.0
    max_clip_duration = 30.0
    highlight_segments = []
    
    process = Process()
    
    print("Url: ", end="")
    mp4_url = input()
    
    print("Did you give the this url before? Y/n: ", end="")
    is_download = input()
    
    if is_download == "y" or is_download == "Y":
        is_downloaded = True
    elif is_download == "n":
        is_downloaded = False
    else:
        print('Wrong key!')
        sys.exit(0)
    
    process.process_all(mp4_url, is_downloaded)
    video = VideoFileClip(process.video_path)
    video_duration = video.duration
    video.close()
    used_intervals = []
    
    highlight_obj = Highlighting()
    highlight_obj.generate_highlight(process.transcription_text)
    response_message = highlight_obj.highlight_parsing()
    
    model = SentenceTransformer("all-mpnet-base-v2")
    
    # Split all transcription segments
    split_segments = []
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
        
        selected = highlight_obj.select_segments_for_highlight(
            highlight, 
            sorted_splitted_long_segments, 
            used_intervals, 
            model, 
            similarity_threshold, 
            MAX_SEGMENTS_PER_HIGHLIGHT, 
            MIN_TIME_GAP
        )
        
        if selected:
            print(f"\nSelected Segments ({len(selected)}):")
            for seg in selected:
                start = max(0, seg['start'] - 1)
                end = min(seg['end'] + 1, video_duration)
                print(f"• {start:.1f}s-{end:.1f}s ({end-start:.1f}s) | {seg['text'][:50]}...")
                
                highlight_segments.append({
                    'text': highlight,
                    'start': start,
                    'end': end
                })
        else:
            print(f"\nNo segments found for highlight {highlight_idx}. Trying with lower threshold...")
            # Try again with lower threshold for important moments
            selected = highlight_obj.select_segments_for_highlight(
                highlight, 
                sorted_splitted_long_segments, 
                used_intervals, 
                model, 
                similarity_threshold * 0.7,  # Lower threshold for retry
                MAX_SEGMENTS_PER_HIGHLIGHT, 
                MIN_TIME_GAP
            )
            if selected:
                print(f"\nSelected Segments with lower threshold ({len(selected)}):")
                for seg in selected:
                    start = max(0, seg['start'] - 1)
                    end = min(seg['end'] + 1, video_duration)
                    print(f"• {start:.1f}s-{end:.1f}s ({end-start:.1f}s) | {seg['text'][:50]}...")
                    
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
    
    if filtered_highlight_segments:
        highlight_obj.create_highlight_video(filtered_highlight_segments, process.video_path)
    else:
        print("No highlights generated.")

if __name__ == "__main__":
    main()