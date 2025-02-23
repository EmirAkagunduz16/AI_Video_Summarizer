import argparse
import moviepy as mp
import os.path as path

def extract_audio(source: str, bitrate="64k"):
    """
    Extracts the audio from the given source video file
    and saves into a mp3 file, with same name
    """
    video = mp.VideoFileClip(source)
    target = path.splitext(source)[0] + ".mp3"
    # - bitrate is optional though. The size and quality of the
    # audio will vary depending on the bitrate. I found 64k
    # to be optimal for my use case.
    # - write_logfile captures the output of ffmpeg and additional
    # date into a separate log file.
    video.audio.write_audiofile(target, bitrate=bitrate, write_logfile=True)

def main():
    parser = argparse.ArgumentParser(description="Extract audio from a video and save to mp3")
    parser.add_argument("-s", "--source",
                        type=str, 
                        help="Source file to extract", required=True)
    args = parser.parse_args()
    extract_audio(args.source)

if __name__ == "__main__":
    main()