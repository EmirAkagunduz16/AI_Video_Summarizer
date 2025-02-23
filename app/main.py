import os 
import sys
from fastapi import FastAPI, HTTPException, Request


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from summarization.summarization import Process
# from highlight_and_timestamps.highlight_of_videos import 

