import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import dotenv
from groq import Groq
from process import Process

dotenv.load_dotenv()
summarization_api_key = os.getenv("SUM_API_KEY")


class Summarizer:
  def __init__(self, api_key: str):
    self.sum_api_key = api_key
    
  def read_text_file(self, text_file):
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
      
      
if __name__ == "__main__":
  process = Process()
  summarizer = Summarizer(api_key=summarization_api_key)
  mp4_url = input('Url: ')

  is_download = input('Did you give the this url before? Y/n: ').lower()

  if is_download == "y":
    is_downloaded = True
  elif is_download == "n":
    is_downloaded = False
  else:
    print('Wrong key!')
    sys.exit(0)
  
  process.process_all(mp4_url, is_download)
  texts = summarizer.read_text_file(process.text_file_path)
  summarizer.summarize_text(texts)     
  summarizer.print_summarized_text()
  