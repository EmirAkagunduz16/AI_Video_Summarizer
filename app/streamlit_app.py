import requests
import streamlit as st
import time
import os
from urllib.parse import urlparse

# API endpoints
API_BASE_URL = "http://127.0.0.1:8000"
HIGHLIGHT_URL = f"{API_BASE_URL}/highlight"
SUMMARIZE_URL = f"{API_BASE_URL}/summarize"

# Page configuration
st.set_page_config(
    page_title="AI Video Summarizer",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS for better styling
st.markdown(
    """
    <style>
        .main {
            background-color: #1e1e1e;
            color: #f0f0f0;
        }
        .stTextArea, .stButton>button {
            background-color: #333333 !important;
            color: #ffffff !important;
        }
        .stTextArea textarea {
            background-color: #333333 !important;
            color: #ffffff !important;
        }
        .result-container {
            background-color: #2d2d2d;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #444;
            margin-top: 20px;
        }
        .highlight-box {
            margin-top: 10px;
            padding: 15px;
            background-color: #3a3a3a;
            border-radius: 5px;
        }
        .video-container {
            margin-top: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

def is_valid_url(url):
    """Check if the provided string is a valid URL."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def show_progress_bar(task_name):
    """Display a progress bar for processing operations."""
    progress_text = f"Processing {task_name}... Please wait."
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(100):
        progress_bar.progress(i + 1)
        status_text.text(f"{progress_text} ({i+1}%)")
        time.sleep(0.05)
    
    status_text.text(f"{task_name} completed!")
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()

def main():
    st.title("🎬 AI Video Summarizer & Highlighter")
    st.write("Generate AI-powered summaries and highlights from videos.")
    
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            video_url = st.text_input("Enter the YouTube video URL:", placeholder="https://www.youtube.com/watch?v=...")
            
        with col2:
            download_option = st.checkbox("Save video locally", value=True, help="Download and save the video to your local machine")
    
    if not video_url:
        st.info("Please enter a YouTube video URL to get started.")
        
        # Example section
        with st.expander("See example URLs"):
            st.markdown("""
            Try these example videos:
            - TED Talk: `https://www.youtube.com/watch?v=8jPQjjsBbIc`
            - Educational: `https://www.youtube.com/watch?v=dQw4w9WgXcQ`
            - Tutorial: `https://www.youtube.com/watch?v=_9WiB2PDO7k`
            """)
            
        st.stop()
    
    if not is_valid_url(video_url):
        st.error("Please enter a valid URL.")
        st.stop()
    
    # Create tabs for different operations
    tab1, tab2 = st.tabs(["📝 Summarize", "✨ Highlight"])
    
    with tab1:
        if st.button("Generate Summary", key="summarize_button"):
            try:
                with st.spinner("Generating summary..."):
                    # Format the request payload according to your API's expectations
                    # Looking at your FastAPI code, the API expects:
                    # - An object with a 'url' field (Url class)
                    # - An object with an 'is_downloaded' field (Download class)
                    headers = {"Content-Type": "application/json"}
                    payload = {
                        "url": {
                            "url": video_url  # Nested structure matching your Pydantic model
                        },
                        "download": {
                            "is_downloaded": download_option
                        }
                    }
                    
                    # Debug information
                    with st.expander("Debug Request"):
                        st.json(payload)
                    
                    # Make API request
                    response = requests.post(SUMMARIZE_URL, json=payload, headers=headers)
                    
                    # Debug response
                    with st.expander("Debug Response"):
                        st.write(f"Status Code: {response.status_code}")
                        st.json(response.json() if response.status_code == 200 else response.text)
                    
                    if response.status_code == 200:
                        summary_data = response.json().get("summarized_text", "No summary generated.")
                        
                        st.markdown("<div class='result-container'>", unsafe_allow_html=True)
                        st.subheader("📝 Video Summary")
                        st.markdown(summary_data)
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Add download button for the summary
                        st.download_button(
                            label="Download Summary",
                            data=summary_data,
                            file_name="video_summary.txt",
                            mime="text/plain"
                        )
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    
    with tab2:
        if st.button("Generate Highlights", key="highlight_button"):
            try:
                with st.spinner("Processing video and generating highlights..."):
                    # Format payload correctly for your API
                    headers = {"Content-Type": "application/json"}
                    payload = {
                        "url": {
                            "url": video_url  # Nested structure matching your Pydantic model
                        },
                        "download": {
                            "is_downloaded": download_option
                        }
                    }
                    
                    # Debug information
                    with st.expander("Debug Request"):
                        st.json(payload)
                    
                    # Make API request
                    response = requests.post(HIGHLIGHT_URL, json=payload, headers=headers)
                    
                    # Debug response
                    with st.expander("Debug Response"):
                        st.write(f"Status Code: {response.status_code}")
                        st.json(response.json() if response.status_code == 200 else response.text)
                    
                    if response.status_code == 200:
                        highlight_result = response.json().get("highlight_video", "No highlights created.")
                        
                        st.markdown("<div class='result-container'>", unsafe_allow_html=True)
                        st.subheader("✨ Video Highlights")
                        
                        if highlight_result != "No highlights created.":
                            # Display the highlight video if available
                            st.markdown("<div class='video-container'>", unsafe_allow_html=True)
                            
                            # Assuming the response contains a path to the highlight video
                            if os.path.exists(highlight_result):
                                st.video(highlight_result)
                            else:
                                st.markdown(f"**Highlight video created at:** {highlight_result}")
                                st.info("Video file not accessible directly through Streamlit. Check the file location on your server.")
                            
                            st.markdown("</div>", unsafe_allow_html=True)
                        else:
                            st.warning(highlight_result)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    
    # Add a footer with additional information
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888;">
        AI Video Summarizer & Highlighter Tool | Built with Streamlit
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()