import requests
import streamlit as st


HIGHLIGHT_URL = "http://127.0.0.1:8000/highlight"
SUMMARIZE_URL = "http://127.0.0.1:8000/summarize"


# Apply dark theme styles
st.markdown(
    """
    <style>
        body {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        .stTextArea, .stButton>button {
            background-color: #333333 !important;
            color: #ffffff !important;
        }
        .stTextArea textarea {
            background-color: #333333 !important;
            color: #ffffff !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    st.title("AI Video Summarizer")
    st.write("Generate AI-powered Summarize the videos.")

    video_url = st.text_area("Enter the video url below:", height=200)
    
    if st.button("Generate Summarize"):
        if video_url.strip():
            headers = {"Content-Type": "application/json"}
            summarize = requests.post(SUMMARIZE_URL, json={"text": video_url}, headers=headers)
            
            if summarize.status_code == 200:
                response_data = summarize.json().get("summarize", "No response generated.")
                st.subheader("AI Generated Summarize:")
                st.markdown(f"""
                    <div style="background-color:#333333;padding:15px;border-radius:10px;color:white;">
                        {response_data}
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.error(f"Error: {summarize.status_code} - {summarize.json().get('detail', 'Unknown error')}")
        else:
            st.warning("Please enter an video url before generating a summarized text.")

    if st.button("Generate Highlight"):
        if video_url.strip():
            headers = {"Content-Type": "application/json"}
            highlight = requests.post(HIGHLIGHT_URL, json={"text": video_url}, headers=headers)
            
            if highlight.status_code == 200:
                response_data = highlight.json().get("highlight", "No response generated.")
                st.subheader("AI Generated Highlight:")
                st.markdown(f"""
                    <div style="background-color:#333333;padding:15px;border-radius:10px;color:white;">
                        {response_data}
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.error(f"Error: {highlight.status_code} - {highlight.json().get('detail', 'Unknown error')}")
        else:
            st.warning("Please enter an video url before generating a summarized text.")