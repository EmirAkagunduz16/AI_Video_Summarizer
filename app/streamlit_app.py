import requests
import streamlit as st


API_URL = "http://localhost:8000"

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