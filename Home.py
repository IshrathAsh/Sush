import streamlit as st

st.set_page_config(
    page_title="SUSH - Home",
    page_icon="👋",
)

st.write("# Welcome to SUSH! 👋")

st.sidebar.success("Select a page above.")

st.markdown(
    """
    ### Sush - Profanity Censorship Project
    
    This project allows you to censor profanity in audio and video files using AI.
    
    **👈 Select a page from the sidebar to get started:**
    
    - **Use Sush**: The main tool to upload, transcribe, and censor your media.
    - **Code Sush**: Learn about the code and logic behind the project.
    - **About Ish**: Meet the creator.
    
    ---
    *Powered by WhisperX, MoviePy, and Streamlit.*
    """
)
