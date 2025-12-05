@echo off
echo Starting SUSH Streamlit App...
set "PATH=%CD%\ffmpeg\bin;%PATH%"
streamlit run Home.py
pause
