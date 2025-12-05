# 🤫 SUSH! - Profanity Censorship Project

> **"Censor the bad, keep the good."**

**SUSH!** is an AI-powered tool designed to automatically detect and censor profanity in audio and video files. Built with **Streamlit**, **WhisperX**, and **MoviePy**, it provides a seamless way to clean up your media content.

---

## ✨ Features

-   **🎥 Video & Audio Support**: Upload MP4, MP3, WAV, and more.
-   **🤖 AI Transcription**: Uses **WhisperX** for highly accurate, word-level timestamps.
-   **🚫 Smart Censorship**: Automatically detects profanity using a custom list or built-in dictionary.
-   **📝 Custom Control**: Manage your own list of banned words directly in the app.
-   **🔇 Beep!**: Replaces bad words with a classic beep sound (or silence).
-   **⚡ GPU Accelerated**: Optimized for CUDA to speed up transcription (if available).

---

## 🛠️ Tech Stack

-   **[Streamlit](https://streamlit.io/)**: The interactive web interface.
-   **[WhisperX](https://github.com/m-bain/whisperX)**: State-of-the-art speech recognition with forced alignment.
-   **[MoviePy](https://zulko.github.io/moviepy/)**: Video and audio editing.
-   **[Better Profanity](https://github.com/snguyenthanh/better_profanity)**: Fast profanity detection.
-   **[FFmpeg](https://ffmpeg.org/)**: The backbone of media processing.

---

## 🚀 Getting Started

### Prerequisites

1.  **Python 3.8+** installed.
2.  **FFmpeg** installed and added to your system PATH.
    *   *Windows*: Download from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/), extract, and add `bin` to PATH.
3.  **CUDA (Optional)**: For faster processing if you have an NVIDIA GPU.

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/sush-project.git
    cd sush-project
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: You may need to install `torch` separately compatible with your CUDA version.*

3.  **Run the App**:
    *   **Easy Mode**: Double-click `run_app.bat`.
    *   **Manual**:
        ```bash
        streamlit run Home.py
        ```

---

## 📖 How to Use

1.  **Select "Use Sush"** from the sidebar.
2.  **Upload** your video or audio file.
3.  Click **"Transcribe & Censor"**.
4.  Wait for the AI to work its magic ✨.
5.  **Preview** the result and **Download** your clean file!

> **Tip:** You can edit the list of banned words in the "Manage Cuss Words" section.

---

## 📂 Project Structure

```
SUSH/
├── Home.py                 # Main entry point
├── pages/
│   ├── 1_Use_Sush_📜.py    # Core censorship tool
│   ├── 2_Code_Sush_🐍.py   # Code explanation
│   └── 3_About_Ish_👀.py   # About the author
├── cuss_words.txt          # Custom list of banned words
├── requirements.txt        # Python dependencies
└── run_app.bat             # Quick start script
```

---

## 👨‍💻 Author

**Shaik Mahammad Ishrath (Ish)**
*Curious Engineer | Tech, Design, & Storytelling*

Built as a major project to explore the intersection of AI and media processing.

---

*Made with ❤️ and 🐍 Python.*
