import streamlit as st

st.set_page_config(page_title="SUSH - Code", layout="centered", page_icon="🐍")

st.title("🐍 You can also do SUSH !")
st.divider()
# Title or Heading

# Informative note before code
st.info("""
**Note Before Reading the Code:**

This implementation is based on the logic and thought process outlined in the Notion document:

🔗 [Sush Profanity Filter - Notes by Ishrath(Me)](https://verbose-apogee-f7b.notion.site/Sush-Profanity-Filter-20039954225380d086b2ed5bd93de4a5)

I wrote all this while making this project, so read this once u can understand the flow the project.


""")
st.caption("Here’s the core code behind the SUSH censorship tool. Feel free to explore and learn how it works.")

# You can store your full code in a multiline string
code = '''
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.audio.AudioClip import AudioClip, concatenate_audioclips
import speech_recognition as sr
import numpy as np
import os
import math

# --- Configuration ---
INPUT_VIDEO_PATH = "input_video.mp4"  # Replace with your video file
OUTPUT_VIDEO_PATH = "output_video_censored.mp4"
TEMP_AUDIO_PATH = "temp_extracted_audio.wav"
CUSS_WORDS_FILE = "cuss_words.txt"
BEEP_FREQUENCY = 1000  # Hz
BEEP_VOLUME = 0.5      # 0.0 to 1.0

# --- Load Cuss Words ---
def load_cuss_words(filepath):
    try:
        with open(filepath, 'r') as f:
            words = [line.strip().lower() for line in f if line.strip()]
        if not words:
            print(f"Warning: Cuss word file '{filepath}' is empty or contains only whitespace.")
        return set(words)
    except FileNotFoundError:
        print(f"Error: Cuss word file '{filepath}' not found. No words will be censored.")
        return set()

CUSS_WORDS = load_cuss_words(CUSS_WORDS_FILE)
if not CUSS_WORDS:
    print("No cuss words loaded. The video will not be censored.")

# --- Audio & Video Functions ---
def extract_audio(video_path, audio_output_path):
    print(f"Extracting audio from '{video_path}'...")
    try:
        video_clip = VideoFileClip(video_path)
        audio_clip = video_clip.audio
        if audio_clip is None:
            print(f"Error: Video '{video_path}' has no audio track.")
            return None
        audio_clip.write_audiofile(audio_output_path, codec='pcm_s16le')
        audio_clip.close()
        video_clip.close()
        print(f"Audio extracted to '{audio_output_path}'")
        return audio_output_path
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None

def transcribe_audio_with_timestamps(audio_path):
    print(f"Transcribing audio from '{audio_path}' (this may take a while)...")
    recognizer = sr.Recognizer()
    word_timings = []

    with sr.AudioFile(audio_path) as source:
        try:
            audio_data = recognizer.record(source)
            results = recognizer.recognize_google(audio_data, show_all=True)

            if results and 'alternative' in results and results['alternative']:
                best_transcript = results['alternative'][0]
                if 'words' in best_transcript:
                    for word_info in best_transcript['words']:
                        word = word_info['word'].lower()
                        start_time = float(word_info.get('startTime', '0s').replace('s', ''))
                        end_time = float(word_info.get('endTime', '0s').replace('s', ''))
                        word_timings.append({"word": word, "start": start_time, "end": end_time})

        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
        except Exception as e:
            print(f"An unexpected error occurred during transcription: {e}")

    if word_timings:
        print(f"Transcription complete. Found {len(word_timings)} words with timings.")
    else:
        print("Transcription failed or no word timings could be extracted.")
    return word_timings

def find_cuss_word_segments(word_timings, cuss_words_set):
    cuss_segments = []
    if not word_timings:
        print("No word timings available to find cuss words.")
        return cuss_segments

    print("Scanning for cuss words...")
    for entry in word_timings:
        if entry["word"] in cuss_words_set:
            print(f"Found cuss word: '{entry['word']}' from {entry['start']:.2f}s to {entry['end']:.2f}s")
            cuss_segments.append({"start": entry["start"], "end": entry["end"], "word": entry["word"]})
    
    if not cuss_segments:
        print("No cuss words found in transcription.")
    return cuss_segments

def make_beep_sound(duration, freq=1000, fps=44100, volume=0.5):
    n_samples = int(duration * fps)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    wave = volume * np.sin(2 * np.pi * freq * t)
    beep_wave = np.array(list(zip(wave, wave)))

    return AudioClip(lambda t_param: beep_wave[int(t_param * fps)] if 0 <= t_param * fps < len(beep_wave) else [0,0],
                     duration=duration, fps=fps)

def create_censored_audio(original_audio_path, cuss_segments, beep_freq, beep_vol):
    print("Creating censored audio track...")
    if not cuss_segments:
        print("No cuss words to censor. Using original audio.")
        return AudioFileClip(original_audio_path)

    original_audio = AudioFileClip(original_audio_path)
    final_audio_duration = original_audio.duration
    audio_clips = []
    current_time = 0.0
    cuss_segments.sort(key=lambda x: x['start'])

    for segment in cuss_segments:
        start_cuss = max(0, segment["start"])
        end_cuss = min(segment["end"], final_audio_duration)
        if start_cuss >= end_cuss:
            continue
        if start_cuss > current_time:
            try:
                clip_before = original_audio.subclip(current_time, start_cuss)
                audio_clips.append(clip_before)
            except Exception as e:
                print(f"Error sub-clipping audio before cuss: {current_time} to {start_cuss}. Error: {e}")

        beep_duration = end_cuss - start_cuss
        if beep_duration > 0:
            beep_sound = make_beep_sound(duration=beep_duration, freq=beep_freq, fps=original_audio.fps, volume=beep_vol)
            audio_clips.append(beep_sound)
            print(f"Added beep from {start_cuss:.2f}s to {end_cuss:.2f}s")
        current_time = end_cuss

    if current_time < final_audio_duration:
        try:
            clip_after_last = original_audio.subclip(current_time, final_audio_duration)
            audio_clips.append(clip_after_last)
        except Exception as e:
            print(f"Error sub-clipping audio after last cuss: {current_time} to {final_audio_duration}. Error: {e}")

    if not audio_clips:
        print("No valid audio clips generated for censoring. Returning original audio.")
        original_audio.close()
        return AudioFileClip(original_audio_path)

    try:
        censored_audio = concatenate_audioclips(audio_clips)
        print("Audio clips concatenated successfully.")
    except Exception as e:
        print(f"Error concatenating audio clips: {e}. Returning original audio.")
        original_audio.close()
        return AudioFileClip(original_audio_path)

    original_audio.close()
    return censored_audio

def combine_video_and_audio(video_path, audio_clip, output_path):
    print(f"Combining video from '{video_path}' with new audio...")
    try:
        video_clip = VideoFileClip(video_path)
        if audio_clip.duration > video_clip.duration:
            audio_clip = audio_clip.subclip(0, video_clip.duration)
        final_clip = video_clip.set_audio(audio_clip)
        final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac",
                                   temp_audiofile='temp-audio.m4a', remove_temp=True,
                                   threads=4, preset='medium')
        print(f"Censored video saved to '{output_path}'")
        video_clip.close()
        if hasattr(audio_clip, 'close'):
            audio_clip.close()
        final_clip.close()
    except Exception as e:
        print(f"Error combining video and audio: {e}")

def cleanup_temp_files(*files):
    for f_path in files:
        if os.path.exists(f_path):
            try:
                os.remove(f_path)
                print(f"Removed: {f_path}")
            except Exception as e:
                print(f"Error removing {f_path}: {e}")

# --- Main ---
def main():
    print("--- Starting Sush Censorship Project ---")
    if not os.path.exists(INPUT_VIDEO_PATH):
        print(f"Error: Input video file not found: {INPUT_VIDEO_PATH}")
        return

    extracted_audio_file = extract_audio(INPUT_VIDEO_PATH, TEMP_AUDIO_PATH)
    if not extracted_audio_file:
        print("Failed to extract audio. Exiting.")
        return

    word_timings_data = transcribe_audio_with_timestamps(extracted_audio_file)
    if not word_timings_data:
        print("Transcription failed or no timings obtained. Using original audio.")
        original_video_clip = VideoFileClip(INPUT_VIDEO_PATH)
        final_audio_for_video = original_video_clip.audio
    else:
        cuss_word_segments = find_cuss_word_segments(word_timings_data, CUSS_WORDS)
        if cuss_word_segments:
            final_audio_for_video = create_censored_audio(extracted_audio_file, cuss_word_segments, BEEP_FREQUENCY, BEEP_VOLUME)
        else:
            print("No cuss words found. Using original audio.")
            final_audio_for_video = AudioFileClip(extracted_audio_file)

    if final_audio_for_video:
        combine_video_and_audio(INPUT_VIDEO_PATH, final_audio_for_video, OUTPUT_VIDEO_PATH)
    else:
        video_clip_no_audio = VideoFileClip(INPUT_VIDEO_PATH).without_audio()
        video_clip_no_audio.write_videofile(OUTPUT_VIDEO_PATH, codec="libx264", threads=4, preset='medium')
        video_clip_no_audio.close()

    cleanup_temp_files(TEMP_AUDIO_PATH)
    if 'temp-audio.m4a' in os.listdir('.'):
        cleanup_temp_files('temp-audio.m4a')

    print("--- Sush Censorship Project Finished ---")

if __name__ == "__main__":
    main()

'''

# Display with syntax highlighting
st.code(code, language='python')

st.info("🛠️ For the full working version, explore the actual script file in your repo or folder.")

