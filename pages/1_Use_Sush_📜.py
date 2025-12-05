import streamlit as st
import whisperx
from pydub import AudioSegment
from pydub.generators import Sine
from better_profanity import profanity
import tempfile
import os
import torch
import subprocess # For calling ffmpeg

# --- Page Config ---
st.set_page_config(page_title="SUSH! - Censor Tool", layout="centered", page_icon="🤫")

# --- WhisperX Model Loading (Cached) ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "int8"
# Using "tiny.en" or "base.en" can be faster if English is guaranteed and sufficient.
# For more general use and better accuracy on diverse audio, "small" is a good balance.
ASR_MODEL_NAME = "small" # Changed back to small for better accuracy, tiny can be too inaccurate
BATCH_SIZE = 16 # Adjust based on GPU memory if using GPU

@st.cache_resource
def load_asr_model(model_name=ASR_MODEL_NAME):
    print(f"Loading ASR model '{model_name}' on {DEVICE} with compute_type {COMPUTE_TYPE}...")
    # Specifying language="en" here can pre-select English models if available for faster loading
    # and potentially more accurate alignment for English-only content.
    # If you need multi-language, remove language="en" or pass None.
    model = whisperx.load_model(model_name, DEVICE, compute_type=COMPUTE_TYPE, language="en")
    print(f"ASR model '{model_name}' loaded.")
    return model

@st.cache_resource
def load_align_model(language_code="en"):
    print(f"Loading alignment model for language '{language_code}' on {DEVICE}...")
    # Ensure language_code is valid; WhisperX might throw an error for unsupported codes.
    # Defaulting to 'en' if language detection is not robust or for English-only apps.
    try:
        model_a, metadata_a = whisperx.load_align_model(language_code=language_code, device=DEVICE)
        print(f"Alignment model for '{language_code}' loaded.")
    except Exception as e:
        st.warning(f"Could not load alignment model for '{language_code}', defaulting to 'en'. Error: {e}")
        model_a, metadata_a = whisperx.load_align_model(language_code="en", device=DEVICE) # Fallback
        print("Alignment model for 'en' loaded as fallback.")
    return model_a, metadata_a

# --- Custom Cuss Words Logic ---
CUSS_WORDS_FILE = "cuss_words.txt"

def load_cuss_words():
    if not os.path.exists(CUSS_WORDS_FILE):
        return []
    with open(CUSS_WORDS_FILE, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def save_cuss_words(words_list):
    with open(CUSS_WORDS_FILE, "w") as f:
        f.write("\n".join(words_list))

# --- UI Elements ---
st.title("📜 How to use - SUSH!")
st.caption("Censor profanity in your audio and video files.")
st.divider()

# --- Cuss Words Management ---
with st.expander("📝 Manage Cuss Words"):
    current_words = load_cuss_words()
    st.write("Current list of words to censor:")
    
    # Text area to edit words
    words_input = st.text_area("Edit words (one per line)", value="\n".join(current_words), height=150)
    
    if st.button("Save Cuss Words"):
        new_words = [w.strip() for w in words_input.split('\n') if w.strip()]
        save_cuss_words(new_words)
        st.success("✅ Cuss words updated!")
        current_words = new_words # Update local variable

st.markdown('''1.Upload your file.  
            2.Click Transcribe & Detect.  
            3.Click Censor & Download to get the clean audio!''')

st.subheader("Step 1: Upload Your File", divider="blue")
# Combined uploader for audio and video
input_file = st.file_uploader(
    "Choose your uncensored audio or video file",
    type=['mp3', 'wav', 'm4a', 'ogg', 'mp4', 'mov', 'mkv', 'avi', 'webm'] # Added more video types
)

if input_file:
    file_info = {"name": input_file.name, "type": input_file.type, "size": input_file.size}
    file_category = file_info["type"].split('/')[0] # 'audio' or 'video'

    # Display uploaded file
    if file_category == "video":
        st.video(input_file)
    else: # audio
        st.audio(input_file, format=file_info["type"])

    # Suggestion for long videos
    MAX_VIDEO_DURATION_FOR_AUTO_PLAY_PREVIEW_S = 60 # seconds
    # This is a placeholder. Getting actual video duration without fully processing is tricky with just streamlit.
    # For a real check, you'd need ffprobe (part of ffmpeg) or a library that can read metadata.
    # For now, we can use file size as a rough proxy.
    IS_LARGE_VIDEO = file_category == "video" and file_info["size"] > 50 * 1024 * 1024 # 50MB threshold

    if IS_LARGE_VIDEO:
        st.info("ℹ️ This looks like a large video. Processing and preview might take a while. "
                "Consider using a shorter clip for quicker testing if needed.")

    st.subheader("Step 2: Transcribe & Censor", divider="blue")

    if st.button("Transcribe & Censor", use_container_width=True, type="primary"):
        # Initialize paths
        temp_input_path = None
        temp_audio_to_process_path = None
        temp_censored_audio_path = None
        temp_final_output_path = None # For the final video or audio to be downloaded

        try:
            with st.spinner("Hold tight! SUSH!-ing your content... ⏳"):
                # 1. Save uploaded file to a temporary path
                file_extension = os.path.splitext(file_info["name"])[1]
                print("file_info: "+file_info["name"])
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_raw_file:
                    temp_raw_file.write(input_file.getvalue())
                    temp_input_path = temp_raw_file.name

                # 2. Prepare audio for processing
                if file_category == "video":
                    st.write("🎞️ Extracting audio from video...")
                    # Use a .wav suffix for the extracted audio for best quality with pydub/whisper
                    temp_audio_to_process_path = tempfile.mktemp(suffix=".wav")
                    ffmpeg_command = [
                        "ffmpeg", "-y",
                        "-i", temp_input_path,
                        "-vn",             # No video
                        "-acodec", "pcm_s16le", # WAV format
                        "-ar", "16000",    # Sample rate Whisper prefers
                        "-ac", "1",        # Mono channel
                        temp_audio_to_process_path
                    ]
                    process = subprocess.run(ffmpeg_command, capture_output=True, text=True, check=False)
                    if process.returncode != 0:
                        st.error(f"FFmpeg audio extraction failed: {process.stderr}")
                        raise Exception(f"FFmpeg audio extraction error: {process.stderr}")
                    st.write("🔊 Audio extracted successfully.")
                else: # It's an audio file
                    # We might still want to convert to WAV 16kHz mono for consistency with WhisperX
                    st.write("🔊 Preparing audio...")
                    temp_audio_to_process_path = tempfile.mktemp(suffix=".wav")
                    ffmpeg_command = [
                        "ffmpeg", "-y",
                        "-i", temp_input_path,
                        "-acodec", "pcm_s16le",
                        "-ar", "16000",
                        "-ac", "1",
                        temp_audio_to_process_path
                    ]
                    process = subprocess.run(ffmpeg_command, capture_output=True, text=True, check=False)
                    if process.returncode != 0:
                        st.error(f"FFmpeg audio conversion failed: {process.stderr}")
                        # Fallback to using the original audio if conversion fails
                        st.warning("Audio conversion failed, attempting to use original audio format (may affect accuracy).")
                        temp_audio_to_process_path = temp_input_path # Use original if conversion fails
                    else:
                        st.write("🔊 Audio prepared for processing.")


                # 3. Transcribe
                st.write("🎤 Transcribing audio (this may take a moment)...")
                
                asr_model_instance = load_asr_model() # Use default ASR_MODEL_NAME or specify
                transcription_result = asr_model_instance.transcribe(temp_audio_to_process_path, batch_size=BATCH_SIZE, language="en")
                detected_language = transcription_result.get("language", "en") # Default to 'en'
                st.write(f"💬 Detected language: {detected_language.upper()}")

                full_transcript = " ".join([seg["text"] for seg in transcription_result["segments"]]).strip()
                if not full_transcript:
                    st.warning("⚠️ Could not transcribe any text from the audio.")
                    if file_category == "video": # Still provide original video for download
                         st.info("Offering original video for download.")
                         input_file.seek(0)
                         st.download_button(label="📥 Download Original Video", data=input_file.read(), file_name=file_info["name"], mime=file_info["type"], use_container_width=True)
                    st.stop() # Stop processing if no transcript


                # 4. Profanity Check
                st.write("🧐 Checking for profanity...")
                
                # Load custom words
                custom_cuss_words = load_cuss_words()
                profanity.load_censor_words(custom_cuss_words) # Load CUSTOM word list into better_profanity
                
                contains_profanity_bool = profanity.contains_profanity(full_transcript)

                if not contains_profanity_bool:
                    st.success("✅ Hooray! No profanity detected in the transcript.")
                    st.subheader(f"Step 3: Download Your Original {file_category.capitalize()}", divider="blue")
                    input_file.seek(0) # Reset pointer
                    st.download_button(
                        label=f"📥 Download Original {file_category.capitalize()}",
                        data=input_file.read(),
                        file_name=f"original_{file_info['name']}",
                        mime=file_info["type"],
                        use_container_width=True
                    )
                else:
                    st.info("🚨 Profanity detected! Proceeding with alignment and censoring...")

                    # 5. Align (only if profanity found)
                    st.write("🔄 Aligning transcript for precise timestamps...")
                    align_model_instance, align_metadata_instance = load_align_model(detected_language)
                    aligned_result = whisperx.align(
                        transcription_result["segments"],
                        align_model_instance,
                        align_metadata_instance,
                        temp_audio_to_process_path,
                        DEVICE
                    )

                    # 6. Censor Audio with Pydub
                    st.write("🔇 Applying beeps to audio...")
                    # Load the (potentially converted) audio file that was transcribed/aligned
                    audio_segment = AudioSegment.from_file(temp_audio_to_process_path)
                    censored_audio_segment = audio_segment
                    
                    profane_words_beeped_count = 0
                    pre_beep_pad_ms = 200
                    post_beep_pad_ms = 100

                    for seg in aligned_result["segments"]:
                        for word_data in seg.get("words", []):
                            word = word_data.get("word", "")
                            # Check if this specific word is profane using the loaded custom list
                            if profanity.contains_profanity(word): 
                                if 'start' in word_data and 'end' in word_data:
                                    start_ms = int(word_data['start'] * 1000)
                                    end_ms = int(word_data['end'] * 1000)

                                    beep_start = max(0, start_ms - pre_beep_pad_ms)
                                    beep_end = end_ms + post_beep_pad_ms

                                    if beep_end > beep_start:
                                        beep_duration = beep_end - beep_start
                                        beep_sound = Sine(400).to_audio_segment(duration=beep_duration).apply_gain(-5)
                                        censored_audio_segment = censored_audio_segment[:beep_start] + beep_sound + censored_audio_segment[beep_end:]
                                        profane_words_beeped_count += 1
                                    else:
                                        st.caption(f"Skipped beep for '{word}' (invalid duration after padding).")
                                else:
                                    st.caption(f"Skipped beep for '{word}' (missing timestamps).")
                    
                    if profane_words_beeped_count == 0:
                        st.warning("🤔 Profanity was flagged in the full transcript, but no specific words were beeped (e.g., due to timestamp issues or multi-word phrases not caught by word-level check).")
                        st.subheader(f"Step 3: Download Your Original {file_category.capitalize()}", divider="blue")
                        st.info("Offering original file as no beeps were applied.")
                        input_file.seek(0)
                        st.download_button(
                            label=f"📥 Download Original {file_category.capitalize()} (No Beeps Applied)",
                            data=input_file.read(),
                            file_name=f"original_nobeeep_{file_info['name']}",
                            mime=file_info["type"],
                            use_container_width=True
                        )
                    else:
                        st.success(f"🔊 Censored audio generated with {profane_words_beeped_count} beep(s).")
                        # Save the censored audio segment
                        temp_censored_audio_path = tempfile.mktemp(suffix=".mp3") # Use mp3 for broad compatibility
                        censored_audio_segment.export(temp_censored_audio_path, format="mp3")

                        # 7. Prepare final output (merge for video, or use censored audio directly)
                        st.subheader(f"Step 3: Preview and Download Your Censored {file_category.capitalize()}", divider="blue")
                        
                        output_display_path = None
                        output_download_name = f"SUSHED_{file_info['name']}"
                        output_mime_type = file_info["type"] # Default to original type

                        if file_category == "video":
                            st.write("🎞️ Stitching censored audio back into video...")
                            # Ensure the output video name has a common video extension like .mp4
                            base_name, _ = os.path.splitext(output_download_name)
                            output_download_name = base_name + ".mp4"
                            output_mime_type = "video/mp4"

                            temp_final_output_path = tempfile.mktemp(suffix=".mp4")
                            ffmpeg_merge_command = [
                                "ffmpeg", "-y",
                                "-i", temp_input_path,             # Original video (for video stream)
                                "-i", temp_censored_audio_path,    # New censored audio
                                "-c:v", "copy",                    # Copy video stream without re-encoding (fast)
                                "-c:a", "aac",                     # Re-encode audio to AAC (common for mp4)
                                "-map", "0:v:0",                   # Map video from first input
                                "-map", "1:a:0",                   # Map audio from second input
                                "-shortest",                       # Finish when shortest input ends
                                temp_final_output_path
                            ]
                            merge_process = subprocess.run(ffmpeg_merge_command, capture_output=True, text=True, check=False)
                            if merge_process.returncode != 0:
                                st.error(f"FFmpeg video merge failed: {merge_process.stderr}")
                                raise Exception(f"FFmpeg video merge error: {merge_process.stderr}")
                            output_display_path = temp_final_output_path
                            st.write("✅ Video successfully SUSH!-ed!")
                        else: # Audio file
                            output_display_path = temp_censored_audio_path # This is already an mp3
                            output_mime_type = "audio/mpeg" # For mp3
                            base_name, _ = os.path.splitext(output_download_name)
                            output_download_name = base_name + ".mp3"
                            st.write("✅ Audio successfully SUSH!-ed!")

                        # Preview
                        if output_display_path:
                            with open(output_display_path, "rb") as f_preview:
                                preview_bytes = f_preview.read()
                                if file_category == "video":
                                    # Only autoplay short videos for preview to save bandwidth/processing
                                    if not IS_LARGE_VIDEO: # Using the earlier rough check
                                        st.video(preview_bytes)
                                    else:
                                        st.info("Video preview is not autoplayed for large files. Please download to view.")
                                else: # audio
                                    st.audio(preview_bytes, format=output_mime_type)
                        
                        # Download button for the final processed file
                        with open(output_display_path, "rb") as f_download:
                            st.download_button(
                                label=f"📥 Download Censored {file_category.capitalize()}",
                                data=f_download.read(),
                                file_name=output_download_name,
                                mime=output_mime_type,
                                use_container_width=True
                            )

        except Exception as e:
            st.error(f"💥 Oops! An error occurred: {str(e)}")
            # import traceback
            # st.error(f"Traceback: {traceback.format_exc()}") # Uncomment for more detailed error for debugging
        finally:
            # Clean up all temporary files
            paths_to_clean = [
                temp_input_path,
                temp_audio_to_process_path,
                temp_censored_audio_path,
                temp_final_output_path
            ]
            for path in paths_to_clean:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception as ex_clean:
                        st.caption(f"Note: Could not delete temporary file {os.path.basename(path)}: {ex_clean}")
