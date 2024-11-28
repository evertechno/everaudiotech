import streamlit as st
import google.generativeai as genai
import os
from pydub import AudioSegment
import tempfile
import torch
from transformers import pipeline

# Set up Google API configuration
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Title and description
st.title("Ever AI - Voice Cloning")
st.write("Use generative AI for voice cloning. Upload your audio file, and we will transcribe and clone your voice.")

# Function to transcribe audio using Whisper (OpenAI's model)
def transcribe_audio(audio_file):
    # Load the Whisper transcription pipeline
    transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-large")

    # Process audio file
    transcription = transcriber(audio_file)
    return transcription['text']

# Handle audio file upload
audio_file = st.file_uploader("Upload an Audio File", type=["mp3", "wav", "ogg"])

if audio_file is not None:
    # Temporary save the uploaded audio file to process
    with tempfile.NamedTemporaryFile(delete=False) as tmp_audio_file:
        tmp_audio_file.write(audio_file.read())
        tmp_audio_file_path = tmp_audio_file.name
        
    st.audio(tmp_audio_file_path, format="audio/wav")

    # Transcribe the audio file to text
    try:
        transcription = transcribe_audio(tmp_audio_file_path)
        st.write("Transcription:")
        st.write(transcription)
    except Exception as e:
        st.error(f"Error in transcription: {e}")
    
    # Allow users to enter their text prompt for voice cloning
    text_prompt = st.text_input("Enter your text to clone the voice", transcription)

    if st.button("Clone Voice"):
        try:
            # Using the Open-Source Coqui TTS or other TTS library for voice synthesis
            # Assuming you have a pre-trained model or API for voice synthesis
            # Here we are assuming Coqui TTS, but you can integrate another API

            # You can install Coqui TTS with pip, e.g., `pip install TTS`
            from TTS.utils.synthesizer import Synthesizer
            from TTS.config import load_config
            from TTS.utils.generic_utils import download_model

            # Setup model
            model_name = "tts_models/en/ljspeech/tacotron2-DDC"
            config_path = download_model(model_name)
            config = load_config(config_path)
            synthesizer = Synthesizer(config)
            
            # Create speech synthesis with the transcribed text
            audio_output = synthesizer.tts(text_prompt)
            temp_output_path = tempfile.mktemp(suffix=".wav")
            synthesizer.save_wav(audio_output, temp_output_path)
            
            # Play the cloned voice audio
            st.audio(temp_output_path, format="audio/wav")
            st.success("Voice Cloning Complete!")

        except Exception as e:
            st.error(f"Error in voice cloning: {e}")

else:
    st.write("Please upload an audio file to get started.")
