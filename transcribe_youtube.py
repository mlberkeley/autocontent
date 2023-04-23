import argparse
import os
import yt_dlp
import openai
import io
import math
from pydub import AudioSegment

# Get the OpenAI API key from an environment variable
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Download the YouTube video's audio
def download_audio(video_url, filename):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': filename.replace('.wav', '') + '.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

# Convert the audio to the required format (WAV, 16kHz, mono)
def convert_audio(input_file, output_file, start=0, end=0):
    audio = AudioSegment.from_wav(input_file)
    # Save the audio segment if start and end times are specified
    if start >= 0 and end > 0:
        # Convert start and end times from seconds to milliseconds
        start = start * 1000
        end = end * 1000
        audio = audio[start:end]
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(output_file, format="wav")

# Transcribe the audio using the Whisper API
def transcribe_audio(audio_file):
    with io.open(audio_file, "rb") as audio_data:
        response = openai.Audio.transcribe(
            file=audio_data,
            model="whisper-1",
            # Specific language in ISO-639-1 format
            language="en"
        )

    transcript = response.get("text")
    return transcript

# Set the path to FFmpeg executable
AudioSegment.converter = "/path/to/ffmpeg"
AudioSegment.ffmpeg = "/path/to/ffmpeg"

# Increase the maximum memory limit for loading audio files (in bytes)
AudioSegment.MAX_MEMORY_FOR_PIL = 800 * 1024 * 1024  # 800 MB

# Define the argument parser
parser = argparse.ArgumentParser(description="Transcribe a YouTube video using the OpenAI Whisper ASR API.")
parser.add_argument("video_url", help="The YouTube video URL to transcribe.")

# Parse the command-line arguments
args = parser.parse_args()

# Define temporary file names
downloaded_audio = "downloaded_audio.wav"
converted_audio = "converted_audio.wav"

# Download and convert the audio
download_audio(args.video_url, downloaded_audio)

# Get the total duration of the audio
audio = AudioSegment.from_wav(downloaded_audio)
duration = audio.duration_seconds

num_segments = math.ceil(duration / 600)
for i in range(num_segments):
    start = i * 600
    end = (i + 1) * 600
    if end > duration:
        end = duration
    convert_audio(downloaded_audio, converted_audio, start=start, end=end)
    transcript = transcribe_audio(converted_audio)
    with open("transcript.txt", "a") as f:
        f.write(transcript)