import argparse
import os
import yt_dlp
import openai
import io
import math
import re
from pydub import AudioSegment
from moviepy.editor import VideoFileClip


# Get the OpenAI API key from an environment variable
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Check if the input is a YouTube URL
def is_youtube_url(url):
    youtube_regex = re.compile(
        r'(https?://)?(www\.)?'
        '(youtube|youtu|youtube-nocookie)\.(com|be)/'
        '(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')

    return bool(youtube_regex.match(url))

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

# Extract audio from local video file
def extract_audio_from_video(video_file, output_file):
    video = VideoFileClip(video_file)
    audio = video.audio
    audio.write_audiofile(output_file)

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
parser = argparse.ArgumentParser(description="Transcribe a YouTube video or local MP4 file using the OpenAI Whisper ASR API.")
parser.add_argument("input", help="The YouTube video URL or local MP4 file to transcribe.")

# Parse the command-line arguments
args = parser.parse_args()

# Define temporary file names
input_audio = "input_audio.wav"
converted_audio = "converted_audio.wav"

# Check if the input is a YouTube URL or a local MP4 file
if is_youtube_url(args.input):
    # Download and convert the audio from YouTube video
    download_audio(args.input, input_audio)
else:
    # Extract and convert the audio from local MP4 file
    extract_audio_from_video(args.input, input_audio)

# Get the total duration of the audio
audio = AudioSegment.from_wav(input_audio)
duration = audio.duration_seconds

num_segments = math.ceil(duration / 600)
# Remove transcript.txt if it already exists
if os.path.exists("transcript.txt"):
    os.remove("transcript.txt")
for i in range(num_segments):
    start = i * 600
    end = (i + 1) * 600
    if end > duration:
        end = duration
    convert_audio(input_audio, converted_audio, start=start, end=end)
    transcript = transcribe_audio(converted_audio)
    with open("transcript.txt", "a+") as f:
        f.write(transcript)