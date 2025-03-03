import time
import wave
import numpy as np
import torch
import pyaudio
from faster_whisper import WhisperModel
from groq import Groq
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import save
import os
from flask import Flask, request, jsonify
from flask_socketio import SocketIO
from flask_cors import CORS
import threading
import tempfile
import io
from pydub import AudioSegment
from pydub.playback import play as pydub_play

# Initialize Flask app
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load environment variables
load_dotenv()

system_prompt = """
آپ کو گندم کے کاشتکاروں کے لیے اہم پیغامات فراہم کرنے ہیں۔ ان پیغامات میں گندم کی کاشت، پانی دینے کے اوقات، بیج کی اقسام، جڑی بوٹیوں کا کنٹرول، بیماریوں سے بچاؤ، اور دیگر زرعی مشورے شامل ہیں۔ یہ پیغامات مختلف وقتوں، حالات اور جغرافیائی مقامات کے لیے مخصوص ہیں اور کسانوں کو گندم کی بہتر پیداوار کے لیے رہنمائی فراہم کرتے ہیں۔

گندم کی کاشت کا موزوں وقت یکم سے 15 نومبر ہے۔
15 نومبر تک بوائی کے لیے بیج کی مقدار 50 کلوگرام فی ایکڑ رکھیں، بعد میں 60 کلوگرام فی ایکڑ۔
بروقت کاشت کے لیے کپاس، مکئی، اور کماد کے کھیتوں میں پانی بند کرنے کے بعد روٹا ویٹر اور دوہرا ہل چلا کر گندم کی بوائی کریں۔
بیج کی اقسام کی سفارشیں: بارانی علاقوں میں جی اے 2002، عقاب 2000، این اے ار سی 2009، اور آبپاش علاقوں میں عقاب 2000، پنجند 1 وغیرہ۔
زمین کی تیاری کے لیے ہل چلانے کی تعداد: وریال زمینوں میں چار پانچ دفعہ اور بھاری زمین میں دو بار ہل چلائیں۔
بیج کو گریڈ کروا کر اور زہر لگانے کے بعد کاشت کریں۔
پہلی آبپاشی کے بعد 2ہریرو چلائیں، اور بارانی علاقوں میں بارش کی صورت میں پانی ذخیرہ کرنے کے لیے گہرا ہل چلائیں۔
فصل پر سست تیلے کے حملے کے لیے پودوں کا معائنہ کریں اور متاثرہ حصوں میں تیلے کو کنٹرول کریں۔
جڑی بوٹیوں کا کنٹرول: دھند، تیز ہوا، اور بارش میں سپرے نہ کریں، اور سپرے کے لیے فلیٹ فین نوزل کا استعمال کریں۔
گندم کی فصل کی کٹائی موسمی حالات کو مدنظر رکھتے ہوئے کریں اور کٹائی کے وقت موسمی پیشگوئی پر توجہ دیں۔

---
User Query (Example):
مجھے یہ بتاؤ کہ گندم میں کیڑے مارنے کے لیے کون سی زہر یوز کرنی چاہیے۔

Response Structure:
جب صارف اس موضوع پر سوال کرے گا، آپ کو اس کو گندم کی فصل میں کیڑے مار دواؤں کے بارے میں معلومات فراہم کرنی ہیں۔ اس میں مختلف کیڑے مار دواؤں کی قسمیں شامل ہوں گی جو گندم کی فصل کو مختلف کیڑوں سے بچانے میں مدد کرتی ہیں۔

### Suggested Insecticides for Wheat:
ایندوسلفن: یہ ایک وسیع الطیف کیڑے مار دوائی ہے جو مختلف قسم کے کیڑوں کے خلاف موثر ہے، بشمول سست تیلے، مکڑی، اور دوسرے افات کیڑے۔
سیالوٹرن: یہ دوائی بھی مختلف قسم کے کیڑوں کے خلاف موثر ہے، خاص طور پر سست تیلے اور دوسرے افات کیڑوں کے لیے۔
ایمیٹاف: یہ دوائی خاص طور پر سست تیلے اور دوسرے افات کیڑوں کے لیے موثر ہے۔
ٹیوفنوٹ: یہ دوائی بھی سست تیلے اور دوسرے افات کیڑوں کے خلاف موثر ہے۔
ڈائی فلورین: یہ دوائی مختلف قسم کے کیڑوں کے خلاف موثر ہے، بشمول سست تیلے، مکڑی، اور دوسرے افات کیڑے۔
### Important Notes:
کیڑے مار دواؤں کا استعمال صرف ضرورت کے مطابق کیا جائے۔
کسی بھی کیڑے مار دوائی کا استعمال کرنے سے پہلے اس کی مقدار اور طریقہ کار کو اچھی طرح سے سمجھیں۔
فصلوں پر کیڑوں کا حملہ ہونے پر فوری اقدامات کریں تاکہ فصل کی پیداوار پر منفی اثر نہ پڑے۔
---
یہ معلومات کسانوں کو گندم کی فصل کے کیڑوں سے بچاؤ کے لیے مفید رہنمائی فراہم کرے گی۔
#Your responses should be very short,concise and to the point and make sure you are a helpful AI assistant who guides about agriculture to the farmers.
"""

# Initialize clients
try:
    tts_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY", "sk_5207215d199ad1a8e2bafedd6affda66b16cf53f472f02dc"))
    client = Groq(api_key=os.getenv("GROQ_API_KEY", "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"))
    print("API clients initialized successfully")
except Exception as e:
    print(f"Error initializing API clients: {e}")

# Load Silero VAD model and utility functions
try:
    vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True)
    (get_speech_timestamps, _, _, _, _) = utils
    print("VAD model loaded successfully")
except Exception as e:
    print(f"Error loading VAD model: {e}")
    vad_model = None
    utils = None

# Initialize Whisper model
try:
    model_size = "large-v3-turbo"
    # Check if CUDA is available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
    print(f"Whisper model loaded successfully on {device}")
except Exception as e:
    print(f"Error loading Whisper model: {e}")
    whisper_model = None

# Audio recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 512

# Global variables
is_listening = False
audio_buffer = []
last_voice_time = time.time()
silence_threshold = 2.0
audio_interface = None
stream = None
is_processing = False  # Flag to indicate if we're processing audio or playing response
processing_lock = threading.Lock()  # Lock to synchronize access to processing state

def initialize_audio():
    global audio_interface, stream
    try:
        audio_interface = pyaudio.PyAudio()
        stream = audio_interface.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        print("Audio interface initialized successfully")
        return True
    except Exception as e:
        print(f"Error initializing audio interface: {e}")
        return False

def cleanup_audio():
    global audio_interface, stream
    if stream:
        stream.stop_stream()
        stream.close()
    if audio_interface:
        audio_interface.terminate()
    print("Audio resources cleaned up")

def process_audio_data(data):
    global last_voice_time, audio_buffer, is_processing
    
    # If we're currently processing audio or playing a response, don't process new audio
    if is_processing:
        return False
    
    try:
        audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)
        
        speech_prob = vad_model(audio_tensor, sr=RATE).mean().item()
        socketio.emit('speech_probability', {'probability': speech_prob})
        
        if speech_prob > 0.5:
            last_voice_time = time.time()
            audio_buffer.append(data)
            return False  # Not done processing
        else:
            if audio_buffer and (time.time() - last_voice_time) > silence_threshold:
                return True  # Done processing, silence detected
            return False  # Not done processing
    except Exception as e:
        print(f"Error processing audio data: {e}")
        return False

def transcribe_audio():
    global audio_buffer
    
    if not audio_buffer:
        return None
    
    try:
        recorded_audio = b"".join(audio_buffer)
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            filename = temp_file.name
        
        # Write audio data to the temporary file
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio_interface.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(recorded_audio)
        
        # Transcribe the audio
        segments, info = whisper_model.transcribe(filename, beam_size=5, language="ur")
        print(f"Detected language '{info.language}' with probability {info.language_probability}")
        
        transcribed_text = " ".join([segment.text for segment in segments])
        print(f"Transcribed Text: {transcribed_text}")
        
        # Clean up the temporary file
        os.unlink(filename)
        
        return transcribed_text
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None

def get_ai_response(text):
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=500,
            top_p=0.9,
            frequency_penalty=0,
            presence_penalty=0,
        )
        response_text = chat_completion.choices[0].message.content
        print(f"Groq Response: {response_text}")
        return response_text
    except Exception as e:
        print(f"Error getting AI response: {e}")
        return "Sorry, I couldn't process your request."

def play_audio_response(text):
    global is_processing
    
    try:
        socketio.emit('audio_playing')
        
        # Get audio bytes from ElevenLabs
        audio_response = tts_client.text_to_speech.convert(
            text=text,
            voice_id="JBFqnCBsd6RMkjVDRZzb",
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )
        
        # Convert the generator to bytes
        audio_bytes = b''
        for chunk in audio_response:
            audio_bytes += chunk
        
        # Save audio to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_file.write(audio_bytes)
            filename = temp_file.name
        
        # Play the audio using pydub
        sound = AudioSegment.from_mp3(filename)
        pydub_play(sound)
        
        # Clean up the temporary file
        os.unlink(filename)
        
    except Exception as e:
        print(f"Error playing audio response: {e}")
    finally:
        # Signal that we're done playing audio
        socketio.emit('audio_ended')
        
        # Reset processing flag to allow new audio input
        with processing_lock:
            is_processing = False
            print("Processing completed, ready for new input")

def process_voice_input():
    global is_processing, audio_buffer
    
    # Set processing flag to prevent new audio input
    with processing_lock:
        is_processing = True
        print("Starting audio processing, pausing voice detection")
    
    try:
        # Transcribe the audio
        transcribed_text = transcribe_audio()
        
        if transcribed_text and transcribed_text.strip():
            socketio.emit('transcribed_text', {'text': transcribed_text})
            
            # Get AI response
            response_text = get_ai_response(transcribed_text)
            socketio.emit('response', {'text': response_text})
            
            # Play audio response
            play_audio_response(response_text)
        else:
            # If no valid transcription, reset processing flag
            with processing_lock:
                is_processing = False
                print("No valid transcription, resuming voice detection")
        
        # Clear the buffer
        audio_buffer = []
    except Exception as e:
        print(f"Error in process_voice_input: {e}")
        # Reset processing flag in case of error
        with processing_lock:
            is_processing = False
            print("Error in processing, resuming voice detection")

def listening_thread_function():
    global is_listening, audio_buffer, is_processing
    
    print("Listening thread started")
    
    while is_listening:
        try:
            if stream and not is_processing:  # Only process audio if not already processing
                data = stream.read(CHUNK, exception_on_overflow=False)
                
                if process_audio_data(data):
                    print("Silence detected. Processing recorded audio...")
                    
                    # Process voice input in a separate thread
                    threading.Thread(target=process_voice_input).start()
            
            time.sleep(0.01)
        except Exception as e:
            print(f"Error in listening thread: {e}")
    
    print("Listening thread stopped")

@app.route('/process_text', methods=['POST'])
def process_text():
    global is_processing
    
    try:
        # Set processing flag to prevent new audio input
        with processing_lock:
            is_processing = True
            print("Starting text processing, pausing voice detection")
        
        data = request.json
        text = data.get('text', '')
        
        if not text:
            # Reset processing flag if no text
            with processing_lock:
                is_processing = False
            return jsonify({'error': 'No text provided'}), 400
        
        # Get AI response
        response_text = get_ai_response(text)
        
        # Play audio response in a separate thread
        threading.Thread(target=play_audio_response, args=(response_text,)).start()
        
        return jsonify({'response': response_text})
    except Exception as e:
        print(f"Error processing text: {e}")
        # Reset processing flag in case of error
        with processing_lock:
            is_processing = False
        return jsonify({'error': str(e)}), 500

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('start_listening')
def handle_start_listening():
    global is_listening, audio_buffer, is_processing
    
    if is_listening:
        return
    
    print("Starting listening...")
    
    # Initialize audio if not already done
    if not audio_interface or not stream:
        if not initialize_audio():
            socketio.emit('error', {'message': 'Failed to initialize audio'})
            return
    
    # Clear the buffer
    audio_buffer = []
    
    # Reset processing flag
    with processing_lock:
        is_processing = False
    
    # Start listening
    is_listening = True
    
    # Start the listening thread
    threading.Thread(target=listening_thread_function).start()

@socketio.on('stop_listening')
def handle_stop_listening():
    global is_listening
    
    if not is_listening:
        return
    
    print("Stopping listening...")
    
    # Stop listening
    is_listening = False

# Add a new event to check if system is ready for input
@socketio.on('check_ready')
def handle_check_ready():
    global is_processing
    socketio.emit('system_ready', {'ready': not is_processing})

if __name__ == '__main__':
    try:
        # Initialize audio
        initialize_audio()
        
        # Start the server
        print("Starting server on http://localhost:5000")
        socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("Server stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")
    finally:
        # Clean up resources
        cleanup_audio()