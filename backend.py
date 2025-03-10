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
import queue
import json  # for conversation history storage

# ----- Conversation History Setup -----
CONVERSATION_HISTORY_FILE = "conversation_history.json"
# Overwrite conversation history file on server start
with open(CONVERSATION_HISTORY_FILE, "w", encoding="utf-8") as f:
    json.dump([], f, ensure_ascii=False, indent=4)

# Initialize conversation history variables
conversation_history = []
history_lock = threading.Lock()
MAX_HISTORY_LENGTH = 5  # Keep last 5 exchanges (10 messages)

def save_conversation_history():
    """Save the current conversation history to a JSON file."""
    with open(CONVERSATION_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(conversation_history, f, ensure_ascii=False, indent=4)

def update_conversation_history(user_message, assistant_response):
    """Update conversation history and store it locally."""
    global conversation_history
    with history_lock:
        conversation_history.append({
            "role": "user",
            "content": user_message
        })
        conversation_history.append({
            "role": "assistant",
            "content": assistant_response
        })
        if len(conversation_history) > MAX_HISTORY_LENGTH * 2:
            conversation_history = conversation_history[-MAX_HISTORY_LENGTH * 2:]
        save_conversation_history()

# ----- Flask App Initialization -----
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load environment variables
load_dotenv()

# ----- System Prompt with Conversation History -----
system_prompt = f"""
You are an AI assistant designed to provide crucial agricultural guidance to wheat farmers. Your responses must be in Urdu, concise, and focused on wheat cultivation practices, including sowing times, irrigation schedules, seed varieties, weed control, disease prevention, and other farming advice. Tailor your responses to specific seasons, conditions, and geographical contexts to help farmers optimize wheat yields.

### Key Guidelines:
- **Sowing Time**: 1st to 15th November is optimal for wheat cultivation.  
- **Seed Quantity**: Use 50 kg/acre if sown by 15th November; increase to 60 kg/acre afterward.  
- **Land Preparation**: Use rotavators and double ploughing after clearing cotton, maize, or sugarcane fields.  
- **Seed Varieties**: Recommend GA 2002, Aqab 2000, NARC 2009 for rainfed areas; Aqab 2000, Punjab-1 for irrigated zones.  
- **Irrigation**: Perform "Herrio" twice after the first watering. In rainfed areas, deep ploughing helps retain rainwater.  
- **Pest Control**: Monitor for aphid attacks and use approved insecticides. Avoid spraying during fog, wind, or rain.  
- **Weed Management**: Use flat-fan nozzles for herbicide application.  

### Example Interaction:
**User Query (Urdu)**:  
"مجھے یہ بتاؤ کہ گندم میں کیڑے مارنے کے لیے کون سی زہر یوز کرنی چاہیے۔"  

**Your Response (Urdu)**:  
**گندم کے کیڑوں کے لیے سفارش کردہ زہر**:  
- **ایندوسلفن**: وسیع الطیف زہر، سست تیلے اور مکڑیوں کے خلاف مؤثر۔  
- **سیالوٹرن**: سست تیلے کے لیے بہترین۔  
- **ایمیٹاف**: تیزی سے اثر کرنے والا۔  
**ہدایات**: زہر کا استعمال ہدایت کے مطابق کریں۔ سپرے سے پہلے موسم کی پیشگوئی ضرور چیک کریں۔  

### Rules:
1. **Urdu Responses Only**: All answers must be in Urdu, using simple language for farmers.  
2. **Conciseness**: Keep responses under 4-5 lines unless details are critical.  
3. **Topic Enforcement**: Politely redirect users to agriculture-related queries if they deviate.  
4. **Urgent Issues**: Prioritize warnings (e.g., pest outbreaks, weather risks) in bold.

Historical conversation context:
{conversation_history}
"""

# ----- API Clients Initialization -----
try:
    tts_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY", "sk_47693bddc28209f1fddc944af4898cae6ffd4f14800c7525"))
    client = Groq(api_key=os.getenv("GROQ_API_KEY", "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"))
    print("API clients initialized successfully")
except Exception as e:
    print(f"Error initializing API clients: {e}")

# ----- VAD and Whisper Initialization -----
try:
    vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True)
    (get_speech_timestamps, _, _, _, _) = utils
    print("VAD model loaded successfully")
except Exception as e:
    print(f"Error loading VAD model: {e}")
    vad_model = None
    utils = None

try:
    model_size = "large-v3-turbo"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
    print(f"Whisper model loaded successfully on {device}")
except Exception as e:
    print(f"Error loading Whisper model: {e}")
    whisper_model = None

# ----- Audio Recording Parameters and Globals -----
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 512

audio_queue = queue.Queue(maxsize=10)  # Queue for audio segments
audio_buffer = []
last_voice_time = time.time()
silence_threshold = 2.0

audio_interface = None
stream = None

# Worker thread flag and mic state flag
is_listening = False
is_speaking = False  # Used to block mic input during TTS playback

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
    """
    Process the raw audio data with VAD.
    Append data to the audio_buffer if speech is detected.
    Return True when a full segment (i.e. silence following speech) is ready.
    """
    global last_voice_time, audio_buffer

    try:
        audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)

        speech_prob = vad_model(audio_tensor, sr=RATE).mean().item()
        socketio.emit('speech_probability', {'probability': speech_prob})

        if speech_prob > 0.5:
            last_voice_time = time.time()
            audio_buffer.append(data)
            return False  # Continue accumulating
        else:
            if audio_buffer and (time.time() - last_voice_time) > silence_threshold:
                return True  # Silence detected; segment complete
            return False  # Still in silence, but no segment to process yet
    except Exception as e:
        print(f"Error processing audio data: {e}")
        return False

def transcribe_audio(audio_segment_data):
    """
    Transcribe audio from a given audio segment.
    """
    try:
        # Write the audio segment to a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            filename = temp_file.name

        with wave.open(filename, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio_interface.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(audio_segment_data)

        segments, info = whisper_model.transcribe(filename, beam_size=5, language="ur")
        print(f"Detected language '{info.language}' with probability {info.language_probability}")

        transcribed_text = " ".join([segment.text for segment in segments])
        print(f"Transcribed Text: {transcribed_text}")

        os.unlink(filename)
        return transcribed_text
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None

def get_ai_response(text):
    """
    Get AI response using conversation history.
    The system prompt (which includes historical context) is sent along with the user message.
    """
    try:
        with history_lock:
            history = conversation_history.copy()
        messages = [{"role": "system", "content": system_prompt}, *history, {"role": "user", "content": text}]
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=500,
            top_p=0.9,
            frequency_penalty=0,
            presence_penalty=0,
        )
        response_text = chat_completion.choices[0].message.content
        print(f"Groq Response: {response_text}")
        update_conversation_history(text, response_text)
        return response_text
    except Exception as e:
        print(f"Error getting AI response: {e}")
        return "Sorry, I couldn't process your request."

def play_audio_response(text):
    """
    Play the audio response and block mic input during playback.
    """
    global is_speaking
    try:
        # Block mic input before starting playback
        is_speaking = True
        socketio.emit('mic_toggle_enabled', {'enabled': False})
        socketio.emit('audio_playing')
        
        audio_response = tts_client.text_to_speech.convert(
            text=text,
            voice_id="Sxk6njaoa7XLsAFT7WcN",
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )
        audio_bytes = b''.join(chunk for chunk in audio_response)
        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_file.write(audio_bytes)
            filename = temp_file.name
        
        sound = AudioSegment.from_mp3(filename)
        pydub_play(sound)
        
        os.unlink(filename)
    except Exception as e:
        print(f"Error playing audio response: {e}")
    finally:
        # Re-enable mic input after playback
        is_speaking = False
        socketio.emit('mic_toggle_enabled', {'enabled': True})
        socketio.emit('audio_ended')

def process_voice_input(audio_segment):
    """
    Process a single complete audio segment:
    transcribe, get AI response, and play the response.
    """
    try:
        transcribed_text = transcribe_audio(audio_segment)
        if transcribed_text and transcribed_text.strip():
            socketio.emit('transcribed_text', {'text': transcribed_text})
            response_text = get_ai_response(transcribed_text)
            socketio.emit('response', {'text': response_text})
            play_audio_response(response_text)
        else:
            print("No valid transcription; skipping processing for this segment.")
    except Exception as e:
        print(f"Error processing voice input: {e}")

def audio_processing_worker():
    """
    Worker thread that processes audio segments from the queue.
    """
    while True:
        audio_segment = audio_queue.get()  # This blocks until an item is available
        if audio_segment is None:
            break  # Use None as a sentinel to shut down the worker
        process_voice_input(audio_segment)
        audio_queue.task_done()

def listening_thread_function():
    """
    Continuously capture audio, run VAD, and enqueue complete audio segments.
    """
    global is_listening, audio_buffer
    print("Listening thread started")
    while is_listening:
        try:
            if stream:
                data = stream.read(CHUNK, exception_on_overflow=False)
                if process_audio_data(data):
                    print("Silence detected. Enqueuing recorded audio segment...")
                    recorded_audio = b"".join(audio_buffer)
                    
                    if not audio_queue.full():
                        audio_queue.put(recorded_audio)
                    else:
                        print("Audio queue full. Dropping segment.")
                    
                    audio_buffer = []
            time.sleep(0.01)
        except Exception as e:
            print(f"Error in listening thread: {e}")
    print("Listening thread stopped")

# ----- Flask Endpoints and SocketIO Events -----
@app.route('/process_text', methods=['POST'])
def process_text():
    try:
        data = request.json
        text = data.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        response_text = get_ai_response(text)
        threading.Thread(target=play_audio_response, args=(response_text,)).start()
        return jsonify({'response': response_text})
    except Exception as e:
        print(f"Error processing text: {e}")
        return jsonify({'error': str(e)}), 500

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('start_listening')
def handle_start_listening():
    global is_listening, audio_buffer
    if is_listening:
        return

    print("Starting listening...")
    if not audio_interface or not stream:
        if not initialize_audio():
            socketio.emit('error', {'message': 'Failed to initialize audio'})
            return

    audio_buffer = []
    is_listening = True
    threading.Thread(target=listening_thread_function).start()

@socketio.on('stop_listening')
def handle_stop_listening():
    global is_listening
    if not is_listening:
        return
    print("Stopping listening...")
    is_listening = False

@socketio.on('check_ready')
def handle_check_ready():
    socketio.emit('system_ready', {'ready': True})

if __name__ == '__main__':
    try:
        initialize_audio()
        # Start the audio processing worker thread
        worker_thread = threading.Thread(target=audio_processing_worker, daemon=True)
        worker_thread.start()

        print("Starting server on http://localhost:5000")
        socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("Server stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")
    finally:
        # Signal the worker thread to exit and clean up
        audio_queue.put(None)
        cleanup_audio()
