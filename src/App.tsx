import React, { useState, useEffect } from 'react';
import {
  Mic,
  MicOff,
  Volume2,
  Leaf,
  Wheat,
  Info,
  Settings,
  History,
} from 'lucide-react';
import { io } from 'socket.io-client';
import axios from 'axios';

// Initialize socket connection
const socket = io('http://localhost:5000');

function App() {
  const [isListening, setIsListening] = useState(false);
  const [speechProbability, setSpeechProbability] = useState(0);
  const [transcribedText, setTranscribedText] = useState('');
  const [responseText, setResponseText] = useState('');
  const [isPlaying, setIsPlaying] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [conversationHistory, setConversationHistory] = useState<
    { question: string; answer: string }[]
  >([]);
  const [activeTab, setActiveTab] = useState('main');
  const [systemReady, setSystemReady] = useState(true);

  // Connect to backend
  useEffect(() => {
    // Socket connection events
    socket.on('connect', () => {
      setConnectionStatus('connected');
      addLog('Connected to server');
    });

    socket.on('disconnect', () => {
      setConnectionStatus('disconnected');
      addLog('Disconnected from server');
    });

    socket.on('speech_probability', (data) => {
      setSpeechProbability(data.probability);
    });

    socket.on('transcribed_text', (data) => {
      setTranscribedText(data.text);
      addLog(`Transcribed Text: ${data.text}`);
      setSystemReady(false);
    });

    socket.on('response', (data) => {
      setResponseText(data.text);
      addLog(`Response: ${data.text}`);

      // Add to conversation history
      setConversationHistory((prev) => [
        ...prev,
        {
          question: transcribedText,
          answer: data.text,
        },
      ]);
    });

    socket.on('audio_playing', () => {
      setIsPlaying(true);
      setSystemReady(false);
      addLog('Playing audio response');
    });

    socket.on('audio_ended', () => {
      setIsPlaying(false);
      setSystemReady(true);
      addLog('Audio playback ended');

      // Check if system is ready for new input
      socket.emit('check_ready');
    });

    socket.on('system_ready', (data) => {
      setSystemReady(data.ready);
    });

    // Cleanup on component unmount
    return () => {
      socket.off('connect');
      socket.off('disconnect');
      socket.off('speech_probability');
      socket.off('transcribed_text');
      socket.off('response');
      socket.off('audio_playing');
      socket.off('audio_ended');
      socket.off('system_ready');
    };
  }, [transcribedText]);

  // Periodically check if system is ready
  useEffect(() => {
    const readyCheckInterval = setInterval(() => {
      if (connectionStatus === 'connected' && !isPlaying) {
        socket.emit('check_ready');
      }
    }, 2000);

    return () => clearInterval(readyCheckInterval);
  }, [connectionStatus, isPlaying]);

  const toggleListening = () => {
    if (connectionStatus === 'disconnected') {
      addLog('Cannot start listening: Not connected to server');
      return;
    }

    if (!systemReady) {
      addLog('System is currently processing. Please wait.');
      return;
    }

    if (!isListening) {
      // Start listening
      socket.emit('start_listening');
      setIsListening(true);
      addLog('Started listening...');
      setTranscribedText('');
      setResponseText('');
    } else {
      // Stop listening
      socket.emit('stop_listening');
      setIsListening(false);
      addLog('Stopped listening');
    }
  };

  const addLog = (message: string) => {
    setLogs((prev) => [
      ...prev,
      `[${new Date().toLocaleTimeString()}] ${message}`,
    ]);
  };

  // Function to manually test with a sample question
  const testWithSample = () => {
    if (!systemReady) {
      addLog('System is currently processing. Please wait.');
      return;
    }

    const sampleQuestion =
      'مجھے یہ بتاؤ کہ گندم میں کیڑے مارنے کے لیے کون سی زہر یوز کرنی چاہیے۔';
    setTranscribedText(sampleQuestion);
    addLog(`Test Question: ${sampleQuestion}`);
    setSystemReady(false);

    // Send to backend
    axios
      .post('http://localhost:5000/process_text', { text: sampleQuestion })
      .then((response) => {
        setResponseText(response.data.response);
        addLog(`Response: ${response.data.response}`);

        // Add to conversation history
        setConversationHistory((prev) => [
          ...prev,
          {
            question: sampleQuestion,
            answer: response.data.response,
          },
        ]);
      })
      .catch((error) => {
        addLog(`Error: ${error.message}`);
        setSystemReady(true);
      });
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-green-50 to-green-100 flex flex-col">
      {/* Header */}
      <header className="bg-gradient-to-r from-green-700 to-green-600 text-white p-4 shadow-md">
        <div className="container mx-auto flex items-center justify-between">
          <div className="flex items-center">
            <Wheat size={32} className="mr-3" />
            <div>
              <h1 className="text-3xl font-bold">AgriBot</h1>
              <p className="text-sm opacity-80">Urdu Wheat Farming Assistant</p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <span
              className={`inline-block w-3 h-3 rounded-full ${
                connectionStatus === 'connected' ? 'bg-green-300' : 'bg-red-500'
              }`}
            ></span>
            <span className="text-sm">
              {connectionStatus === 'connected' ? 'Connected' : 'Disconnected'}
            </span>
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="bg-green-600 text-white shadow-md">
        <div className="container mx-auto flex">
          <button
            onClick={() => setActiveTab('main')}
            className={`px-6 py-3 flex items-center text-lg transition-colors duration-200 ${
              activeTab === 'main' ? 'bg-green-800' : 'hover:bg-green-700'
            }`}
          >
            <Mic size={20} className="mr-2" /> Main
          </button>
          <button
            onClick={() => setActiveTab('history')}
            className={`px-6 py-3 flex items-center text-lg transition-colors duration-200 ${
              activeTab === 'history' ? 'bg-green-800' : 'hover:bg-green-700'
            }`}
          >
            <History size={20} className="mr-2" /> History
          </button>
          <button
            onClick={() => setActiveTab('info')}
            className={`px-6 py-3 flex items-center text-lg transition-colors duration-200 ${
              activeTab === 'info' ? 'bg-green-800' : 'hover:bg-green-700'
            }`}
          >
            <Info size={20} className="mr-2" /> About
          </button>
        </div>
      </nav>

      <div className="flex flex-1 overflow-hidden">
        {/* Main content */}
        <main className="flex-1 p-6 container mx-auto">
          {activeTab === 'main' && (
            <>
              {/* Speech visualization */}
              <div className="bg-white rounded-lg shadow-lg p-8 mb-8 flex flex-col items-center border-l-4 border-green-500 transform hover:scale-[1.01] transition-transform duration-300">
                <div className="flex items-center justify-center mb-6">
                  <button
                    onClick={toggleListening}
                    className={`p-6 rounded-full ${
                      !systemReady
                        ? 'bg-gray-400 cursor-not-allowed'
                        : isListening
                        ? 'bg-red-500 hover:bg-red-600'
                        : 'bg-green-600 hover:bg-green-700'
                    } text-white transition-all duration-300 shadow-lg hover:shadow-xl transform hover:scale-105`}
                    disabled={
                      connectionStatus === 'disconnected' || !systemReady
                    }
                  >
                    {isListening ? <MicOff size={32} /> : <Mic size={32} />}
                  </button>
                </div>

                <div className="w-full mb-5">
                  <div className="text-base text-gray-600 mb-2 flex justify-between">
                    <span>Speech Probability</span>
                    <span>{(speechProbability * 100).toFixed(2)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-3">
                    <div
                      className={`h-3 rounded-full transition-all duration-300 ${
                        speechProbability > 0.5 ? 'bg-green-500' : 'bg-gray-400'
                      }`}
                      style={{ width: `${speechProbability * 100}%` }}
                    ></div>
                  </div>
                </div>

                <div className="flex items-center gap-3 text-base text-gray-600">
                  <span>Status:</span>
                  {!systemReady ? (
                    <span className="text-yellow-500 font-medium flex items-center gap-2">
                      <span className="relative flex h-3 w-3">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-yellow-400 opacity-75"></span>
                        <span className="relative inline-flex rounded-full h-3 w-3 bg-yellow-500"></span>
                      </span>
                      Processing
                    </span>
                  ) : isListening ? (
                    <span className="text-green-500 font-medium flex items-center gap-2">
                      <span className="relative flex h-3 w-3">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                        <span className="relative inline-flex rounded-full h-3 w-3 bg-green-500"></span>
                      </span>
                      Listening
                    </span>
                  ) : (
                    <span className="text-gray-600 font-medium">Ready</span>
                  )}

                  {isPlaying && (
                    <span className="text-green-600 font-medium flex items-center gap-2 ml-4">
                      <Volume2 size={18} className="animate-pulse" />
                      Playing Audio
                    </span>
                  )}
                </div>

                {connectionStatus === 'disconnected' && (
                  <div className="mt-5 text-red-500 text-base bg-red-50 p-3 rounded-md border border-red-200">
                    Backend server not connected. Make sure the Python backend
                    is running.
                  </div>
                )}

                <div className="mt-6">
                  <button
                    onClick={testWithSample}
                    className={`px-5 py-2.5 ${
                      !systemReady
                        ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                        : 'bg-green-100 text-green-800 hover:bg-green-200'
                    } rounded-md border border-green-300 text-base font-medium transition-colors duration-200`}
                    disabled={!systemReady}
                  >
                    Test with Sample Question
                  </button>
                </div>
              </div>

              {/* Transcription and Response */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
                <div className="bg-white rounded-lg shadow-lg p-6 border-l-4 border-green-400 transform hover:scale-[1.01] transition-transform duration-300">
                  <h2 className="text-xl font-semibold mb-4 text-green-800 flex items-center">
                    <Mic size={20} className="mr-2" /> Transcribed Text
                  </h2>
                  <div
                    className="min-h-40 p-5 bg-green-50 rounded-md border border-green-200 text-right text-2xl gulzar-regular"
                    dir="rtl"
                    style={{ lineHeight: '2' }}
                  >
                    {transcribedText || (
                      <span className="text-gray-400 italic">
                        {isListening
                          ? 'Listening for speech...'
                          : 'Click the microphone button to start'}
                      </span>
                    )}
                  </div>
                </div>

                <div className="bg-white rounded-lg shadow-lg p-6 border-l-4 border-green-400 transform hover:scale-[1.01] transition-transform duration-300">
                  <h2 className="text-xl font-semibold mb-4 text-green-800 flex items-center">
                    <Volume2 size={20} className="mr-2" /> AI Response
                  </h2>
                  <div
                    className="min-h-40 p-5 bg-green-50 rounded-md border border-green-200 text-right text-2xl gulzar-regular"
                    dir="rtl"
                    style={{ lineHeight: '2' }}
                  >
                    {responseText || (
                      <span className="text-gray-400 italic">
                        Waiting for transcription...
                      </span>
                    )}
                  </div>
                </div>
              </div>

              {/* System Logs */}
              <div className="bg-white rounded-lg shadow-lg p-6 flex-1 overflow-hidden border-l-4 border-green-500 transform hover:scale-[1.01] transition-transform duration-300">
                <h2 className="text-xl font-semibold mb-4 text-green-800">
                  System Logs
                </h2>
                <div className="bg-gray-900 text-green-400 p-4 rounded-md font-mono text-sm h-48 overflow-y-auto">
                  {logs.length === 0 ? (
                    <div className="text-gray-500">
                      No logs yet. Start the system to see activity.
                    </div>
                  ) : (
                    logs.map((log, index) => (
                      <div key={index} className="mb-1">
                        {log}
                      </div>
                    ))
                  )}
                </div>
              </div>
            </>
          )}

          {activeTab === 'history' && (
            <div className="bg-white rounded-lg shadow-lg p-8 border-l-4 border-green-500 transform hover:scale-[1.01] transition-transform duration-300">
              <h2 className="text-2xl font-semibold mb-6 text-green-800 flex items-center">
                <History size={24} className="mr-3" /> Conversation History
              </h2>

              {conversationHistory.length === 0 ? (
                <div className="text-gray-500 p-6 bg-green-50 rounded-md text-center text-lg">
                  No conversation history yet. Start talking to AgriBot to see
                  your conversations here.
                </div>
              ) : (
                <div className="space-y-8">
                  {conversationHistory.map((item, index) => (
                    <div key={index} className="border-b border-green-100 pb-6">
                      <div className="mb-4">
                        <div className="font-medium text-green-800 mb-2 text-lg">
                          Question:
                        </div>
                        <div
                          className="p-4 bg-green-50 rounded-md text-right text-2xl gulzar-regular"
                          dir="rtl"
                          style={{ lineHeight: '2' }}
                        >
                          {item.question}
                        </div>
                      </div>
                      <div>
                        <div className="font-medium text-green-800 mb-2 text-lg">
                          Answer:
                        </div>
                        <div
                          className="p-4 bg-green-50 rounded-md text-right text-2xl gulzar-regular"
                          dir="rtl"
                          style={{ lineHeight: '2' }}
                        >
                          {item.answer}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {activeTab === 'info' && (
            <div className="bg-white rounded-lg shadow-lg p-8 border-l-4 border-green-500 transform hover:scale-[1.01] transition-transform duration-300">
              <h2 className="text-2xl font-semibold mb-6 text-green-800 flex items-center">
                <Info size={24} className="mr-3" /> About AgriBot
              </h2>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div className="bg-green-50 p-6 rounded-md border border-green-200">
                  <h3 className="text-xl font-medium mb-4 text-green-700 flex items-center">
                    <Leaf size={22} className="mr-2" /> What is AgriBot?
                  </h3>
                  <p className="text-gray-700 mb-5 text-lg leading-relaxed">
                    AgriBot is an AI-powered voice assistant designed
                    specifically for wheat farmers in Pakistan. It provides
                    expert advice on wheat cultivation, pest control,
                    irrigation, and other farming practices in Urdu language.
                  </p>

                  <h3 className="text-xl font-medium mb-4 text-green-700">
                    System Components
                  </h3>
                  <ul className="list-disc pl-6 text-gray-700 space-y-2 mb-4 text-lg">
                    <li>
                      <span className="font-medium">Speech Recognition:</span>{' '}
                      Whisper Model for accurate Urdu speech recognition
                    </li>
                    <li>
                      <span className="font-medium">
                        Voice Activity Detection:
                      </span>{' '}
                      Silero VAD for detecting when someone is speaking
                    </li>
                    <li>
                      <span className="font-medium">AI Processing:</span> LLaMA 3.3 70B model for generating responses
                    </li>
                    <li>
                      <span className="font-medium">Text-to-Speech:</span>{' '}
                      ElevenLabs for natural-sounding Urdu voice responses
                    </li>
                  </ul>
                </div>

                <div className="bg-green-50 p-6 rounded-md border border-green-200">
                  <h3 className="text-xl font-medium mb-4 text-green-700">
                    How It Works
                  </h3>
                  <ol className="list-decimal pl-6 text-gray-700 space-y-3 mb-5 text-lg">
                    <li>Click the microphone button to start listening</li>
                    <li>Speak your question about wheat farming in Urdu</li>
                    <li>The system detects when you've finished speaking</li>
                    <li>Your speech is transcribed to text</li>
                    <li>The AI generates a helpful response</li>
                    <li>The response is converted to speech and played back</li>
                  </ol>

                  <h3 className="text-xl font-medium mb-4 text-green-700">
                    Getting Started
                  </h3>
                  <p className="text-gray-700 text-lg leading-relaxed">
                    Make sure the Python backend is running before using
                    AgriBot. You can ask questions about:
                  </p>
                  <ul className="list-disc pl-6 text-gray-700 space-y-2 mt-3 text-lg">
                    <li>Wheat planting times and techniques</li>
                    <li>Seed varieties and quantities</li>
                    <li>Irrigation schedules</li>
                    <li>Pest control and disease prevention</li>
                    <li>Harvesting best practices</li>
                  </ul>
                </div>
              </div>
            </div>
          )}
        </main>
      </div>

      {/* Footer */}
      <footer className="bg-green-800 text-white p-4 text-center">
        <p className="text-base">© 2025 AgriBot - Wheat Farming Assistant | Powered by AI</p>
      </footer>
    </div>
  );
}

export default App;