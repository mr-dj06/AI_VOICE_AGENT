# 🎙️ 30-Days-of-AI-Voice-Agents – Day 10  
AI Voice Conversational Agent using **FastAPI**, **Gemini AI**, **AssemblyAI**, and **Murf TTS**  

This project is part of the "30 Days of AI Voice Agents" challenge.  
It provides an API for **conversational AI** that can take **text or audio input**, transcribe audio to text, process it with Google’s Gemini LLM, and respond back with **text + generated voice** using Murf TTS.  
It also includes **chat history persistence** for multi-turn conversations.

---

## ✨ Features
- 🎤 **Speech-to-Text**: Converts audio to text using [AssemblyAI](https://www.assemblyai.com/)  
- 🤖 **AI Chat**: Uses [Gemini AI](https://ai.google/) for natural language responses  
- 🔊 **Text-to-Speech**: Converts AI responses to speech using [Murf API](https://murf.ai/)  
- 💬 **Chat History**: Maintains session-based conversation context  
- 🌐 **Cross-Origin Support**: Built-in CORS for frontend integration  
- 📂 **Static File Serving**: Serves an HTML UI for testing

---

## 🛠️ Tech Stack
- **Backend**: [FastAPI](https://fastapi.tiangolo.com/)
- **Speech-to-Text**: [AssemblyAI API](https://www.assemblyai.com/)
- **LLM**: [Google Generative AI (Gemini)](https://ai.google/)
- **Text-to-Speech**: [Murf API](https://murf.ai/)
- **Data Storage**: JSON file for chat history

---

## 📂 Project Structure
```
.
├── main.py                # FastAPI server
├── tts_murf.py            # Murf TTS integration helper
├── static/
│   └── index.html         # Frontend UI
├── chat_history.json      # Chat history storage
├── .env                   # API keys
├── requirements.txt       # requirements
└── README.md
```

---

## 🔑 Environment Variables
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_gemini_api_key_here
ASSEMBLY_API_KEY=your_assemblyai_api_key_here
MURF_API_KEY=your_murf_api_key_here
```

---

## 📦 Installation
```bash
# Clone repository
git clone https://github.com/mr-dj06/AI_VOICE_AGENT.git
cd ai-voice-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate     # For Windows

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Running the Server
```bash
uvicorn main:app --reload
```
Server will start at:  
```
http://127.0.0.1:8000
```

---

## 📌 API Endpoints

### 1️⃣ **Query LLM (Single-turn)**  
`POST /llm/query`  
**Form Data**:
- `text`: (string) Optional text input  
- `file`: (audio file) Optional audio input (WAV format preferred)

**Response**:
```json
{
  "transcribed_text": "Hello there",
  "ai_text": "Hi! How can I help you today?",
  "audio_url": "https://murf.ai/output/audio-file.mp3"
}
```

---

### 2️⃣ **Conversational Chat (Multi-turn)**  
`POST /agent/chat/{session_id}`  
Maintains chat history for a given `session_id`.

**Form Data**:
- `text`: (string) Optional
- `file`: (audio file) Optional

**Response**:
```json
{
  "transcribed_text": "What's the weather today?",
  "ai_text": "Today's weather is sunny with a high of 30°C.",
  "audio_url": "https://murf.ai/output/weather.mp3",
  "session_id": "abc123"
}
```

---

### 3️⃣ **Get Chat History**  
`GET /agent/history/{session_id}`

**Response**:
```json
[
  { "role": "user", "content": "Hello" },
  { "role": "assistant", "content": "Hi there!" }
]
```

---

## 🗄️ Chat History
- Stored in `chat_history.json` in the root directory.
- Automatically created if it doesn’t exist.

---

## 🛠️ Helper Function – `tts_murf.py`

This file contains the `generate_tts_with_murf(text, voice_id=None)` function, which uses the **Murf API** to convert AI-generated text into speech.  

### Parameters:
- **text** *(str)*: The text to convert to speech  
- **voice_id** *(str, optional)*: Murf voice ID (defaults to `"en-IN-aarav"`)  

### Returns:
- **audio_url** *(str)*: The URL of the generated audio file from Murf

### Example:
```python
from tts_murf import generate_tts_with_murf

audio_url = generate_tts_with_murf("Hello world!")
print("Audio available at:", audio_url)
```

---

## ⚠️ Notes
- AssemblyAI **upload URL** must be correct (`/v2/upload`), ensure it’s not truncated.
- Audio files should be in **WAV format** for best transcription results.
- The `tts_murf.py` function must exist for TTS features to work.

---

## 📜 License
All rights reserved.
© 2025 Darshan