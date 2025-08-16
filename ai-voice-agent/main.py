import traceback
import os
import tempfile
import time
import requests
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import google.generativeai as genai
from typing import Dict, List

from services.tts_murf import generate_tts_with_murf # type: ignore

from fastapi import WebSocket

# Load ENV & Configure APIs
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ASSEMBLY_API_KEY = os.getenv("ASSEMBLY_API_KEY")
MURF_API_KEY = os.getenv("MURF_API_KEY")

if not GEMINI_API_KEY or not ASSEMBLY_API_KEY or not MURF_API_KEY:
    raise RuntimeError("❗ Missing one or more API keys in environment variables")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# Chat history file
CHAT_HISTORY_FILE = "chat_history.json"

# Load chat history from JSON file
def load_chat_history() -> Dict[str, List[dict]]:
    try:
        with open(CHAT_HISTORY_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

# Save chat history to JSON file
def save_chat_history(history: Dict[str, List[dict]]):
    with open(CHAT_HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

# Initialize chat history
chat_history: Dict[str, List[dict]] = load_chat_history()

# FastAPI App
app = FastAPI(title="30-Days-of-AI-Voice-Agents – Day 10")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve Static Files
static_path = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_path), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse(static_path / "index.html")

FALLBACK_MESSAGE="I'm having trouble connecting right now."

# Core LLM Endpoint (handles both text & audio)
@app.post("/llm/query")
async def query_llm(
    text: str = Form(None),
    file: UploadFile = File(None)
):
    """
    Handles either:
    1. Text input
    2. Audio input (transcribed via AssemblyAI)
    """
    try:
        # Case 1: If audio file is provided
        if file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(await file.read())
                tmp_path = tmp.name

            # Upload to AssemblyAI
            try:
                headers = {"authorization": ASSEMBLY_API_KEY}
                upload_url = "https://api.assemblyai.com/v2/uploa"
                with open(tmp_path, "rb") as f:
                    upload_resp = requests.post(upload_url, headers=headers, data=f)
                audio_url = upload_resp.json()["upload_url"]
            except Exception as e:
                print("upload failed due to=> ", e)
                return {"services": FALLBACK_MESSAGE}

            # Start transcription
            try:
                transcript_url = "https://api.assemblyai.com/v2/transcript"
                transcript_req = {"audio_url": audio_url}
                transcript_resp = requests.post(transcript_url, json=transcript_req, headers=headers)
                transcript_id = transcript_resp.json()["id"]
            except Exception as e:
                print("transcription failed due to=> ",e)
                return {"services": FALLBACK_MESSAGE}

            # Poll until complete
            while True:
                poll_resp = requests.get(f"{transcript_url}/{transcript_id}", headers=headers)
                status = poll_resp.json()["status"]
                if status == "completed":
                    text = poll_resp.json()["text"]
                    break
                elif status == "error":
                    raise HTTPException(status_code=500, detail="Transcription failed")
                time.sleep(2)

        if not text:
            raise HTTPException(status_code=400, detail="No text or audio provided")

        # Step 2: Send text to Gemini
        try:
            gemini_resp = gemini_model.generate_content(text)
            ai_text = gemini_resp.text.strip()
        except Exception as e:
            print("error from gemini ai due to => ", e)
            return {"services": FALLBACK_MESSAGE}

        # Step 3: Send AI text to Murf for TTS
        murf_audio_url = generate_tts_with_murf(ai_text)
        return {
            "transcribed_text": text if file else None,
            "ai_text": ai_text,
            "audio_url": murf_audio_url
        }

    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/agent/chat/{session_id}")
async def chat_with_history(
    session_id: str,
    text: str = Form(None),
    file: UploadFile = File(None)
):
    """
    Handles conversational chat with history using session_id.
    1. Transcribe audio if provided
    2. Fetch and append to chat history
    3. Query LLM with combined history and new input
    4. Store response in history
    5. Return TTS audio
    """
    try:
        # Initialize chat history for session if not exists
        if session_id not in chat_history:
            chat_history[session_id] = []

        # Step 1: Transcribe audio if provided
        if file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(await file.read())
                tmp_path = tmp.name

            # Upload to AssemblyAI
            headers = {"authorization": ASSEMBLY_API_KEY}
            upload_url = "https://api.assemblyai.com/v2/upload"
            with open(tmp_path, "rb") as f:
                upload_resp = requests.post(upload_url, headers=headers, data=f)
            audio_url = upload_resp.json()["upload_url"]

            # Start transcription
            transcript_url = "https://api.assemblyai.com/v2/transcript"
            transcript_req = {"audio_url": audio_url}
            transcript_resp = requests.post(transcript_url, json=transcript_req, headers=headers)
            transcript_id = transcript_resp.json()["id"]

            # Poll until complete
            while True:
                poll_resp = requests.get(f"{transcript_url}/{transcript_id}", headers=headers)
                status = poll_resp.json()["status"]
                if status == "completed":
                    text = poll_resp.json()["text"]
                    break
                elif status == "error":
                    raise HTTPException(status_code=500, detail="Transcription failed")
                time.sleep(2)

        if not text:
            raise HTTPException(status_code=400, detail="No text or audio provided")

        # Step 2: Fetch chat history and append new user message
        history = chat_history[session_id]
        history.append({"role": "user", "content": text})
        save_chat_history(chat_history)

        # Step 3: Combine history and new message for LLM
        conversation = ""
        for message in history:
            role = "User" if message["role"] == "user" else "Assistant"
            conversation += f"{role}: {message['content']}\n"
        conversation += "Assistant: "

        # Step 4: Query Gemini with conversation history
        gemini_resp = gemini_model.generate_content(conversation)
        ai_text = gemini_resp.text.strip()

        # Step 5: Append assistant response to history
        history.append({"role": "assistant", "content": ai_text})
        save_chat_history(chat_history)

        # Step 6: Generate TTS for assistant response
        murf_audio_url = generate_tts_with_murf(ai_text)

        return {
            "transcribed_text": text if file else None,
            "ai_text": ai_text,
            "audio_url": murf_audio_url,
            "session_id": session_id
        }

    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))
    
@app.get("/agent/history/{session_id}")
async def get_history(session_id: str):
    return chat_history.get(session_id, [])

# websocket endpoint
@app.websocket("/ws")
async def ws_endpoint(websocket:WebSocket):
    """
    Simple WebSocket Echo Server
    1. Accepts connection
    2. Receives message from client
    3. Echoes message back
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            print(f"📩Received: {data}")
            await websocket.send_text(f"Echo: {data}")
    except Exception as e:
        print("⚠️ WebSocket connection closed:", e)